import glob
import os
import os.path as osp
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib

from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide

from .conversation import get_default_conv_template
from .data_processing import get_mask_from_json
from .reason_seg_dataset import ReasonSegDataset
from .refer import REFER
from .refer_seg_dataset import ReferSegDataset
from .sem_seg_dataset import SemSegDataset
from .utils import (
    DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX, 
    convert2imagesplit, UNIFIED_SHORT_QUESTION_LIST, UNIFIED_LONG_QUESTION_LIST
)
from .vqa_dataset import VQADataset
from .chatunivi_dataset import ChatUniviDataset
from .rvos_dataset import RVOSDataset
from .random_list import get_random_list
from .dataset_config import LISA_ROOT


def collate_fn(
    batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1
):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    num_frame_list = []
    num_conv_list = []
    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        inference,
    ) in batch:
        image_path_list.append(image_path)

        if images.ndim == 3:
            images = images.unsqueeze(0)
        assert images.ndim == 4
        images_list.append(images)

        if images_clip.ndim == 3:
            images_clip = images_clip.unsqueeze(0)
        assert images_clip.ndim == 4
        images_clip_list.append(images_clip)
        num_frame = images_clip.shape[0]
        num_frame_list.append(num_frame)

        conversation_list.extend(conversations)
        label_list.append(label)
        num_conv_list.append(len(conversations))


        if masks.ndim == 3:  # [num_classes, H, W]
            if masks.shape[0] == 0:  # [0, H, W] -> [num_classes, 0, H, W]
                masks = torch.stack([masks, ] * len(conversations), dim=0).float()
            else: # [num_classes, H, W] -> [num_classes, 1, H, W]
                masks = masks.unsqueeze(1).float()
        assert masks.ndim == 4
        masks_list.append(masks.float())

        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)

        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        # replace <image> token
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IMAGE_TOKEN
            replace_token = (
                DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            )
            conversation_list[i] = conversation_list[i].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

    for i in range(len(conversation_list)):
        if DEFAULT_VIDEO_TOKEN in conversation_list[i]:
            if conversation_list[i].count(DEFAULT_VIDEO_TOKEN) == 1:
                replace_video_token = DEFAULT_IMAGE_TOKEN * num_frame
                conversation_list[i] = conversation_list[i].replace(DEFAULT_VIDEO_TOKEN, replace_video_token)
            else:
                raise ValueError("num video token > 1: ", conversation_list[i].count(DEFAULT_VIDEO_TOKEN))


    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversation_list
    ]
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()

    if conv_type == "llava_v1":
        sep = conv.sep + conv.roles[1] + ": "
    else:
        sep = "[/INST] "
    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            # if len(parts) != 2:
            #     break
            assert len(parts) == 2, (len(parts), rou)
            parts[0] += sep

            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            assert cur_len == total_len

    return {
        "image_paths": image_path_list,
        "images": images_list, #BS : T(or 1 for chatunivi) * 3 * H * W
        "images_clip": images_clip_list, #BS : T * 3 * H * W
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list, # [Conv*Frame*H*W, ...]
        "label_list": label_list, # [H*W, ...]
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list), #[0, num_conv0, num_conv1, ...]
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
        "num_frame_list": num_frame_list,
        "num_conv_list": num_conv_list,
    }


class HybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        dataset="sem_seg||refer_seg||vqa||reason_seg",
        sample_rate=[9, 3, 3, 1],
        sem_seg_data="ade20k||cocostuff||partimagenet||pascal_part||paco_lvis||mapillary",
        refer_seg_data="refclef||refcoco||refcoco+||refcocog",
        vqa_data="llava_instruct_150k",
        reason_seg_data="ReasonSeg|train",
        rvos_seg_data="mevis_train||refytvos_train||davis17_train",
        rvos_sample_ratio='4000||15000||400',
        rvos_num_frames_sample_range="6,12",
        rvos_sample_policy="uniform",
        univi_data_list = "mimic||sqa||video",
        univi_data_ratio = "1||1||1",
        univi_max_image_len = 64,
        explanatory=0.1,
        univi_sample_frame_range="10,12",
        balance_sample=True,
    ):
        self.exclude_val = exclude_val
        self.dataset = dataset
        self.samples_per_epoch = samples_per_epoch
        self.explanatory = explanatory
        self.num_classes_per_sample = num_classes_per_sample
        sample_rate = np.array(sample_rate)
        self.sample_rate = sample_rate / sample_rate.sum()

        self.base_image_dir = base_image_dir = LISA_ROOT
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision

        self.datasets = dataset.split("||")
        self.num_datasets = len(self.datasets)

        self.num_be_called = 0
        if balance_sample:
            self.dataset_sample_list = get_random_list(probabilities=self.sample_rate.tolist(), values=list(range(self.num_datasets)), length=samples_per_epoch)
            chatunivi_sample_range = [int(i) for i in univi_sample_frame_range.split(',')]
            chatunivi_range_length = chatunivi_sample_range[-1] - chatunivi_sample_range[0] + 1
            self.chatunivi_sample_list = get_random_list(probabilities=[float(1/chatunivi_range_length) for _ in range(chatunivi_range_length)], values=list(range(chatunivi_sample_range[0],chatunivi_sample_range[-1]+1)), length=10000)
            rvos_sample_range = [int(i) for i in rvos_num_frames_sample_range.split(',')]
            rvos_range_length = rvos_sample_range[-1] - rvos_sample_range[0] + 1
            self.rvos_sample_list = get_random_list(probabilities=[float(1/rvos_range_length) for _ in range(rvos_range_length)], values=list(range(rvos_sample_range[0],rvos_sample_range[-1]+1)), length=10000)
        else:
            self.dataset_sample_list = None
            self.chatunivi_sample_list = []
            self.rvos_sample_list = []
        

        self.all_datasets = []
        for dataset in self.datasets:
            if dataset == "sem_seg":
                self.all_datasets.append(
                    SemSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        sem_seg_data,
                    )
                )
            elif dataset == "refer_seg":
                self.all_datasets.append(
                    ReferSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        refer_seg_data,
                    )
                )
            elif dataset == "vqa":
                self.all_datasets.append(
                    VQADataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        vqa_data,
                    )
                )
            elif dataset == "reason_seg":
                self.all_datasets.append(
                    ReasonSegDataset(
                        base_image_dir,
                        tokenizer,
                        vision_tower,
                        samples_per_epoch,
                        precision,
                        image_size,
                        num_classes_per_sample,
                        exclude_val,
                        reason_seg_data,
                        explanatory,
                    )
                )
            elif dataset == "chatunivi":
                self.all_datasets.append(
                    ChatUniviDataset(
                        tokenizer                = tokenizer,
                        vision_tower             = vision_tower,
                        samples_per_epoch        = samples_per_epoch,
                        precision                = precision,
                        image_size               = image_size,
                        univi_data_list          = univi_data_list,
                        univi_data_ratio         = univi_data_ratio,
                        univi_max_image_len      = univi_max_image_len,
                        image_aspect_ratio       = 'pad',
                        univi_sample_frame_range = univi_sample_frame_range,
                        univi_sample_list        = self.chatunivi_sample_list,
                    )
                )
            elif dataset == "rvos":            
                self.all_datasets.append(
                    RVOSDataset(
                        tokenizer                = tokenizer,
                        vision_tower             = vision_tower,
                        samples_per_epoch        = samples_per_epoch,
                        precision                = precision,
                        image_size               = image_size,
                        num_classes_per_sample   = num_classes_per_sample,
                        num_frames_sample_range  = rvos_num_frames_sample_range,
                        rvos_sample_policy       = rvos_sample_policy,
                        rvos_seg_data            = rvos_seg_data,
                        rvos_sample_ratio        = rvos_sample_ratio,
                        rvos_sample_list         = self.rvos_sample_list,
                    )
                )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        self.num_be_called += 1
        if self.dataset_sample_list == None:
            ind = np.random.choice(list(range(len(self.datasets))), p=self.sample_rate)
        else:
            ind = self.dataset_sample_list[self.num_be_called % self.samples_per_epoch]
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference


class ValDataset(torch.utils.data.Dataset):
    pixel_mean   = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std    = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size     = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,
        image_size=1024,
    ):
        self.base_image_dir = base_image_dir
        splits = val_dataset.split("|")
        if len(splits) == 2:
            ds, split = splits
            images = glob.glob(
                os.path.join(self.base_image_dir, "reason_seg", ds, split, "*.jpg")
            )
            self.images = images
            self.data_type = "reason_seg"
        elif len(splits) == 3:
            ds, splitBy, split = splits
            refer_api = REFER(osp.join(base_image_dir, 'refer_seg'), ds, splitBy)
            ref_ids_val = refer_api.getRefIds(split=split)
            images_ids_val = refer_api.getImgIds(ref_ids=ref_ids_val)
            refs_val = refer_api.loadRefs(ref_ids=ref_ids_val)
            refer_seg_ds = {}
            refer_seg_ds["images"] = []
            loaded_images = refer_api.loadImgs(image_ids=images_ids_val)
            for item in loaded_images:
                item = item.copy()
                if ds == "refclef":
                    item["file_name"] = os.path.join(
                        osp.join(base_image_dir, 'refer_seg'), 
                        "images/saiapr_tc-12", 
                        item["file_name"]
                    )
                elif ds in ["refcoco", "refcoco+", "refcocog", "grefcoco"]:
                    item["file_name"] = os.path.join(
                        osp.join(base_image_dir, 'refer_seg'),
                        "images/mscoco/images/train2014",
                        item["file_name"],
                    )
                refer_seg_ds["images"].append(item)
            refer_seg_ds["annotations"] = refer_api.Anns  # anns_val

            img2refs = {}
            for ref in refs_val:
                image_id = ref["image_id"]
                img2refs[image_id] = img2refs.get(image_id, []) + [
                    ref,
                ]
            refer_seg_ds["img2refs"] = img2refs
            self.refer_seg_ds = refer_seg_ds
            self.data_type = "refer_seg"

        self.ds = ds
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

    def __len__(self):
        if self.data_type == "refer_seg":
            return len(self.refer_seg_ds["images"])
        else:
            return len(self.images)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        if self.data_type == "refer_seg":
            refer_seg_ds = self.refer_seg_ds
            images = refer_seg_ds["images"]
            annotations = refer_seg_ds["annotations"]
            img2refs = refer_seg_ds["img2refs"]

            image_info = images[idx]
            image_path = image_info["file_name"]
            image_id = image_info["id"]

            refs = img2refs[image_id]
            if len(refs) == 0:
                raise ValueError("image {} has no refs".format(image_id))

            sents = []
            ann_ids = []
            for ref in refs:
                for sent in ref["sentences"]:
                    sents.append(sent["sent"].strip().lower())
                    ann_ids.append(ref["ann_id"])

            sampled_sents = sents
            sampled_ann_ids = ann_ids
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            is_sentence = False
        else:
            image_path = self.images[idx]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            json_path = image_path.replace(".jpg", ".json")
            mask_json, sampled_sents, is_sentence = get_mask_from_json(json_path, image)
            sampled_sents = [sampled_sents[0]]

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        i = 0
        while i < len(sampled_sents):
            conv.messages = []
            text = sampled_sents[i].strip()
            if is_sentence:
                conv.append_message(
                    conv.roles[0],
                    convert2imagesplit(UNIFIED_LONG_QUESTION_LIST[0].format(sent=text), 1),
                )
                conv.append_message(conv.roles[1], "Sure, it is [SEG].")
            else:
                conv.append_message(
                    conv.roles[0],
                    convert2imagesplit(UNIFIED_SHORT_QUESTION_LIST[0].format(sent=text), 1),
                )
                conv.append_message(conv.roles[1], "Sure, it is [SEG].")
            conversations.append(conv.get_prompt())
            i += 1

        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        # preprocess image for sam
        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        if self.data_type == "refer_seg":
            masks = []
            for i, ann_id in enumerate(sampled_ann_ids):
                ann = annotations[ann_id]
                if len(ann["segmentation"]) == 0 and sampled_sents[i] != "":
                    m = np.zeros((image_info["height"], image_info["width"], 1))
                else:
                    if type(ann["segmentation"][0]) == list:  # polygon
                        rle = mask.frPyObjects(
                            ann["segmentation"],
                            image_info["height"],
                            image_info["width"],
                        )
                    else:
                        rle = ann["segmentation"]
                        for i in range(len(rle)):
                            if not isinstance(rle[i]["counts"], bytes):
                                rle[i]["counts"] = rle[i]["counts"].encode()
                    m = mask.decode(rle)
                m = np.sum(
                    m, axis=2
                )  # sometimes there are multiple binary map (corresponding to multiple segs)
                m = m.astype(np.uint8)  # convert to np.uint8
                masks.append(m)
        else:
            masks = [mask_json]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        return (
            image_path,
            image,
            torch.stack([image_clip, image_clip],dim=0),
            conversations,
            masks,
            labels,
            resize,
            None,
            None,
            inference,
        )
