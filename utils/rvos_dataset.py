###########################################################################
# Created by: BUAA
# Email: clyanhh@gmail.com
# Copyright (c) 2024
###########################################################################
import itertools
import json
import os
import os.path as osp
import pickle
import sys
import cv2
import time
import random
import torch
import math
import torch.nn.functional as F
from pprint import pprint
from termcolor import colored

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
import pycocotools.mask as maskUtils
from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from transformers import CLIPImageProcessor
from PIL import Image

from .d2_datasets.refytvos_utils import load_refytvos_json
from .d2_datasets.mevis_utils import load_mevis_json
from .utils import (
    DEFAULT_VIDEO_TOKEN, DEFAULT_IMAGE_TOKEN,
    UNIFIED_SHORT_QUESTION_LIST, UNIFIED_LONG_QUESTION_LIST, ANSWER_LIST,
    convert2imagesplit
)

from .dataset_config import RVOS_DATA_INFO as _DATA_INFO
from .dataset_config import RVOS_ROOT

def get_zero_image(processor):
    i = Image.new('RGB', (224, 224), (0, 0, 0))
    return processor.preprocess(i, return_tensors='pt')['pixel_values'][0]


class RVOSDataset(torch.utils.data.Dataset):
    # davis17_train, refytvos_train, mevis_train
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255
    def __init__(
        self, 
        tokenizer,
        vision_tower,
        samples_per_epoch       : int   = 500 * 8 * 2 * 10,
        precision               : str   = "fp32",
        image_size              : int   = 224,
        num_classes_per_sample  : int   = 3,
        num_frames_sample_range : int   = "8,12",
        rvos_sample_policy      : str   = "uniform",
        rvos_seg_data           : str   = "mevis_train||refytvos_train||davis17_train||revos_train",
        rvos_sample_ratio       : str   = '4000||15000||400||6000',
        rvos_sample_list        : list  = [],
        prob_no_in_video        : float = 0.05,
    ):
        self.root = RVOS_ROOT
        self.num_classes_per_sample = num_classes_per_sample
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.samples_per_epoch = samples_per_epoch
        self.transform = ResizeLongestSide(image_size)
        self.num_frames_sample_range = [int(x) for x in num_frames_sample_range.split(",")]
        assert len(self.num_frames_sample_range) == 2 and self.num_frames_sample_range[0] <= self.num_frames_sample_range[1], f"invalid num_frames_sample_range {num_frames_sample_range}"
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.prob_no_in_video = prob_no_in_video

        assert rvos_sample_policy in ["random", "uniform"], f"invalid rvos_sample_policy {rvos_sample_policy}"
        self.rvos_sample_policy = rvos_sample_policy
        self.rvos_sample_list = rvos_sample_list
        self.num_be_called = 0

        self.short_question_list = UNIFIED_SHORT_QUESTION_LIST
        self.long_question_list = UNIFIED_LONG_QUESTION_LIST

        self.answer_list = ANSWER_LIST

        self.rvos_seg_ds_list = rvos_seg_data.split("||")
        rvos_sample_ratio = np.array([float(x) for x in rvos_sample_ratio.split("||")])
        self.rvos_sample_ratio = rvos_sample_ratio / rvos_sample_ratio.sum()
        self.rvos_seg_data = {}
        for dataset in self.rvos_seg_ds_list:
            assert dataset in _DATA_INFO.keys(), f"dataset {dataset} not found!"
            print(f"loading dataset {dataset} into memory...")
            image_root, json_file = _DATA_INFO[dataset]
            image_root = osp.join(self.root, image_root)
            json_file = osp.join(self.root, json_file)
            if 'mevis' in dataset or 'revos' in dataset or 'lvvis' in dataset:
                metas, mask_dict, vid2metaid, is_train = load_mevis_json(image_root, json_file, dataset, is_train = True)
            elif 'refytvos' in dataset or 'davis' in dataset:
                metas, mask_dict, vid2metaid, is_train = load_refytvos_json(image_root, json_file, dataset)
            else:
                raise ValueError(f"Unknown dataset name: {dataset}")
            assert is_train, 'only support training mode for now'
            print(f'Loaded {dataset} dataset, with {len(metas)} expressions, {len(vid2metaid)} videos')

            self.rvos_seg_data[dataset] = {
                'image_root': image_root,
                'json_file' : json_file,
                'metas'     : metas,
                'mask_dict' : mask_dict,
                'is_train'  : is_train,
                'vid2metaid': vid2metaid,
            }

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        data = self.sample_data()

        frame_list = [cv2.imread(x) for x in data['video_frame_path_list']]
        frame_list = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in frame_list]
        frame_clip_list = [self.clip_image_processor(x, return_tensors="pt")["pixel_values"][0] for x in frame_list]
        video_len = len(frame_clip_list)
        
        seg_frame = cv2.imread(data['seg_frame_path'])
        seg_frame = cv2.cvtColor(seg_frame, cv2.COLOR_BGR2RGB)
        seg_frame_clip = self.clip_image_processor(seg_frame, return_tensors="pt")["pixel_values"][0]
        seg_frame_sam = self.transform.apply_image(seg_frame)
        resize = seg_frame_sam.shape[:2]
        frame_sam_list = [self.preprocess(torch.from_numpy(seg_frame_sam).permute(2, 0, 1).contiguous())]

        questions = []
        answers = []
        conversations = []
        masks = []
        conv = conversation_lib.default_conversation.copy()
        for exp, mask in data['exp_mask_pairs']:
            text = exp.strip()
            assert len(text.split('||')) == 1
            if text[-1] == "?":
                question = random.choice(self.long_question_list).format(sent=text)
            else:
                question = random.choice(self.short_question_list).format(sent=text)
            question = convert2imagesplit(question, video_len)

            seg_replace = ", ".join(f'({i}) [SEG]' for i in range(video_len))
            answer = random.choice(self.answer_list).format(seg=seg_replace)

            questions.append(question)
            answers.append(answer)

            conv.messages = []
            conv.append_message(conv.roles[0], questions[-1])
            conv.append_message(conv.roles[1], answers[-1])
            conversations.append(conv.get_prompt())

            masks.append(mask)

        masks = torch.from_numpy(np.stack(masks, axis=0)) # (num_classes, num_frame, H, W)
        label = torch.ones(masks.shape[-2], masks.shape[-1]) * self.ignore_label
        
        return (
            ','.join(data['video_frame_path_list']),
            torch.stack(frame_sam_list,dim=0),
            torch.stack(frame_clip_list + [seg_frame_clip],dim=0),
            conversations,
            masks,
            label,
            resize,
            questions,
            [exp for exp, _ in data['exp_mask_pairs']],
        )

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
    

    def sample_data(self,):
        ds         = np.random.choice(list(range(len(self.rvos_seg_ds_list))), p=self.rvos_sample_ratio)
        ds         = self.rvos_seg_ds_list[ds]
        metas      = self.rvos_seg_data[ds]['metas']
        mask_dict  = self.rvos_seg_data[ds]['mask_dict']
        image_root = self.rvos_seg_data[ds]['image_root']
        vid2metaid = self.rvos_seg_data[ds]['vid2metaid']

        # sample a video 
        vid = np.random.choice(list(vid2metaid.keys()))
        meta_ids = vid2metaid[vid]
        # random choose self.num_classes_per_sample indices
        meta_ids = np.random.choice(meta_ids, min(self.num_classes_per_sample, len(meta_ids)), replace=False)
        video_name = metas[meta_ids[0]]['video']
        assert all([metas[meta_id]['video'] == video_name for meta_id in meta_ids]), "video name not match"

        record = {}
        vid_dict_first = metas[meta_ids[0]]
        record["file_names"] = [
            os.path.join(image_root, 'JPEGImages', vid_dict_first['video'], vid_dict_first["frames"][i]+ '.jpg') 
            for i in range(vid_dict_first["length"])
        ]
        record["length"] = vid_dict_first["length"]
        # 随机选择 self.num_frames_per_sample 帧
        # self.num_frames_sample_range
        if len(self.rvos_sample_list) > 0:
            num_frames_per_sample = self.rvos_sample_list[self.num_be_called % len(self.rvos_sample_list)]
            self.num_be_called += 1
        else:
            num_frames_per_sample = np.random.randint(self.num_frames_sample_range[0], self.num_frames_sample_range[1] + 1)

        if vid_dict_first["length"] > num_frames_per_sample:
            if self.rvos_sample_policy == "random":
                frame_ids = np.random.choice(vid_dict_first["length"], num_frames_per_sample, replace=False).tolist()
                frame_ids = sorted(frame_ids)
            elif self.rvos_sample_policy == "uniform":
                num_length = vid_dict_first["length"]
                split_point = np.linspace(0, num_length, num=num_frames_per_sample+1, dtype=int)
                frame_ids = [np.random.randint(split_point[i], split_point[i+1]) for i in range(num_frames_per_sample)]

        else:
            frame_ids = list(range(vid_dict_first["length"]))
        video_frame_path_list = [record["file_names"][i] for i in frame_ids]
        # 随机选择一帧用作分割
        seg_frame_id = np.random.choice(frame_ids)
        seg_frame_path = record["file_names"][seg_frame_id]
        image_shape = cv2.imread(record["file_names"][0]).shape[:2]
        # 提取 不同exp 的 mask
        exp_mask_pairs = []
        for meta_id in meta_ids:
            vid_dict = metas[meta_id]
            assert vid_dict['video'] == video_name, "video name not match"
            assert vid_dict['length'] == vid_dict_first['length'], "video length not match"
            anno_ids = vid_dict['anno_id']
            obj_ids = vid_dict['obj_id']
            exp = vid_dict['exp']
            if 'lvvis' in ds:
                exp = exp.replace('_', ' ')     
            m_final_list = []
            for seg_frame_id in [seg_frame_id]:
                m_final = np.zeros(image_shape, dtype=np.uint8)
                for x, obj_id in zip(anno_ids, obj_ids):
                    segm = mask_dict[x][seg_frame_id]
                    if segm is not None:
                        m = maskUtils.decode(segm)
                        if m.ndim == 3:
                            m = m.sum(axis=2).astype(np.uint8)
                        else:
                            m = m.astype(np.uint8)
                        m_final = m_final | m
                m_final_list.append(m_final)
            m_final_list = np.stack(m_final_list, axis=0)  # (num_frame, H, W)

            exp_mask_pairs.append((exp, m_final_list))

        # 以 self.prob_no_in_video 的概率随机选择其他视频的帧, 并将mask置为0
        if random.random() < self.prob_no_in_video:
            vid_other = np.random.choice([x for x in vid2metaid.keys() if x != vid])
            meta_id_other = random.choice(vid2metaid[vid_other])
            vid_dict_other = metas[meta_id_other]
            assert vid_dict_other['video'] != video_name, "video name not match"
            # 随机选择一帧用作分割
            seg_frame_id_other = np.random.choice(vid_dict_other["length"])
            frame_path = os.path.join(image_root, 'JPEGImages', vid_dict_other['video'], vid_dict_other["frames"][seg_frame_id_other]+ '.jpg')
            seg_frame_path = frame_path
            image_shape = cv2.imread(frame_path).shape[:2]
            m_final = np.zeros(image_shape, dtype=np.uint8)

            exp_mask_pairs = [(x[0], m_final.copy()) for x in exp_mask_pairs]

        data = {
            "video_name"           : video_name,
            "video_frame_path_list": video_frame_path_list,
            "seg_frame_path"       : seg_frame_path,
            "exp_mask_pairs"       : exp_mask_pairs,
        }

        return data
    
