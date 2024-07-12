import os
import random

import cv2
import numpy as np
import torch
import copy
import traceback
import json
import torch.nn.functional as F
from termcolor import colored
from pycocotools import mask
from transformers import CLIPImageProcessor
from typing import Dict, Sequence
from PIL import Image
from decord import VideoReader, cpu

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .dataset_config import MIMIC_imageonly, SQA, VIDEO
from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, convert2imagesplit)

def _get_rawvideo_dec(video_path, image_processor, max_frames=64, image_resolution=224, video_framerate=1, s=None, e=None):
    # speed up video decode via decord.
    # video_mask = np.zeros(max_frames, dtype=np.int64)

    # T x 3 x H x W
    # video = np.zeros((max_frames, 3, image_resolution, image_resolution), dtype=np.float64)

    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)
        start_time = start_time if start_time >= 0. else 0.
        end_time = end_time if end_time >= 0. else 0.
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        print(video_path)
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
    num_frames = f_end - f_start + 1
    assert num_frames > 0, f'num_frames: {num_frames}, f_start: {f_start}, f_end: {f_end}, fps: {fps}, video_path: {video_path}'
    # T x 3 x H x W
    if num_frames <= max_frames:
        sample_pos = range(f_start, f_end + 1)
    else:
        split_point = np.linspace(0, num_frames, num=max_frames+1, dtype=int)
        sample_pos = [np.random.randint(split_point[i], split_point[i+1]) for i in range(max_frames)]
    patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]

    patch_images = [image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in patch_images]
    slice_len = len(patch_images)

    assert slice_len > 0, f'slice_len: {slice_len}, f_start: {f_start}, f_end: {f_end}, fps: {fps}, video_path: {video_path}'

    return patch_images

def get_zero_image(processor):
    i = Image.new('RGB', (224, 224), (0, 0, 0))
    return processor.preprocess(i, return_tensors='pt')['pixel_values'][0]



class ChatUniviDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    name2cfg = {
        "mimic": MIMIC_imageonly,
        "sqa"  : SQA,
        "video": VIDEO,
    }

    def __init__(
        self, 
        tokenizer,
        vision_tower,
        samples_per_epoch        : int = 500 * 8 * 2 * 10,
        precision                : str = "fp32",
        image_size               : int = 224,
        univi_data_list          : str = "mimic||sqa||video",
        univi_data_ratio         : str = "1||1||1",
        univi_max_image_len      : int = 64,
        image_aspect_ratio       : str = 'pad',
        univi_sample_frame_range : str = "10,12",
        univi_sample_list : list = [],
    ):
        self.image_aspect_ratio = image_aspect_ratio
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.num_be_called = 0
        self.univi_sample_frame_range = [int(x) for x in univi_sample_frame_range.split(",")]
        assert len(self.univi_sample_frame_range) == 2
        self.univi_sample_list = univi_sample_list

        self.samples_per_epoch = samples_per_epoch
        self.univi_data_list = univi_data_list.split("||")
        univi_data_ratio = np.array([float(x) for x in univi_data_ratio.split("||")])
        self.univi_data_ratio = univi_data_ratio / univi_data_ratio.sum()
        self.univi_max_image_len = univi_max_image_len
        self.name2list_data_dict = {}
        self.folder_dict = {}
        for dataset_name in self.univi_data_list:
            self.name2list_data_dict[dataset_name] = self.load_data(dataset_name)
            print(colored(f'Loaded {len(self.name2list_data_dict[dataset_name])} samples from {dataset_name}', 'green'))
            image_folder = [folder for folder in self.name2cfg[dataset_name] if folder != "chat_path"]
            for folder in image_folder:
                if folder not in self.folder_dict:
                    self.folder_dict[folder] = self.name2cfg[dataset_name][folder]

    def load_data(self, dataset_name: str):
        list_data_dict = json.load(open(self.name2cfg[dataset_name]["chat_path"], 'r'))
        if dataset_name == 'sqa':
            list_data_dict = [e for e in list_data_dict if "image" in e]  # only keep the data with image
            for idx, data_dict in enumerate(list_data_dict):
                if data_dict["conversations"][0]["value"].endswith("<image>"):
                    data_dict["conversations"][0]["value"] = data_dict["conversations"][0]["value"].replace("<image>", "")
                    list_data_dict[idx]["conversations"][0]["value"] = "<image>\n" + data_dict["conversations"][0]["value"]
                assert not list_data_dict[idx]["conversations"][0]["value"].endswith("<image>"), f'Found <image> in the end of {list_data_dict[idx]["conversations"]}'
        return list_data_dict

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, i, max_try: int = 10):
        for idx_try in range(max_try):
            try:
                image, file_path, question, answer = self.sample_data()

                ori_size = (1024, 1024)
                masks = torch.rand(0, *ori_size)
                label = torch.ones(ori_size) * self.ignore_label

                conv = conversation_lib.default_conversation.copy()
                conv.messages = []
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], answer)
                conversations = [conv.get_prompt(), ]

                return (
                    file_path,
                    torch.zeros((3, ori_size[0], ori_size[1]), dtype=torch.float32),
                    torch.stack(image,dim=0),
                    conversations,
                    masks,
                    label,
                    ori_size,
                    [question, ],
                    [question, ],
                )
            
            except:
                print(colored(f'[ {idx_try + 1} / {max_try} ] Error in sample_data: {traceback.format_exc()}', 'red'))
        raise RuntimeError(f'Cannot find a valid sample after {max_try} tries.')

    def sample_data(self, ):
        # 随机选择一个dataset
        dataset_name = np.random.choice(self.univi_data_list, p=self.univi_data_ratio)
        list_data_dict = self.name2list_data_dict[dataset_name]
        # 随机选择一个sample
        source = list_data_dict[np.random.randint(0, len(list_data_dict))]
        if 'image' in source:
            image_file = source['image']
            file = image_file[0] if type(image_file) is list else image_file

            if "\\" in file:
                image_folder = self.folder_dict['ScienceQA']
            elif "CGD" in file:
                image_folder = self.folder_dict['CDG']
            elif "DC" in file:
                image_folder = self.folder_dict['DC']
            elif "LA" in file:
                image_folder = self.folder_dict['LA']
            elif "SD" in file:
                image_folder = self.folder_dict['SD']
            elif "SN" in file:
                image_folder = self.folder_dict['SN']
            elif "TVC" in file:
                image_folder = self.folder_dict['TVC']
            elif "VST" in file:
                image_folder = self.folder_dict['VST']
            elif "GCC" in file:
                image_folder = self.folder_dict['CC3M']
            elif "COCO_train2014" in file:
                image_folder = self.folder_dict['COCO2014']
            else:
                image_folder = self.folder_dict['COCO2017']


            if type(image_file) is list:
                image = [Image.open(os.path.join(image_folder, file.replace("\\", "/"))).convert('RGB') for file in image_file]
                file_path_record = ','.join([os.path.join(image_folder, file.replace("\\", "/")) for file in image_file])
                if self.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result

                    image = [expand2square(i, tuple(int(x * 255) for x in self.clip_image_processor.image_mean)) for i in image]
                    image = [self.clip_image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]
                else:
                    image = [self.clip_image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]
            else:
                image = Image.open(os.path.join(image_folder, image_file.replace("\\", "/"))).convert('RGB')
                file_path_record = os.path.join(image_folder, image_file.replace("\\", "/"))
                if self.image_aspect_ratio == 'pad':
                    def expand2square(pil_img, background_color):
                        width, height = pil_img.size
                        if width == height:
                            return pil_img
                        elif width > height:
                            result = Image.new(pil_img.mode, (width, width), background_color)
                            result.paste(pil_img, (0, (width - height) // 2))
                            return result
                        else:
                            result = Image.new(pil_img.mode, (height, height), background_color)
                            result.paste(pil_img, ((height - width) // 2, 0))
                            return result
                    image = expand2square(image, tuple(int(x * 255) for x in self.clip_image_processor.image_mean))
                    image = [self.clip_image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0], ]
                else:
                    image = [self.clip_image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0], ]

            conversations = copy.deepcopy(source["conversations"])

            question = ""
            answer = ""
            for conv in conversations:
                if conv['from'] == 'human':
                    question = conv['value']
                else:
                    answer = conv['value']
            assert (question != "") and (answer != ""), f'question: {question}, answer: {answer}, conversations: {conversations}'
            
            if "<image><image>" in question:
                question = question.replace("<image><image>", DEFAULT_VIDEO_TOKEN)
                question = convert2imagesplit(question, 2)

        elif "video" in source:
            video_file = source['video']
            video_folder = self.folder_dict['VIDEO']
            file_path_record = os.path.join(video_folder, video_file)

            # get sample frame by self.univi_sample_frame_range
            if len(self.univi_sample_list) > 0:
                sample_frame_len = self.univi_sample_list[self.num_be_called % len(self.univi_sample_list)]
                self.num_be_called += 1
            else:
                sample_frame_len = np.random.randint(self.univi_sample_frame_range[0], self.univi_sample_frame_range[1] + 1)
            image = _get_rawvideo_dec(file_path_record, self.clip_image_processor, max_frames=sample_frame_len)

            
            video_len = len(image)

            conversations = copy.deepcopy(source["conversations"])

            question = ""
            answer = ""
            for conv in conversations:
                if conv['from'] == 'human':
                    question = conv['value']
                else:
                    answer = conv['value']
            assert (question != "") and (answer != ""), f'question: {question}, answer: {answer}, conversations: {conversations}'

            question = convert2imagesplit(question, video_len)

        else:
            raise NotImplementedError

        return image, file_path_record, question, answer




if __name__ == '__main__':
    pass
