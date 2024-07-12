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
import logging
import math
import torch
import torch.nn.functional as F
from pprint import pprint
from tqdm import tqdm
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
from .d2_datasets.refytvos_val_videos import REFYTVOS_VAL_VIDEOS
from .utils import (
    UNIFIED_SHORT_QUESTION_LIST, UNIFIED_LONG_QUESTION_LIST, ANSWER_LIST,
    DEFAULT_VIDEO_TOKEN, DEFAULT_IMAGE_TOKEN, convert2imagesplit
)
from .dataset_config import RVOS_DATA_INFO as _DATA_INFO
from .dataset_config import RVOS_ROOT

logger = logging.getLogger(__name__)

def get_zero_image(processor):
    i = Image.new('RGB', (224, 224), (0, 0, 0))
    return processor.preprocess(i, return_tensors='pt')['pixel_values'][0]

class RVOSEvalDataset(torch.utils.data.Dataset):
    # davis17_train, refytvos_train, mevis_train
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255
    def __init__(
        self, 
        tokenizer,
        vision_tower,
        output_dir: str,  # osp.join(output_dir, "Annotations", video_name, exp_id, f"{frame_id}.png")
        precision               : str  = "fp32",
        image_size              : int  = 224,
        rvos_dataset_name       : str  = "refytvos_train",
        max_image_token         : int  = 12,
    ):
        assert rvos_dataset_name in _DATA_INFO.keys(), f"dataset {rvos_dataset_name} not found!"
        self.root = RVOS_ROOT
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        assert max_image_token < 20, "max_image_token must < 20"
        self.max_image_token = max_image_token

        self.long_question_list = UNIFIED_LONG_QUESTION_LIST
        self.short_question_list = UNIFIED_SHORT_QUESTION_LIST

        self.answer_list = ANSWER_LIST

        self.output_dir = output_dir
        self.rvos_dataset_name = rvos_dataset_name
        assert self.rvos_dataset_name in _DATA_INFO.keys(), f"dataset {self.rvos_dataset_name} not found!"
        print(f"loading dataset {self.rvos_dataset_name} into memory...")
        image_root, json_file = _DATA_INFO[self.rvos_dataset_name]
        self.image_root = osp.join(self.root, image_root)
        self.json_file = osp.join(self.root, json_file)
        self.d2_dataset_dicts, self.lisa_dataset_dicts = self.load_data()

    def __len__(self):
        return len(self.lisa_dataset_dicts)
    
    def load_data(self, ):
        metas, mask_dict, vid2metaid, is_train = load_mevis_json(self.image_root, self.json_file, self.rvos_dataset_name)
        d2_dataset_dicts = []
        lisa_dataset_dicts = []
        
        tmp_valid_vid = 0
        for idx_vd, vid_dict in tqdm(enumerate(metas), desc=f'Loading {self.rvos_dataset_name} ...'):
            record = {}
            if (self.rvos_dataset_name == "refytvos_valid") and (vid_dict['video'] not in REFYTVOS_VAL_VIDEOS):
                continue
            record["file_names"] = [
                os.path.join(self.image_root, 'JPEGImages', vid_dict['video'], vid_dict["frames"][i]+ '.jpg') 
                for i in range(vid_dict["length"])
            ]
            record["length"] = vid_dict["length"]
            video_name, exp, anno_ids, obj_ids, category, exp_id = \
                vid_dict['video'], vid_dict['exp'], vid_dict['anno_id'], vid_dict['obj_id'], vid_dict['category'],  vid_dict['exp_id']

            exp = " ".join(exp.lower().split())
            if "eval_idx" in vid_dict:
                record["eval_idx"] = vid_dict["eval_idx"]

            video_objs = []
            record["annotations"] = video_objs
            record["sentence"]    = exp
            record["exp_id"]      = exp_id
            record["video_name"]  = video_name
            d2_dataset_dicts.append(record)
            
            for file_path in record["file_names"]:
                file_name = osp.basename(file_path).rsplit('.', 1)[0]
                output_path = osp.join(self.output_dir, "Annotations", video_name, exp_id, file_name + '.png')
                lisa_dataset_dicts.append(
                    dict(
                        idx_d2 = tmp_valid_vid,
                        frame_path = file_path,
                        output_path = output_path,
                    )
                )
            tmp_valid_vid += 1

        return d2_dataset_dicts, lisa_dataset_dicts

    def __getitem__(self, idx):
        data_lisa   = self.lisa_dataset_dicts[idx]
        data_d2     = self.d2_dataset_dicts[data_lisa['idx_d2']]
        frame_path  = data_lisa['frame_path']
        image_shape = cv2.imread(frame_path).shape[:2]
        zero_mask   = np.zeros(image_shape, dtype=np.uint8)
        data = {
            "video_name"           : data_d2['video_name'],
            "video_frame_path_list": data_d2['file_names'],
            "seg_frame_path"       : data_lisa['frame_path'],
            "exp_mask_pairs"       : [(data_d2['sentence'], zero_mask)],
        }

        # NOTE: 划分输入，token长度不能太长，只能抽帧
        idx_seg = data['video_frame_path_list'].index(data['seg_frame_path'])
        if self.max_image_token != 1:
            to_devide = (self.max_image_token - 1)
            step_size = math.ceil(len(data['video_frame_path_list']) / to_devide)
            idx_start = idx_seg % step_size
            idx_select = list(range(idx_start, len(data['video_frame_path_list']), step_size))
        else:
            idx_select = [idx_seg, ]
        assert idx_seg in idx_select
        data['video_frame_path_list'] = [data['video_frame_path_list'][i] for i in idx_select]

        frame_list = [cv2.imread(x) for x in data['video_frame_path_list']]
        frame_list = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in frame_list]
        frame_clip_list = [self.clip_image_processor(x, return_tensors="pt")["pixel_values"][0] for x in frame_list]

        video_len = len(frame_clip_list)
        
        seg_frame = cv2.imread(data['seg_frame_path'])
        seg_frame = cv2.cvtColor(seg_frame, cv2.COLOR_BGR2RGB)
        seg_frame_clip = self.clip_image_processor(seg_frame, return_tensors="pt")["pixel_values"][0]
        seg_frame_sam = self.transform.apply_image(seg_frame)
        resize = seg_frame_sam.shape[:2]
        seg_frame_sam = self.preprocess(torch.from_numpy(seg_frame_sam).permute(2, 0, 1).contiguous())

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        questions = []
        answers = []
        conversations = []
        masks = []
        conv = conversation_lib.default_conversation.copy()
        for exp, mask in data['exp_mask_pairs']:
            text = exp.strip()
            assert len(text.split('||')) == 1
            # question_template = random.choice(self.long_question_list)
            if text[-1] == '?':
                question = self.long_question_list[0].format(sent=text)
            else:
                question = self.short_question_list[0].format(sent=text)
            question = convert2imagesplit(question, video_len)

            
            questions.append(question)
            # answers.append(random.choice(self.answer_list))
            answers.append(self.answer_list[0])

            conv.messages = []
            conv.append_message(conv.roles[0], questions[-1])
            conv.append_message(conv.roles[1], answers[-1])
            conversations.append(conv.get_prompt())

            masks.append(mask)

        masks = torch.from_numpy(np.stack(masks, axis=0))
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label

        return (
            data_lisa['output_path'], 
            seg_frame_sam,
            torch.stack(frame_clip_list + [seg_frame_clip], dim=0),
            conversations,
            masks,
            label,
            resize,
            questions,
            [exp for exp, _ in data['exp_mask_pairs']],
            True
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

