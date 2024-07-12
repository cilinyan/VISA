###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################


import json
import logging
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
from collections import defaultdict
"""
This file contains functions to parse MeViS dataset of
COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)

def load_mevis_json(image_root, json_file, dataset_name, is_train: bool = False):

    ann_file = json_file
    with open(str(ann_file), 'r') as f:
        subset_expressions_by_video = json.load(f)['videos']
    videos = list(subset_expressions_by_video.keys())  # d56a6ec78cfa, 377b1c5f365c, ...
    print('number of video in the datasets:{}'.format(len(videos)))
    metas = []
    is_train = (image_root.split('/')[-1] == 'train') or is_train
    if is_train:
        mask_json = os.path.join(image_root, 'mask_dict.json')
        print(f'Loading masks form {mask_json} ...')
        with open(mask_json) as fp:
            mask_dict = json.load(fp)

        vid2metaid = defaultdict(list)
        for vid in videos:  # d56a6ec78cfa, 377b1c5f365c, ...
            # vid_data    = {'expressions': dict, 'vid_id': int, 'frames': List[int]}
            # expressions = {'0': {"exp": str, "obj_id": List[int], "anno_id": List[int]}, ...}
            vid_data   = subset_expressions_by_video[vid]  
            vid_frames = sorted(vid_data['frames'])  # 00000, 00001, ...
            vid_len    = len(vid_frames)
            if vid_len < 2:
                continue
            # if ('rgvos' in dataset_name) and vid_len > 80:
            #     continue
            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                meta['video']    = vid  # 377b1c5f365c
                meta['exp']      = exp_dict['exp']  # 4 lizards moving around
                meta['obj_id']   = [int(x) for x in exp_dict['obj_id']]   # [0, 1, 2, 3, ]
                meta['anno_id']  = [str(x) for x in exp_dict['anno_id']]  # [2, 3, 4, 5, ]
                meta['frames']   = vid_frames  # ['00000', '00001', ...]
                meta['exp_id']   = exp_id  # '0'
                meta['category'] = 0
                meta['length']   = vid_len
                metas.append(meta)
                vid2metaid[vid].append(len(metas) - 1)
    else:
        mask_dict = dict()
        vid2metaid = defaultdict(list)
        for vid in videos:
            vid_data   = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len    = len(vid_frames)
            for exp_id, exp_dict in vid_data['expressions'].items():
                meta = {}
                meta['video']    = vid
                meta['exp']      = exp_dict['exp']
                meta['obj_id']   = -1
                meta['anno_id']  = -1
                meta['frames']   = vid_frames
                meta['exp_id']   = exp_id
                meta['category'] = 0
                meta['length']   = vid_len
                metas.append(meta)
                vid2metaid[vid].append(len(metas) - 1)
    return metas, mask_dict, vid2metaid, is_train