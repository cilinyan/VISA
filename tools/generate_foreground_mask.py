###########################################################################
# Created by: BUAA
# Email: clyanhh@gmail.com
# Copyright (c) 2024
###########################################################################
import os
import os.path as osp
import json
import argparse
import numpy as np
import pycocotools.mask as maskUtils
from tqdm import tqdm
from operator import add
from functools import reduce
from termcolor import colored


def get_args():
    parser = argparse.ArgumentParser(description="Merge instances mask to generate foreground mask")
    parser.add_argument("--meta_expression", type=str, )
    parser.add_argument("--mask_dict", type=str, )
    return parser.parse_args()

def merge_rle(masks_rle_list: list, height: int, width: int):
    num_frames = len(masks_rle_list[0])
    assert all(len(masks_rle) == num_frames for masks_rle in masks_rle_list), "The number of frames in each mask should be the same"
    masks_rle_foreground_list = []
    foreground_ratio_list = []
    for frame_idx in range(num_frames):
        # merge all instance masks in the same frame
        mask_foreground = np.zeros((height, width), dtype=np.uint8)
        for masks_rle in masks_rle_list:
            segm = masks_rle[frame_idx]
            if segm is not None:
                m = maskUtils.decode(segm)
                m = m.sum(axis=2).astype(np.uint8) if m.ndim == 3 else m.astype(np.uint8)
                mask_foreground = (mask_foreground | m).astype(np.uint8)
        
        # calculate the foreground ratio
        foreground_ratio = mask_foreground.sum() / (height * width)
        foreground_ratio_list.append(foreground_ratio)

        # encoder the merged mask to rle
        mask_foreground_rle = maskUtils.encode(np.asfortranarray(mask_foreground))
        mask_foreground_rle['counts'] = mask_foreground_rle['counts'].decode()
        masks_rle_foreground_list.append(mask_foreground_rle)

    return masks_rle_foreground_list, foreground_ratio_list


def main():
    args = get_args()
    assert args.meta_expression.endswith('.json') and args.mask_dict.endswith('.json')

    mask_dict = json.load(open(args.mask_dict, 'r'))
    meta_expression = json.load(open(args.meta_expression, 'r'))['videos']
    mask_dict_foreground_path = args.mask_dict.replace('.json', '_foreground.json')

    assert not osp.exists(mask_dict_foreground_path), f"{mask_dict_foreground_path} already exists"
    mask_dict_foreground = {}  # video_name -> mask_rle (List[rle])

    pbar = tqdm(total = len(meta_expression))
    for video_name, video_info in meta_expression.items():
        height = video_info["height"]
        width = video_info["width"]
        anno_id_list = list(set(
            reduce(
                add, 
                [exp_info["anno_id"] for exp_info in video_info["expressions"].values()]
            )
        ))
        masks_rle_list = [mask_dict[str(anno_id)] for anno_id in anno_id_list]
        masks_rle_foreground_list, foreground_ratio_list = merge_rle(masks_rle_list, height, width)
        mask_dict_foreground[video_name] = {
            "masks_rle": masks_rle_foreground_list,
            "foreground_ratio": foreground_ratio_list
        }

        foreground_ratio = np.array(foreground_ratio_list).mean()
        pbar.set_description(f"Foreground ratio of {video_name}: {foreground_ratio:.2f}")
        if foreground_ratio == 0.0:
            print(colored(f"Foreground ratio of {video_name} is 0.0", "red"))
        pbar.update(1)
    pbar.close()

    with open(mask_dict_foreground_path, 'w') as f:
        json.dump(mask_dict_foreground, f, indent=4)

if __name__ == "__main__":
    main()






