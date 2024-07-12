###########################################################################
# Created by: NTU
# Email: heshuting555@gmail.com
# Copyright (c) 2023
###########################################################################
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
import os
import os.path as osp
import time
import pickle
import argparse
import cv2
import json
import numpy as np
from pycocotools import mask as cocomask
from metrics import db_eval_iou, db_eval_boundary
import multiprocessing as mp

NUM_WOEKERS = 128

def eval_queue(q, rank, out_dict, mevis_pred_path):
    while not q.empty():
        # print(q.qsize())
        vid_name, exp = q.get()

        vid = exp_dict[vid_name]

        exp_name = f'{vid_name}_{exp}'

        if not os.path.exists(f'{mevis_pred_path}/{vid_name}'):
            print(f'{vid_name} not found')
            out_dict[exp_name] = [0, 0]
            continue

        pred_0_path = f'{mevis_pred_path}/{vid_name}/{exp}/00000.png'
        pred_0 = cv2.imread(pred_0_path, cv2.IMREAD_GRAYSCALE)
        h, w = pred_0.shape
        vid_len = len(vid['frames'])
        gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)

        anno_ids = vid['expressions'][exp]['anno_id']

        for frame_idx, frame_name in enumerate(vid['frames']):
            for anno_id in anno_ids:
                mask_rle = mask_dict[str(anno_id)][frame_idx]
                if mask_rle:
                    gt_masks[frame_idx] += cocomask.decode(mask_rle)

            pred_masks[frame_idx] = cv2.imread(f'{mevis_pred_path}/{vid_name}/{exp}/{frame_name}.png', cv2.IMREAD_GRAYSCALE)

        j = db_eval_iou(gt_masks, pred_masks).mean()
        f = db_eval_boundary(gt_masks, pred_masks).mean()
        out_dict[exp_name] = [j, f]

def get_meta_exp(mevis_exp_path, ):
    
    # read expression data
    with open(str(mevis_exp_path), 'r') as f:
        subset_expressions_by_video = json.load(f)['videos']
    videos = sorted(list(subset_expressions_by_video.keys()))

    metas = []
    anno_count = 0  # serve as anno_id
    for vid in videos:
        vid_data   = subset_expressions_by_video[vid]
        vid_frames = sorted(vid_data['frames'])
        vid_len    = len(vid_frames)

        exp_id_list = sorted(list(vid_data['expressions'].keys()))
        for exp_id in exp_id_list:
            subset_expressions_by_video[vid]['expressions'][exp_id]["anno_id"] = [anno_count, ]
            anno_count += 1
    
    return subset_expressions_by_video



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("mevis_pred_path", type=str, )
    parser.add_argument("--mevis_exp_path", type=str, default="/mnt/nlp-ali/usr/yancilin/clyan-data/other-datasets/davis17/meta_expressions/valid/meta_expressions.json")
    parser.add_argument("--mevis_mask_path", type=str, default="/mnt/nlp-ali/usr/yancilin/clyan-data/other-datasets/davis17/valid/mask_dict.pkl")
    parser.add_argument("--save_name", type=str, default="david17_val.json")
    args = parser.parse_args()
    queue = mp.Queue()
    exp_dict = get_meta_exp(args.mevis_exp_path)
    if args.mevis_mask_path.endswith('json'):
        mask_dict = json.load(open(args.mevis_mask_path))
    elif args.mevis_mask_path.endswith('pkl'):
        mask_dict = pickle.load(open(args.mevis_mask_path, 'rb'))
    else:
        raise

    shared_exp_dict = mp.Manager().dict(exp_dict)
    shared_mask_dict = mp.Manager().dict(mask_dict)
    output_dict = mp.Manager().dict()

    for vid_name in exp_dict:
        vid = exp_dict[vid_name]
        for exp in vid['expressions']:
            queue.put([vid_name, exp])

    start_time = time.time()
    processes = []
    for rank in range(NUM_WOEKERS):
        p = mp.Process(target=eval_queue, args=(queue, rank, output_dict, args.mevis_pred_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


    j = [output_dict[x][0] for x in output_dict]
    f = [output_dict[x][1] for x in output_dict]

    output_path = osp.join(args.mevis_pred_path, '..', args.save_name)
    results = {
        'J'  : round(100 * float(np.mean(j)), 2),
        'F'  : round(100 * float(np.mean(f)), 2),
        'J&F': round(100 * float((np.mean(j) + np.mean(f)) / 2), 2),
    }
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(json.dumps(results, indent=4))

    end_time = time.time()
    total_time = end_time - start_time
    print("time: %.4f s" %(total_time))


    