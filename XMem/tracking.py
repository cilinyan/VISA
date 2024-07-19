import sys
sys.path.insert(0, './XMem')

import os
import os.path as osp
import glob
import cv2
import json
import argparse
import multiprocessing as mp
from tqdm import tqdm
from termcolor import colored

from importlib.util import find_spec
if find_spec("GPUtil") is None: os.system("pip install gputil")
import GPUtil

_GPU_LIST = [_.id for _ in GPUtil.getGPUs()]
_GPU_QUEUE = mp.Queue()
for _ in _GPU_LIST: _GPU_QUEUE.put(_)

def run_eval(meta_expression, temp_xmem_anno, final_xmem_anno, img_dir, split_part, xmem_weight, cfgs=" --reversed ", ):
    gpu_id = _GPU_QUEUE.get()
    cmd = f"cd XMem && CUDA_VISIBLE_DEVICES={gpu_id} python eval.py --meta_exp {meta_expression} --output {final_xmem_anno} --generic_path {temp_xmem_anno} --img_dir {img_dir} --split_part {split_part} --model {xmem_weight} --dataset G {cfgs}"
    print(f"Running: {cmd}")
    os.system(cmd)
    _GPU_QUEUE.put(gpu_id)

def generate(obj, temp_xmem_anno, final_xmem_anno):

    obj_dir, video_name, obj_id, tp = obj
    img_list = glob.glob(obj_dir + '/*.png')  # Mask
    img_list.sort()
    frame_id = int(len(img_list) * tp)
    if frame_id == len(img_list):
        frame_id -= 1
    
    used_img = img_list[frame_id]

    img_output_path = osp.join(temp_xmem_anno, video_name, obj_id, osp.basename(used_img))
    final_img_output_dir = osp.join(final_xmem_anno, video_name, obj_id)
    img_output_dir = osp.dirname(img_output_path)
    os.makedirs(img_output_dir, exist_ok=True)
    os.makedirs(final_img_output_dir, exist_ok=True)
    os.system('cp {} {}'.format(used_img, img_output_path))

    img = cv2.imread(img_output_path)
    if img.sum() == 0:
        target_img_list = [i.split('/')[-1] for i in img_list]
        for img_ in target_img_list:
            print(os.path.join(final_img_output_dir, img_))
            os.system('cp {} {}'.format(img_output_path, os.path.join(img_output_dir, img_)))
            os.system('cp {} {}'.format(img_output_path, os.path.join(final_img_output_dir, img_)))
    return 0

def prepare(args):
    video_root      = args.video_root
    temp_xmem_anno  = args.temp_xmem_anno
    final_xmem_anno = args.final_xmem_anno
    os.makedirs(temp_xmem_anno, exist_ok=True)

    data = json.load(open(args.llama_vid_meta, 'r'))['videos']

    all_obj_list = []
    for video_name in data.keys():
        exps = data[video_name]['expressions']
        for obj_id in exps.keys():
            tp = exps[obj_id]['tp']
            obj_dir = os.path.join(video_root, video_name, obj_id)
            all_obj_list.append([obj_dir, video_name, obj_id, tp])

    print('start')
    cpu_num = mp.cpu_count()-1
    print("cpu_num:", cpu_num)
    pool = mp.Pool(cpu_num)
    pbar = tqdm(total=len(all_obj_list))

    for obj in all_obj_list:
        pool.apply_async(
            generate, 
            args           = (obj, temp_xmem_anno, final_xmem_anno ),
            callback       = lambda *a: pbar.update(1),
            error_callback = lambda e: print(colored(e, "red"))
        )
        
    pool.close()
    pool.join()
    pbar.close()

def inference(args):
    p = mp.Pool(8)
    for split_part in [0, 1, 2, 3]:
        for cfgs in ["  ", " --reversed "]:
            p.apply_async(
                run_eval,
                args=(args.llama_vid_meta, args.temp_xmem_anno, args.final_xmem_anno, args.img_dir, split_part, args.xmem_weight, cfgs),
                error_callback=lambda e: print(colored(e, "red"))
            )
    p.close()
    p.join()

"""
python XMem/tracking.py \
    --video_root      /mnt/public03/dataset/ovis/rgvos/visa7b/val_7b/revos_valid/Annotations \
    --temp_xmem_anno  /mnt/public03/dataset/ovis/rgvos/visa7b/val_7b/revos_valid/revos_valid_XMem_temp/Annotations \
    --final_xmem_anno /mnt/public03/dataset/ovis/rgvos/visa7b/val_7b/revos_valid/revos_valid_XMem_final/Annotations \
    --llama_vid_meta  /mnt/public02/usr/yancilin/clyan_data/other-datasets/ReVOS/meta_expressions_valid__llamavid.json \
    --img_dir         /mnt/public02/usr/yancilin/clyan_data/other-datasets/ReVOS/JPEGImages \
    --xmem_weight     /mnt/public02/usr/yancilin/VISA/XMem/weights/XMem.pth
"""

def main():
    parser = argparse.ArgumentParser(description='rgvos')
    parser.add_argument('--video_root',      type=str, help='/PATH/TO/VISA_exp/revos_valid/Annotations', )
    parser.add_argument('--temp_xmem_anno',  type=str, help='/PATH/TO/VISA_exp/revos_valid_XMem_temp/Annotations', )  # 保存单帧 Mask 的路径
    parser.add_argument('--final_xmem_anno', type=str, help='/PATH/TO/VISA_exp/revos_valid_XMem_final/Annotations', )  # 保存 XMem 最后输出结果的路径
    parser.add_argument("--llama_vid_meta",  type=str, help='/PATH/TO/ReVOS/meta_expressions_valid__llamavid.json', )
    parser.add_argument("--img_dir",         type=str, help='/PATH/TO/ReVOS/JPEGImages')
    parser.add_argument("--xmem_weight",     type=str, help='/PATH/TO/XMEM_WEIGHT')
    args = parser.parse_args()

    prepare(args)
    inference(args)

    print('Done.')


if __name__ == '__main__':
    main()
