import sys
import os
import os.path as osp
import glob
import cv2
import multiprocessing
import json
import argparse
from tqdm import tqdm
from termcolor import colored

"""
python generate_xmem_data_single.py \
    --video_root      /PATH/TO/VISA_exp/revos_valid/Annotations \
    --output_dir      /PATH/TO/VISA_exp/revos_valid_XMem_temp/Annotations \
    --final_xmem_anno /PATH/TO/VISA_exp/revos_valid_XMem_final/Annotations \
    --llama_vid_meta  /PATH/TO/ReVOS/meta_expressions_valid__llamavid.json
"""

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

def main():
    parser =  argparse.ArgumentParser(description='rgvos')
    parser.add_argument('--video_root',      type=str, help='/PATH/TO/VISA_exp/revos_valid/Annotations', )
    parser.add_argument('--temp_xmem_anno',  type=str, help='/PATH/TO/VISA_exp/revos_valid_XMem_temp/Annotations', )  # 保存单帧 Mask 的路径
    parser.add_argument('--final_xmem_anno', type=str, help='/PATH/TO/VISA_exp/revos_valid_XMem_final/Annotations', )  # 保存 XMem 最后输出结果的路径
    parser.add_argument("--llama_vid_meta",  type=str, help='/PATH/TO/ReVOS/meta_expressions_valid__llamavid.json', )
    args = parser.parse_args()

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
    cpu_num = multiprocessing.cpu_count()-1
    print("cpu_num:", cpu_num)
    pool = multiprocessing.Pool(cpu_num)
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


if __name__ == '__main__':
    main()