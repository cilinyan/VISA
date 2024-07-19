import os
import time
import torch
import argparse
import multiprocessing as mp
from termcolor import colored
from datetime import datetime
from importlib.util import find_spec
if find_spec("GPUtil") is None: os.system("pip install gputil")
import GPUtil

_GPU_LIST = [_.id for _ in GPUtil.getGPUs()]
_GPU_QUEUE = mp.Queue()
for _ in _GPU_LIST: _GPU_QUEUE.put(_)

def run_eval(meta_expression, temp_xmem_anno, final_xmem_anno, img_dir, split_part, cfgs=" --reversed "):
    gpu_id = _GPU_QUEUE.get()
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python eval.py --meta_exp {meta_expression} --output {final_xmem_anno} --generic_path {temp_xmem_anno} --img_dir {img_dir} --split_part {split_part} --dataset G {cfgs}"
    print(f"Running: {cmd}")
    os.system(cmd)
    _GPU_QUEUE.put(gpu_id)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_expression", type=str, help='/PATH/TO/ReVOS/meta_expressions_valid__llamavid.json')
    parser.add_argument("--temp_xmem_anno",  type=str, help='/PATH/TO/VISA_exp/revos_valid_XMem_temp/Annotations')
    parser.add_argument("--final_xmem_anno", type=str, help='/PATH/TO/VISA_exp/revos_valid_XMem_final/Annotations')
    parser.add_argument("--img_dir",         type=str, help='/PATH/TO/ReVOS/JPEGImages')
    args = parser.parse_args()

    p = mp.Pool(8)
    for split_part in [0, 1, 2, 3]:
        for cfgs in ["  ", " --reversed "]:
            p.apply_async(
                run_eval,
                args=(args.meta_expression, args.temp_xmem_anno, args.final_xmem_anno, args.img_dir, split_part, cfgs),
                error_callback=lambda e: print(colored(e, "red"))
            )
    p.close()
    p.join()

if __name__ == "__main__":
    main()
