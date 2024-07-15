import json
import requests
import os
import glob
import numpy as np
import tqdm
import argparse


import os
import os.path as osp
import json
import requests
import multiprocessing as mp
from tqdm import tqdm
from typing import Tuple, List
from collections import defaultdict
from termcolor import colored

_ports = json.load(open('port.json', 'r'))
_PORTS = mp.Queue()
for port in _ports:
    _PORTS.put(port)

def call(video_dir: str, question: str, ):
    port = _PORTS.get()
    url = "http://localhost:{port}/post".format(port=port)
    r = requests.post(url, json={"video_dir": video_dir, "question": question, }, )
    r = json.loads(r.content.decode())
    _PORTS.put(port)
    return video_dir, question, r

def call_batch(params_list: List[Tuple[str, str]], ):
    num_proc = _PORTS.qsize()
    p = mp.Pool(num_proc)
    pbar = tqdm(total=len(params_list))

    vid2ques2ans = defaultdict(lambda: defaultdict(list))  # video_dir -> question -> list of answer
    def _update(result):
        video_dir, question, answer = result
        vid2ques2ans[video_dir][question].append(answer)
        pbar.update(1)

    for params in params_list:
        p.apply_async(call, args=params, callback=_update, error_callback=lambda e: print(colored(e, 'red')))

    p.close()
    p.join()
    pbar.close()

    vid2ques2ans = {k: dict(v) for k, v in vid2ques2ans.items()}
    return vid2ques2ans

_ELEM_PERCENT_LIST = [
    ("last 0%", 1.0), ("last 10%", 0.9), ("last 20%", 0.8), ("last 30%", 0.7), ("last 40%", 0.6),
    ("last 50%", 0.5), ("last 60%", 0.4), ("last 70%", 0.3), ("last 80%", 0.2), ("last 90%", 0.1), ("last 100%", 0.0),
    ("10%", 0.1), ("20%", 0.2), ("30%", 0.3), ("40%", 0.4), ("50%", 0.5), 
    ("60%", 0.6), ("70%", 0.7), ("80%", 0.8), ("90%", 0.9), ("100%", 1.0), ("0%", 0.0), 
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_root', type=str, default="/mnt/nlp-ali/usr/yancilin/clyan-data/other-datasets/eccv_dataset/export0206_/valid/JPEGImages/")
    parser.add_argument('--data_json_file', type=str, default="/mnt/nlp-ali/usr/yancilin/clyan-data/other-datasets/eccv_dataset/export0206_/valid/meta_expressions.json")
    parser.add_argument('--repeat_num', type=int, default=10)
    args = parser.parse_args()

    video_root = args.video_root
    data_json = json.load(open(args.data_json_file, 'r'))
    out_json_file = args.data_json_file.replace('.json', '_llamavid_mp.json')

    params_list = []
    for video_name in data_json['videos'].keys():
        video_data = data_json['videos'][video_name]
        video_dir = os.path.join(video_root, video_name)
        for obj_id in video_data['expressions'].keys():
            exp = video_data['expressions'][obj_id]['exp']
            if exp[-1] == '?':
                exp = exp[:-1]
            question = ("If I want to find '" + exp + "', which percentage mark of the video should I check?\n \
            Please choose one answer from 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, and 100%.")
            params_list.extend([(video_dir, question)] * args.repeat_num)
    vid2ques2ans = call_batch(params_list)

    for video_name in tqdm(data_json['videos'].keys()):
        video_data = data_json['videos'][video_name]
        video_dir = os.path.join(video_root, video_name)
        for obj_id in video_data['expressions'].keys():
            exp = video_data['expressions'][obj_id]['exp']
            if exp[-1] == '?':
                exp = exp[:-1]
            question = ("If I want to find '" + exp + "', which percentage mark of the video should I check?\n \
            Please choose one answer from 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, and 100%.")
            tp_list = []
            for response in vid2ques2ans[video_dir][question]:
                result = response['answer']
                for elem, percent in _ELEM_PERCENT_LIST:
                    if elem in result:
                        tp_list.append(percent)
                        break
            tt = 0.5 if len(tp_list) == 0 else np.array(tp_list).mean()
            video_data['expressions'][obj_id]['tp'] = tt
    with open(out_json_file, 'w') as f:
        json.dump(data_json, f, indent=4)

if __name__ == '__main__':
    main()