from typing import Any
import cv2
import os
import os.path as osp
import math
import json
import torch
import pickle
import argparse
import numpy as np
import whisper
from decord import VideoReader, cpu
from transformers import CLIPVisionModel, CLIPImageProcessor
from llamavid.model.multimodal_encoder.eva_vit import EVAVisionTowerLavis
from llamavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llamavid.conversation import conv_templates, SeparatorStyle
from llamavid.model.builder import load_pretrained_model
from llamavid.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from importlib.util import find_spec
if find_spec("gpustat") is None: os.system("pip install gpustat")
import GPUtil

class VideoFeatureExtractor:

    def __init__(
        self, 
        vision_tower    : str,
        image_processor : str,
        clip_infer_batch: int,
        gpu_id          : int,
        keep_last       : bool = True, # 保留最后一个视频的特征, 便于连续调用
    ):
        gpu_list = [_.id for _ in GPUtil.getGPUs()]
        assert gpu_id in gpu_list, f'gpu_id {gpu_id} not in {gpu_list}'
        torch.cuda.set_device(gpu_id)
        self.device = torch.device(f'cuda:{gpu_id}')

        self.vision_tower = EVAVisionTowerLavis(vision_tower, image_processor, args=None).to(self.device)
        self.vision_tower.eval()
        self.image_processor = self.vision_tower.image_processor
        self.clip_infer_batch = clip_infer_batch

        self.keep_last = keep_last
        self.last_video_dir = None
        self.last_video_features = None

    @torch.no_grad()
    def __call__(self, video_dir: str) -> dict:
        # Load video
        frame_path_list = sorted([osp.join(video_dir, x) for x in os.listdir(video_dir) if x.endswith('.jpg')])
        frame_list = [cv2.imread(x) for x in frame_path_list]
        frame_list = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in frame_list]
        num_frame = len(frame_list)

        # Extract video features
        video_input = '<image>' * num_frame
        if self.keep_last and self.last_video_dir == video_dir:
            video_features = self.last_video_features
        else:
            video_tensor = self.image_processor.preprocess(frame_list, return_tensors='pt')['pixel_values'].half()
            video_features = torch.FloatTensor(num_frame, 257, 1408).fill_(0)
            n_iter = int(math.ceil(num_frame / float(self.clip_infer_batch)))
            for i in range(n_iter):
                start = i * self.clip_infer_batch
                end = min((i + 1) * self.clip_infer_batch, num_frame)
                video_batch = video_tensor[start:end].to(self.device)
                batch_features = self.vision_tower(video_batch)
                video_features[start:end] = batch_features.detach()
            if self.keep_last:
                self.last_video_dir = video_dir
                self.last_video_features = video_features

        return dict(feats=video_features, inputs=video_input)

class LLaMAVIDGenerator:

    PROMPT_START = 'Below is a movie. Memorize the content and answer my question after watching this movie.'
    PROMPT_END   = 'Now the movie end.'

    def __init__(
        self, 
        model_path : str,
        gpu_id     : int,
        video_token: int  = 2,
        model_base : str  = None,
        load_8bit  : bool = False,
        load_4bit  : bool = False,
        conv_mode  : str  = 'vicuna_v1',
    ):
        gpu_list = [_.id for _ in GPUtil.getGPUs()]
        assert gpu_id in gpu_list, f'gpu_id {gpu_id} not in {gpu_list}'
        torch.cuda.set_device(gpu_id)
        self.device = torch.device(f'cuda:{gpu_id}')

        self.conv_mode   = conv_mode
        self.video_token = video_token

        replace_llama_attn_with_flash_attn(inference=True)
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit, load_4bit, device = self.device, device_map = None)
        self.model.eval()
        self.model = self.model.to(self.device)


    def __call__(
        self, 
        video_info  : dict,
        question    : str,
        prompt_start: str = None,
        prompt_end  : str = None
    ) -> dict:
        video = video_info['feats'][:, 1:].half()
        video = [video]

        prompt_input = video_info['inputs'].replace(DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_TOKEN * self.video_token)
        prompt_start = prompt_start or self.PROMPT_START
        prompt_end   = prompt_end   or self.PROMPT_END
        prompt       = prompt_start + prompt_input + prompt_end

        conv = conv_templates[self.conv_mode].copy()
        if self.model.config.mm_use_im_start_end:
            question = DEFAULT_IM_START_TOKEN + prompt + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            question = prompt + '\n' + question
        
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        print('> Input token num:', len(input_ids[0]))

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        cur_prompt = question
        with torch.inference_mode():
            self.model.update_prompt([[cur_prompt]])
            output_ids = self.model.generate(
                input_ids,
                images            = video,
                do_sample         = True,
                temperature       = 0.6,
                top_p             = 0.9,
                max_new_tokens    = 1024,
                use_cache         = True,
                stopping_criteria = [stopping_criteria]
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        return dict(prompt=prompt, question=question, answer=outputs)

class Inferencer:
    def __init__(
        self, 
        gpu_id          : int,
        # CLIP
        vision_tower    : str,
        image_processor : str,
        clip_infer_batch: int,
        # LLaMA-VID
        model_path      : str,
        video_token     : int = 2,
        model_base      : str = None,
        load_8bit       : bool = False,
        load_4bit       : bool = False,
        conv_mode       : str = 'vicuna_v1',
    ):
        self.video_feature_extractor = VideoFeatureExtractor(vision_tower, image_processor, clip_infer_batch, gpu_id)
        self.llama_vid_generator = LLaMAVIDGenerator(model_path, gpu_id, video_token, model_base, load_8bit, load_4bit, conv_mode)
    
    def __call__(self, video_dir: str, question: str):
        video_info = self.video_feature_extractor(video_dir)
        return self.llama_vid_generator(video_info, question)

def parse_args():
    parser = argparse.ArgumentParser(description="Extract CLIP feature and subtitles for a video")
    # CLIP
    parser.add_argument("--video_dir",        required=True, help="Path to read the videos from.")
    parser.add_argument("--clip_infer_batch", required=False, type=int, default=48, help="Number of frames/images to perform batch inference.")
    parser.add_argument("--vision_tower",     required=False, default='/mnt/nlp-ali/usr/yancilin/clyan-data/checkpoints/eva/eva_vit_g.pth', type=str, help="Path to EVA vision tower.")
    parser.add_argument("--image_processor",  required=False, default='/mnt/nlp-ali/usr/yancilin/clyan-data/checkpoints/openai/clip-vit-large-patch14', type=str, help="Path to CLIP image processor.")
    # LLaMA-VID
    parser.add_argument("--model-path",  type=str, default="/mnt/nlp-ali/usr/yancilin/clyan-data/checkpoints/llama-vid/YanweiLi/llama-vid-13b-full-224-video-fps-1")
    parser.add_argument("--model-base",  type=str, default=None)
    parser.add_argument("--video-token", type=int, default=2)
    parser.add_argument("--question",    type=str, required=True)
    parser.add_argument("--conv-mode",   type=str, default='vicuna_v1')
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    return args

"""
python run_demo.py \
    --video_dir /mnt/nlp-ali/usr/yancilin/clyan-data/other-datasets/eccv_dataset/export0126/JPEGImages/LV-VIS/train/01020 \
    --question "If I want to identify the car closest to the camera at the beginning of the movie, I should focus on those few frames of the movie."
"""

def main():
    args = parse_args()
    print(args)
    inferencer = Inferencer(
        vision_tower     = args.vision_tower,
        image_processor  = args.image_processor,
        clip_infer_batch = args.clip_infer_batch,
        model_path       = args.model_path,
        video_token      = args.video_token,
        model_base       = args.model_base,
        load_8bit        = args.load_8bit,
        load_4bit        = args.load_4bit,
        conv_mode        = args.conv_mode,
    )
    result = inferencer(args.video_dir, args.question)

if __name__ == "__main__":
    main()
