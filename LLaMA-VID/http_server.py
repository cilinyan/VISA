import json
import argparse
from run_demo import Inferencer
from flask import Flask, request

class InferenceServer(Inferencer):

    def __init__(self, **kwargs):
        port = kwargs.pop('port')
        super().__init__(**kwargs)

        server = Flask(__name__)
        server.route('/post', methods=['POST'])(self.post)
        server.run(host="0.0.0.0", port=port, threaded=False)

    def post(self):
        if request.method == "POST":
            video_dir = request.json.get('video_dir')
            question  = request.json.get('question')
            result = self(video_dir, question)
            return json.dumps(result)

def parse_args():
    parser = argparse.ArgumentParser(description="Extract CLIP feature and subtitles for a video")
    # CLIP
    parser.add_argument("--clip_infer_batch", required=False, type=int, default=48, help="Number of frames/images to perform batch inference.")
    parser.add_argument("--vision_tower",     required=False, default='model_zoo/LAVIS/eva_vit_g.pth', type=str, help="Path to EVA vision tower.")
    parser.add_argument("--image_processor",  required=False, default='/mnt/public02/usr/yancilin/clyan_data/weights/openai/clip-vit-large-patch14', type=str, help="Path to CLIP image processor.")
    # LLaMA-VID
    parser.add_argument("--model-path",  type=str, default="/mnt/public02/usr/yancilin/clyan_data/weights/llama-vid/YanweiLi/llama-vid-13b-full-224-video-fps-1")
    parser.add_argument("--model-base",  type=str, default=None)
    parser.add_argument("--video-token", type=int, default=2)
    parser.add_argument("--gpu-id",      type=int, default=0)
    parser.add_argument("--port",        type=int, default=8000)
    parser.add_argument("--conv-mode",   type=str, default='vicuna_v1')
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args)
    inferencer = InferenceServer(
        port             = args.port,
        gpu_id           = args.gpu_id,
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

if __name__ == "__main__":
    main()