import os
import os.path as osp
import multiprocessing as mp
from termcolor import colored

ROOT = 'ckpts'
NUM_PROC = 8
CKPTS = {
    "YanweiLi/llama-vid-13b-full-224-video-fps-1": [
        "https://huggingface.co/YanweiLi/llama-vid-13b-full-224-video-fps-1/resolve/main/.gitattributes?download=true",
        "https://huggingface.co/YanweiLi/llama-vid-13b-full-224-video-fps-1/resolve/main/README.md?download=true",
        "https://huggingface.co/YanweiLi/llama-vid-13b-full-224-video-fps-1/resolve/main/config.json?download=true",
        "https://huggingface.co/YanweiLi/llama-vid-13b-full-224-video-fps-1/resolve/main/generation_config.json?download=true",
        "https://huggingface.co/YanweiLi/llama-vid-13b-full-224-video-fps-1/resolve/main/pytorch_model-00001-of-00003.bin?download=true",
        "https://huggingface.co/YanweiLi/llama-vid-13b-full-224-video-fps-1/resolve/main/pytorch_model-00002-of-00003.bin?download=true",
        "https://huggingface.co/YanweiLi/llama-vid-13b-full-224-video-fps-1/resolve/main/pytorch_model-00003-of-00003.bin?download=true",
        "https://huggingface.co/YanweiLi/llama-vid-13b-full-224-video-fps-1/resolve/main/pytorch_model.bin.index.json?download=true",
        "https://huggingface.co/YanweiLi/llama-vid-13b-full-224-video-fps-1/resolve/main/special_tokens_map.json?download=true",
        "https://huggingface.co/YanweiLi/llama-vid-13b-full-224-video-fps-1/resolve/main/tokenizer.model?download=true",
        "https://huggingface.co/YanweiLi/llama-vid-13b-full-224-video-fps-1/resolve/main/tokenizer_config.json?download=true",
        "https://huggingface.co/YanweiLi/llama-vid-13b-full-224-video-fps-1/resolve/main/trainer_state.json?download=true",
        "https://huggingface.co/YanweiLi/llama-vid-13b-full-224-video-fps-1/resolve/main/training_args.bin?download=true"
    ]
}


def main():
    p = mp.Pool(NUM_PROC)
    for model_name, urls in CKPTS.items():
        model_dir = osp.join(ROOT, model_name)
        os.makedirs(model_dir, exist_ok=True)
        for url in urls:
            url = url.rsplit('?')[0]
            p.apply_async(os.system, args=(f'cd {model_dir} && wget {url}',), error_callback=lambda e: print(colored(e, 'red')))
    p.close()
    p.join()

if __name__ == '__main__':
    main()