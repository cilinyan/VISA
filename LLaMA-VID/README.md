# LLaMA-VID
This codebase is modified from [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID).

# Inference

## Build an API Server for LLaMA-VID

1. Modify the default parameters in the [http_server.py](./http_server.py) file: [vision_tower](https://github.com/cilinyan/VISA/blob/main/LLaMA-VID/http_server.py#L27), [image_processor](https://github.com/cilinyan/VISA/blob/main/LLaMA-VID/http_server.py#L28), [model-path](https://github.com/cilinyan/VISA/blob/main/LLaMA-VID/http_server.py#L30).
2. Run [http_server_mp.py](./http_server_mp.py) to build the API server for LLaMA-VID. 
```shell 
cd ${LLAMA_VID_ROOT}
python http_server_mp.py
```

## Using the API for Inference

```shell
cd ${LLAMA_VID_ROOT} 
python http_client_mp.py \
    --video_root     /mnt/public02/usr/yancilin/clyan_data/other-datasets/ReVOS/JPEGImages \
    --data_json_file /mnt/public02/usr/yancilin/clyan_data/other-datasets/ReVOS/meta_expressions_valid_.json
```
