# VISA: Reasoning Video Object Segmentation via Large Language Model

<font size=7><div align='center' >
[![ GitHub stars ](https://badgen.net/github/stars/cilinyan/VISA)](https://github.com/cilinyan/VISA)
[![arXiv](https://img.shields.io/badge/arXiv-2407.11325-972B24)](http://arxiv.org/abs/2407.11325)
[![Static Badge](https://img.shields.io/badge/dataset-ReVOS-3F7EBC)](https://github.com/cilinyan/ReVOS-api)
</div></font>

<div align=center>
<img src="assert/architecture.png" style="width:100%;">
</div>

## 🚀 Performance

<div style="text-align: justify;">
  VISA demonstrates remarkable proficiency in handling complex segmentation tasks that require: (a) reasoning based on world knowledge; (b) inference of future events; and (c) a comprehensive understanding of video content.
</div>

<div align=center>
<img src="assert/performance.png" style="width:50%;">
</div>

## 🛠️ Installation
```shell
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## 🦄 Training and Validation

### 1. Training Data Preparation

Before training, please download the datasets, and then configure the path in [dataset_config.py](utils/dataset_config.py).

<details open>
<summary> <strong>LISA's Dataset</strong> </summary>
  
Follow [LISA](https://github.com/dvlab-research/LISA/tree/main) to prepare LISA's datasets. The dataset folder should be stored in the `$LISA_ROOT` folder.

```
LISA_ROOT
├── ade20k
├── coco
├── cocostuff
├── llava_dataset
├── mapillary
├── reason_seg
├── refer_seg
└── vlpart
```

</details>


<details open>
<summary> <strong>Chat-UniVi's Dataset</strong> </summary>

Follow [Chat-UniVi/Chat-UniVi-Instruct](https://huggingface.co/datasets/Chat-UniVi/Chat-UniVi-Instruct/tree/main) to prepare `Chat-UniVi-Instruct` datasets. The dataset folder should be stored in the `$ChatUniVi_ROOT` folder.
```
ChatUniVi_ROOT
├── Fine-tuning
│   ├── MIMIC_imageonly
│   └── VIDEO
└── ScienceQA_tuning
```

</details>


<details open>
<summary> <strong>RVOS's Dataset</strong> </summary>

1. Reasoning Video Segmentation Datasets: [ReVOS](https://github.com/cilinyan/ReVOS-api).
2. Referring Video Segmentation Datasets: [Ref-Youtube-VOS](https://github.com/wjn922/ReferFormer/blob/main/docs/data.md), [Ref-DAVIS17](https://github.com/wjn922/ReferFormer/blob/main/docs/data.md), [MeViS](https://github.com/henghuiding/MeViS).
3. Open-Vocabulary Video Instance Segmentation Dataset: [LV-VIS](https://github.com/haochenheheda/LVVIS/tree/main).
  
   Download `mask_dict.json` and `meta_expressions.json` from [OneDrive](https://mailsjlueducn-my.sharepoint.com/:f:/g/personal/yancl9918_mails_jlu_edu_cn/EttXAjMV8yFJhHMQwX3mIw0BP7dymKV-cuw4uAotDaAwYw?e=j6Y44X) or [BaiduPan](https://pan.baidu.com/s/1LOWPnuxXF_LXGSL7osRptA?pwd=visa). Then, put the annotations files in the `$RVOS_ROOT/lvvis/train` directory as follows.
```
RVOS_ROOT
├── ReVOS
│   ├── JPEGImages 
│   ├── mask_dict.json             
│   ├── mask_dict_foreground.json   
│   ├── meta_expressions_train_.json 
│   └── meta_expressions_valid_.json 
├── lvvis
│   └── train
|       ├── JPEGImages
|       ├── mask_dict.json
|       └── meta_expressions.json
├── Ref-Youtube-VOS
├── davis17
└── mevis
```

</details>

### 2. Pre-trained weights

<details open>
<summary> <strong>Chat-UniVi</strong> </summary>

To train VISA-7B or 13B, you need to download Chat-UniVi weights from [Chat-UniVi-7B](https://huggingface.co/Chat-UniVi/Chat-UniVi) and [Chat-UniVi-13B](https://huggingface.co/Chat-UniVi/Chat-UniVi-13B).

</details>

<details open>
<summary> <strong>SAM</strong> </summary>

Download SAM ViT-H pre-trained weights from the [link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).

</details>

### 3. Training VISA

```shell
# Training VISA-7B
bash scripts/train_7b.sh 

# Extracting fp32 consolidated weights from a zero 1, 2 and 3 DeepSpeed checkpoints.
cd /PATH/TO/VISA-7B/ckpt_model && python zero_to_fp32.py . ../pytorch_model.bin

# Merge LoRA Weight
CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version Chat-UniVi/Chat-UniVi \
  --weight /PATH/TO/VISA-7B/pytorch_model.bin \
  --save_path /PATH/TO/VISA-7B/hf_model
```

### 4. Validation

<details open>
<summary> <strong>1. Using `VISA` to generate predicted mask of each video <a href="https://github.com/cilinyan/VISA/blob/main/scripts/val_7b_video.sh">[running demo]</a></strong> </summary>

```shell
deepspeed --master_port=24999 train_ds.py \
  --version="/PATH/TO/VISA-7B/hf_model" \
  --vision_pretrained="/PATH/TO/sam_vit_h_4b8939.pth" \
  --log_base_dir="/PATH/TO/LOG_BASE_DIR" \
  --exp_name="val_7b" \
  --balance_sample \
  --dataset="reason_seg" \
  --sample_rates="13" \
  --val_dataset "revos_valid" \
  --eval_only 
```

</details>

<details open>
<summary> <strong>2. Using <a href="https://github.com/dvlab-research/LLaMA-VID">LLaMA-VID</a> to generate target frame for each video</a></strong> </summary>

> you can directly download the results of our run from [OneDrive](https://mailsjlueducn-my.sharepoint.com/:u:/g/personal/yancl9918_mails_jlu_edu_cn/ETmoJF2i8ZZBsgIwdELiL8gBfptZZoPWjx6Y0eH6Myr3sw?e=mTt6rO) or [BaiduPan](https://pan.baidu.com/s/1YWs6NLPvANfhgUBHKQwnBg?pwd=visa)

- Run http_server_mp.py to build the API server for LLaMA-VID [`running demo`](https://github.com/cilinyan/VISA/blob/c53d2cd31407eab583c5eb04f84fd95b4694f2ce/utils_llamavid/llamavid_server.py#L215-L220)
   ```shell
   python utils_llamavid/llamavid_server.py \
       --vision_tower /PATH/TO/eva_vit_g.pth \
       --image_processor /PATH/TO/openai/clip-vit-large-patch14 \
       --model-path /PATH/TO/YanweiLi/llama-vid-13b-full-224-video-fps-1
   ```
- Using the API for inference [`running demo`](https://github.com/cilinyan/VISA/blob/c53d2cd31407eab583c5eb04f84fd95b4694f2ce/utils_llamavid/llamavid_client.py#L58-L63)
  
   ```shell
   python utils_llamavid/llamavid_client.py \
       --video_root /PATH/TO/ReVOS/JPEGImages \
       --data_json_file /PATH/TO/ReVOS/meta_expressions_valid_.json
   ```

</details>

<details open>
<summary> <strong>3. Using <a href="https://github.com/cilinyan/VISA/blob/main/XMem/tracking.py">XMem</a> for mask propagation <a href="https://github.com/cilinyan/VISA/blob/c53d2cd31407eab583c5eb04f84fd95b4694f2ce/XMem/tracking.py#L103-L110">[running demo]</a> </strong> </summary>
</details>

<details open>

<summary> <strong>4. Evaluate ReVOS's performance <a href="https://github.com/cilinyan/VISA/blob/main/tools/eval_revos.py#L74-L81">[running demo]</a> </strong> </summary>

```shell
cd tools
python eval_revos.py /PATH/TO/FINAL_ANNOTATION [ARGS]
```
  
</details>

## 📑 Todo list
  
- [x] Release code with `Text-guided Frame Sampler`'s Local Sampling

- [ ] Release VISA model weights

- [ ] Release code with `Text-guided Frame Sampler`'s Global-Local Sampling

## ⭐ Cite

If you find this project useful in your research, please consider citing:

```
@article{yan2024visa,
  title={VISA: Reasoning Video Object Segmentation via Large Language Models},
  author={Yan, Cilin and Wang, Haochen and Yan, Shilin and Jiang, Xiaolong and Hu, Yao and Kang, Guoliang and Xie, Weidi and Gavves, Efstratios},
  journal={arXiv preprint arXiv:2407.11325},
  year={2024}
}
```

## 🎖️ Acknowledgement
This work is built upon the [LLaVA](https://github.com/haotian-liu/LLaVA), [SAM](https://github.com/facebookresearch/segment-anything), [LISA](https://github.com/dvlab-research/LISA), [Chat-UniVi](https://github.com/PKU-YuanGroup/Chat-UniVi), [MeViS](https://github.com/henghuiding/MeViS), [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID) and [XMem](https://github.com/hkchengrex/XMem).

