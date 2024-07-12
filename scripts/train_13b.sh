PROJECT_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")
cd $PROJECT_ROOT

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 deepspeed --master_port=24999 train_ds.py \
   --version="/mnt/nlp-ali/usr/yancilin/clyan-data-2/video-llm/Chat-UniVi/Chat-UniVi-13B" \
   --dataset_dir='/mnt/nlp-ali/usr/yancilin/LISA/datasets' \
   --vision_pretrained="/mnt/nlp-ali/usr/yancilin/.cache/sam/sam_vit_h_4b8939.pth" \
   --log_base_dir="/mnt/nlp-ali/usr/yancilin/clyan-data/exps/rgvos_ablation" \
   --exp_name="VISA-13B" \
   --balance_sample \
   --dataset="sem_seg||refer_seg||vqa||reason_seg||chatunivi||rvos" \
   --sample_rates="9,3,3,1,4,12" \
   --univi_sample_frame_range="8,12" \
   --rvos_seg_data="mevis_train||refytvos_train||davis17_train||revos_train||lvvis_train" \
   --rvos_sample_ratio="4000||15000||400||3000||3000"
