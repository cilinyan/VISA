PROJECT_ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")
cd $PROJECT_ROOT

EVAL_DATASET=${1:-revos_valid}  # revos_valid, davis17_valid, refytvos_valid, mevis_test
deepspeed --master_port=24999 train_ds.py \
   --version="/mnt/public03/dataset/ovis/rgvos/visa7b/ckpt_model/hf_model" \
   --vision_pretrained="/mnt/public02/usr/yancilin/clyan_data/weights/sam/sam_vit_h_4b8939.pth" \
   --log_base_dir="/mnt/public03/dataset/ovis/rgvos/visa7b" \
   --exp_name="val_7b" \
   --balance_sample \
   --dataset="reason_seg" \
   --sample_rates="13" \
   --val_dataset "${EVAL_DATASET}" \
   --eval_only 
