import os.path as osp

# LISA
LISA_ROOT = "/mnt/nlp-ali/usr/yancilin/LISA/datasets"

# ChatUniVi
ChatUniVi_ROOT = "/mnt/nlp-ali/usr/yancilin/clyan-data-2/video-llm/Chat-UniVi-Instruct"
MIMIC_imageonly = {
    "chat_path": osp.join(ChatUniVi_ROOT, "Fine-tuning/MIMIC_imageonly/MIMIC-IT-imageonly.json"),
    "CDG"      : osp.join(ChatUniVi_ROOT, "Fine-tuning/MIMIC_imageonly/CDG/images"),
    "LA"       : osp.join(ChatUniVi_ROOT, "Fine-tuning/MIMIC_imageonly/LA/images"),
    "SD"       : osp.join(ChatUniVi_ROOT, "Fine-tuning/MIMIC_imageonly/SD/images"),
}
VIDEO = {
    "chat_path": osp.join(ChatUniVi_ROOT, "Fine-tuning/VIDEO/video_chat.json"),
    "VIDEO"    : osp.join(ChatUniVi_ROOT, "Fine-tuning/VIDEO/Activity_Videos"),
}
SQA = {
    "chat_path": osp.join(ChatUniVi_ROOT, "ScienceQA_tuning/llava_train_QCM-LEA.json"),
    "ScienceQA": osp.join(ChatUniVi_ROOT, "ScienceQA_tuning/train"),
}

# RVOS
RVOS_ROOT = "/mnt/nlp-ali/usr/yancilin/clyan-data/other-datasets/"
RVOS_DATA_INFO = {
    "mevis_train"   : ("mevis/train",           "mevis/train/meta_expressions.json"),
    "mevis_val"     : ("mevis/valid_u",         "mevis/valid_u/meta_expressions.json"),
    "mevis_test"    : ("mevis/valid",           "mevis/valid/meta_expressions.json"),
    "refytvos_train": ('Ref-Youtube-VOS/train', 'Ref-Youtube-VOS/meta_expressions/train/meta_expressions.json'),
    "refytvos_valid": ('Ref-Youtube-VOS/valid', 'Ref-Youtube-VOS/meta_expressions/valid/meta_expressions.json'),
    "davis17_train" : ('davis17/train',         'davis17/meta_expressions/train/meta_expressions.json'),
    "davis17_valid" : ('davis17/valid',         'davis17/meta_expressions/valid/meta_expressions.json'),
    "revos_train"   : ('ReVOS',                 'ReVOS/meta_expressions_train_.json'),
    "revos_valid"   : ('ReVOS',                 'ReVOS/meta_expressions_valid_.json'),
    "lvvis_train"   : ("lvvis/train",           "lvvis/train/meta_expressions.json"),
}