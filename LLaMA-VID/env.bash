cd /mnt/public02/usr/yancilin/llama-vid-2
pip install -r requirements.txt
cd /mnt/public02/usr/yancilin/LLaVA
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
# pip install -e ".[train]"
pip install /mnt/public02/usr/yancilin/clyan_data/weights/flash_attn-2.3.6+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl