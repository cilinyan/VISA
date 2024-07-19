import glob
import os
from PIL import Image
import numpy as np
import tqdm
import multiprocessing

multi_dir = "mevis_val/vis_output/"
outdir = "mevis_val_merge/vis_output/"


all_obj_list = []
video_list = glob.glob(os.path.join(multi_dir, "0/*"))
for video in video_list:
    obj_list = glob.glob(video + "/*")
    all_obj_list = all_obj_list + ['/'.join(i.split('/')[-2:]) for i in obj_list]

def merge(obj):
        obj_output_dir = os.path.join(outdir, obj)
        os.makedirs(obj_output_dir, exist_ok=True)
        img_list = [i.split('/')[-1] for i in glob.glob(os.path.join(multi_dir, "0", obj, "*.png"))]
        for img_name in img_list:
            agg_img = None
            for i in range(7):
                img_path = os.path.join(multi_dir, str(i), obj, img_name)
                tmp_img = (np.array(Image.open(img_path)) > 0).astype(np.uint8)
                if agg_img is not None:
                    agg_img = agg_img + tmp_img
                else:
                    agg_img = tmp_img
            agg_img = (agg_img >= 4).astype(np.uint8)
            agg_img = Image.fromarray(agg_img)
            img_output_path = os.path.join(obj_output_dir, img_name)
            agg_img.save(img_output_path)

print('start')
cpu_num = multiprocessing.cpu_count()-1
print("cpu_num:", cpu_num)
pool = multiprocessing.Pool(cpu_num)

for obj in all_obj_list:
    pool.apply_async(merge, args=(obj,))
    
pool.close()
pool.join()
