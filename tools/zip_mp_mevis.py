###########################################################################
# Created by: BUAA
# Email: clyanhh@gmail.com
# Copyright (c) 2024
###########################################################################
import os
import os.path as osp
import zipfile
from multiprocessing import Pool, cpu_count
import tempfile
import shutil
from tqdm import tqdm
import argparse


def zip_files(path, temp_zip_file):
    with zipfile.ZipFile(temp_zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(path):
            for file in files:
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, path)
                zipf.write(full_path, arcname=relative_path)

def main():
    parser = argparse.ArgumentParser(description='Zip files')
    parser.add_argument('dir', help='path to the directory')
    parser.add_argument('out', help='path to the output zip file')
    parser.add_argument('--rm', action='store_true', help='remove the directory after zipping')
    args = parser.parse_args()
    path      = args.dir[:-1] if args.dir.endswith('/') else args.dir
    iter_name = osp.basename(path)
    exp_dir   = osp.dirname(path)
    exp_name  = osp.basename(exp_dir)
    out_file  = args.out

    num_cpus = cpu_count()  # 获取CPU的数量

    p = Pool(processes=num_cpus)
    subdirs = [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    subdir2zip = dict()
    pbar = tqdm(subdirs, desc='Zipping directories to temporary zip files')
    for subdir in subdirs:
        temp_zip_file = tempfile.mkstemp(suffix=f'_{os.getpid()}.zip')[1]
        p.apply_async(zip_files, args=(subdir, temp_zip_file), error_callback=lambda e: print(e), callback=lambda _: pbar.update())
        subdir2zip[subdir] = temp_zip_file
    p.close()
    p.join()

    # 创建最终的zip文件
    pbar = tqdm(total=len(subdirs), desc='Merging temporary zip files to final zip file')
    with zipfile.ZipFile(out_file, 'w') as final_zipf:
        for subdir, temp_zip_file in subdir2zip.items():
            subdir_name = osp.basename(subdir)
            with zipfile.ZipFile(temp_zip_file, 'r') as temp_zipf:
                for file in temp_zipf.namelist():
                    final_zipf.writestr(osp.join(subdir_name, file), temp_zipf.read(file))
            os.remove(temp_zip_file)
            pbar.update()
    pbar.close()

    if args.rm:
        try:
            shutil.rmtree(path)
        except Exception as e:
            os.system(f'sudo rm -rf {path}')
        print(f'Removed {path}')

if __name__ == '__main__':
    main()