import os
from os import path
import json
import glob

from inference.data.video_reader import VideoReader


class LongTestDataset:

    def __init__(self, meta_expression, data_root, size=-1, img_dir = '', reversed_ = False, split_part = 0):
        self.image_dir = img_dir
        self.mask_dir = data_root
        self.size = size
        self.reversed = reversed_
        self.split_part = split_part

        self.vid_list = []

        videos_names = json.load(open(meta_expression, 'r'))['videos']
        for video_name in videos_names:
            video_mask_dir = path.join(self.mask_dir, video_name)
            obj_ids = [d for d in os.listdir(video_mask_dir) if os.path.isdir(path.join(video_mask_dir, d))]
            for obj_id in obj_ids:
                obj_dir = path.join(video_mask_dir, obj_id)
                img_list = glob.glob(obj_dir + '/*')
                if len(img_list) == 1:
                    self.vid_list.append(path.join(video_name, obj_id))
        

        self.vid_list.sort()
        self.vid_list = [i for idx, i in enumerate(self.vid_list) if idx % 4 == self.split_part]

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, '/'.join(video.split('/')[:-1])), 
                path.join(self.mask_dir, video),
                to_save = [
                    name[:-4] for name in os.listdir(path.join(self.mask_dir, video))  # remove .png
                ],
                size=self.size,
                reversed=self.reversed,
            )

    def __len__(self):
        return len(self.vid_list)


class DAVISTestDataset:
    def __init__(self, data_root, imset='2017/val.txt', size=-1):
        if size != 480:
            self.image_dir = path.join(data_root, 'JPEGImages', 'Full-Resolution')
            self.mask_dir = path.join(data_root, 'Annotations', 'Full-Resolution')
            if not path.exists(self.image_dir):
                print(f'{self.image_dir} not found. Look at other options.')
                self.image_dir = path.join(data_root, 'JPEGImages', '1080p')
                self.mask_dir = path.join(data_root, 'Annotations', '1080p')
            assert path.exists(self.image_dir), 'path not found'
        else:
            self.image_dir = path.join(data_root, 'JPEGImages', '480p')
            self.mask_dir = path.join(data_root, 'Annotations', '480p')
        self.size_dir = path.join(data_root, 'JPEGImages', '480p')
        self.size = size

        with open(path.join(data_root, 'ImageSets', imset)) as f:
            self.vid_list = sorted([line.strip() for line in f])

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                size=self.size,
                size_dir=path.join(self.size_dir, video),
            )

    def __len__(self):
        return len(self.vid_list)


class YouTubeVOSTestDataset:
    def __init__(self, data_root, split, size=480):
        self.image_dir = path.join(data_root, 'all_frames', split+'_all_frames', 'JPEGImages')
        self.mask_dir = path.join(data_root, split, 'Annotations')
        self.size = size

        self.vid_list = sorted(os.listdir(self.image_dir))
        self.req_frame_list = {}

        with open(path.join(data_root, split, 'meta.json')) as f:
            # read meta.json to know which frame is required for evaluation
            meta = json.load(f)['videos']

            for vid in self.vid_list:
                req_frames = []
                objects = meta[vid]['objects']
                for value in objects.values():
                    req_frames.extend(value['frames'])

                req_frames = list(set(req_frames))
                self.req_frame_list[vid] = req_frames

    def get_datasets(self):
        for video in self.vid_list:
            yield VideoReader(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                size=self.size,
                to_save=self.req_frame_list[video], 
                use_all_mask=True
            )

    def __len__(self):
        return len(self.vid_list)
