import os
import cv2
import sys
import torch
import random
import itertools
import numpy as np
import pandas as pd
import ujson as json
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from modules.basic_utils import load_json
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture


class DIDEMODataset(Dataset):
    """
        videos_dir: directory where all videos are stored
        config: AllConfig object
        split_type: 'train'/'test'
        img_transforms: Composition of transforms
    """

    def __init__(self, config: Config, split_type='train', img_transforms=None):
        self.config = config
        self.videos_dir = config.videos_dir
        self.img_transforms = img_transforms
        self.split_type = split_type
        # db_file = 'data/DIDEMO/DIDEMO_data.json'
        test_csv = 'data/didemo/didemo_retrieval/test.tsv'

        train_csv = 'data/didemo/didemo_retrieval/train.tsv'

        # train_df = pd.read_csv(train_csv)
        # self.train_df = pd.read_csv(train_csv)
        self.test_df = pd.read_csv(test_csv, sep='\t')

        # self.db = load_json(db_file)
        if split_type == 'train':
            train_df = pd.read_csv(train_csv, sep='\t')
            self.train_df = train_df
            self.train_vids = train_df['video_id'].unique()
            self.captions = train_df['caption']
            # self._compute_vid2caption()
            # self._construct_all_train_pairs()
            self.meta = self.train_df
        else:
            self.test_df = pd.read_csv(test_csv, sep='\t')
            self.meta = self.test_df

    def __getitem__(self, index):
        video_path, caption, video_id = self._get_vidpath_and_caption_by_index(index)
        imgs, idxs = VideoCapture.load_frames_from_video(video_path,
                                                         self.config.num_frames,
                                                         self.config.video_sample_type)

        # process images of video
        if self.img_transforms is not None:
            imgs = self.img_transforms(imgs)

        return {
            'video_id': video_id,
            'video': imgs,
            'text': caption,
        }

    def __len__(self):
        if self.split_type == 'train':
            return len(self.train_df)
        return len(self.test_df)

    def _get_vidpath_and_caption_by_index(self, index):
        # returns video path and caption as string
        index = index % len(self.meta)
        if self.split_type == 'train':
            caption, vid = self.train_df.iloc[index]
            video_path = os.path.join(self.videos_dir, vid)
        else:
            vid = self.test_df.iloc[index].video_id
            video_path = os.path.join(self.videos_dir, vid)
            caption = self.test_df.iloc[index].caption

        return video_path, caption, vid

    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        if self.split_type == 'train':
            for vid in self.train_vids:
                for caption in self.vid2caption[vid]:
                    self.all_train_pairs.append([vid, caption])

    def _compute_vid2caption(self):
        self.vid2caption = defaultdict(list)
        for annotation in self.train_df['sentences']:
            caption = annotation['caption']
            vid = annotation['video_id']
            self.vid2caption[vid].append(caption)
