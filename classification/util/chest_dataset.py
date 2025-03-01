import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import random
from glob import glob
import pandas as pd
from itertools import chain
class nihchest:
    gray_images = True
    task = 'multilabel'  # choose from 'binary', 'multiclass' or 'multilabel'
    num_labels = 14  # choose None for binary and multiclass

    def __init__(self, root='', mode='train', transform=None):

        self.root = root
        self.transform = transform

        # 1. load csv & set path
        df = pd.read_csv(os.path.join(self.root, 'Data_Entry_2017.csv'))
        # print(os.path.join(self.root, 'vis_224_process/image224'))
        # img_paths = {os.path.basename(x): x for x in
        #              glob(os.path.join(self.root, 'vis_448_process/seg_rec_image', '*.png'))}
        # img_paths = {os.path.basename(x): x for x in
        #              glob(os.path.join(self.root, 'images', '*.png'))}
        img_paths = {os.path.basename(x): x for x in glob(os.path.join(self.root, 'vis_448_process/seg_rec_image', '*.png'))}
        # print(img_paths)
        df['path'] = df['Image Index'].map(img_paths.get)
        # print(df['path'])

        # 2. set train flag
        with open(os.path.join(self.root, 'train_val_list.txt'), 'rt') as f:
            train_flag = {x.strip('\n'): 1 for x in f.readlines()}

        with open(os.path.join(self.root, 'test_list.txt'), 'rt') as f:
            train_flag.update({x.strip('\n'): 0 for x in f.readlines()})

        df['train'] = df['Image Index'].map(train_flag.get)

        # 3. change label to one-hot label
        df['Finding Labels'] = df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
        all_labels = np.unique(list(chain(*df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
        all_labels = [x for x in all_labels if len(x) > 0]

        for c_label in all_labels:
            if len(c_label) > 1:  # leave out empty labels
                df[c_label] = df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

        # 4. split dataset into train, test
        df_split = df[df['train'] == (1 if mode == 'train' else 0)]
        self.x = df_split['path'].values.tolist()
        self.y = df_split[all_labels].values
        self.classes = all_labels
        self.norm_weight = np.array(self.y).sum(axis=0) / (np.array(self.y).sum(axis=0) ** 2).sum() ** 0.5
        # self.weight = (1 / np.array(self.y).mean(axis=0)) / (1 / np.array(self.y).sum(axis=0)).mean()
        self.weight = np.stack([1 / (np.array(self.y) == 0).astype(float).sum(axis=0),
                                1 / (np.array(self.y) == 1).astype(float).sum(axis=0)], axis=1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # print(self.x[idx])
        image = Image.open(self.x[idx])
        label = np.asarray(self.y[idx]).astype(np.float32)
        # print(self.y[idx])
        if self.transform:
            image = self.transform(image)
        # print(label)
        return image, label
class nihchest_mixed:
    gray_images = True
    task = 'multilabel'  # choose from 'binary', 'multiclass' or 'multilabel'
    num_labels = 14  # choose None for binary and multiclass

    def __init__(self, root='', mode='train', transform=None):
        self.root = root
        self.transform = transform

        # 1. load csv & set path
        df = pd.read_csv(os.path.join(self.root, 'Data_Entry_2017.csv'))
        # print(os.path.join(self.root, 'vis_224_process/image224'))
        img_paths = {os.path.basename(x): x for x in
                     glob(os.path.join(self.root, 'vis_448_process/seg_rec_image', '*.png'))}
        mask_paths = {os.path.basename(x): x for x in
                     glob(os.path.join(self.root, 'vis_448_process/seg_rec_mask', '*.png'))}
        # img_paths = {os.path.basename(x): x for x in glob(os.path.join(self.root, 'vis_448_process/seg_rec_image', '*.png'))}
        # print(img_paths)
        df['path'] = df['Image Index'].map(img_paths.get)
        df['maskpath'] = df['Image Index'].map(mask_paths.get)
        # print(df['path'])

        # 2. set train flag
        with open(os.path.join(self.root, 'train_val_list.txt'), 'rt') as f:
            train_flag = {x.strip('\n'): 1 for x in f.readlines()}

        with open(os.path.join(self.root, 'test_list.txt'), 'rt') as f:
            train_flag.update({x.strip('\n'): 0 for x in f.readlines()})

        df['train'] = df['Image Index'].map(train_flag.get)

        # 3. change label to one-hot label
        df['Finding Labels'] = df['Finding Labels'].map(lambda x: x.replace('No Finding', ''))
        all_labels = np.unique(list(chain(*df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
        all_labels = [x for x in all_labels if len(x) > 0]

        for c_label in all_labels:
            if len(c_label) > 1:  # leave out empty labels
                df[c_label] = df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

        # 4. split dataset into train, test
        df_split = df[df['train'] == (1 if mode == 'train' else 0)]
        self.x = df_split['path'].values.tolist()
        self.x_mask = df_split['maskpath'].values.tolist()
        self.y = df_split[all_labels].values
        self.classes = all_labels
        self.norm_weight = np.array(self.y).sum(axis=0) / (np.array(self.y).sum(axis=0) ** 2).sum() ** 0.5
        # self.weight = (1 / np.array(self.y).mean(axis=0)) / (1 / np.array(self.y).sum(axis=0)).mean()
        self.weight = np.stack([1 / (np.array(self.y) == 0).astype(float).sum(axis=0),
                                1 / (np.array(self.y) == 1).astype(float).sum(axis=0)], axis=1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # print(self.x[idx])
        labels = []
        image = Image.open(self.x[idx]).convert('RGB')
        label = np.asarray(self.y[idx]).astype(np.float32)
        mask_image = Image.open(self.x_mask[idx])
        choosen_index = 0
        while (choosen_index) == 0:
            randomidx = random.randint(0, len(self.x) - 1)
            target2 = np.asarray(self.y[randomidx]).astype(np.float32)
            # 7代表健康
            if np.max(target2) == 0:
                choosen_index = 1
                mixed_image = Image.open(self.x[randomidx]).convert('RGB')
                mixed_mask_image = Image.open(self.x_mask[randomidx])
        if self.transform != None:
            if self.transform != None:
                mixed_imgs = self.transform(image, mask_image, mixed_image, mixed_mask_image)
                for ii in range(len(mixed_imgs)):
                    labels.append(label)
                return mixed_imgs, labels
        if self.transform:
            image = self.transform(image)
        return image, label