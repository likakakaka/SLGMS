import random
import torch.utils.data as Data
from torchvision.transforms import transforms
import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
import numpy as np
import torch
class Fibrosis_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 txtpath,
                 data_transform=None,
                 seed=0
                 ):
        super(Fibrosis_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath

        self.txtpath = txtpath

        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform

        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()

    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        imgname=line.split('\t')[0]
        labelname = line
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)
        label = []
        label.append('Fibrosis' in labelname)
        label.append('No Finding' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        sample["lab"] = label
        sample["img"] = image
        sample["img_name"]=imgname
        return image, label, imgname
class Fibrosis_Mixed_Mask_Dataset_DINO(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 data_transform=None,
                 seed=0
                 ):
        super(Fibrosis_Mixed_Mask_Dataset_DINO, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath = maskimgpath
        self.txtpath = txtpath

        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform

        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()

    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample = {}
        labels = []
        sample["idx"] = idx
        line=self.lines[idx]
        imgname=line.split('\t')[0]
        labelname = line
        if '.png' in imgname:
            img_path = os.path.join(self.imgpath, imgname)
            mask_path = os.path.join(self.maskimgpath, imgname)
        else:
            img_path = os.path.join(self.imgpath, imgname + '.png')
            mask_path = os.path.join(self.maskimgpath, imgname + '.png')
        image = Image.open(img_path).convert('RGB')
        # image = self.transforms(image)

        mask_image = Image.open(mask_path)
        # aa=np.array(mask_image)
        # image, mask_image = self.transforms(image, mask_image, imgname)
        # image = self.transforms(image)
        label = []
        label.append('Fibrosis' in labelname)
        label.append('No Finding' in labelname)

        choosen_index = 0
        while (choosen_index) == 0:
            randomidx = random.randint(0, len(self.lines)-1)
            line = self.lines[randomidx]
            if 'No Finding' in line:
                choosen_index = 1
                imgname2 = line.split('\t')[0]
                img_path = os.path.join(self.imgpath, imgname2.split('.png')[0] + '.png')
                mixed_image = Image.open(img_path).convert('RGB')
                mixed_mask_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
                mixed_mask_image = Image.open(mixed_mask_path)

                # mixed_image = self.transforms(mixed_image)
        mixed_imgs= self.transforms(image, mask_image, mixed_image, mixed_mask_image, imgname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        for ii in range(len(mixed_imgs)):
            labels.append(label)
        # sample["lab"] = label
        # sample["img"] = image
        # sample["img_name"]=imgname
        return mixed_imgs,labels,imgname
class Shanxi_Dataset_DINO(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 txtpath,
                 data_transform=None,
                 seed=0
                 ):
        super(Shanxi_Dataset_DINO, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.txtpath = txtpath
        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform

        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()

        ####### pathology masks ########
        # Get our classes.

        # self.tr = transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(DSIZE)])
    def __len__(self):
        return len(self.lines)
    def shuffle_list(self, list):
        random.shuffle(list)
    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        line=self.lines[idx]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        img_path = os.path.join(self.imgpath, imgname.split('.png')[0] + '.png')
        image = Image.open(img_path).convert('RGB')
        image = self.transforms(image)
        label = []
        labels=[]
        label.append('Sick' in labelname)
        label.append('Health' in labelname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        if isinstance(image, list):
            for ii in range(len(image)):
                labels.append(label)
            sample["lab"] = label
            sample["img"] = image
            sample["img_name"] = imgname
            return image, labels, imgname
        else:
            return image, label, imgname
class Fibrosis_Mask_Dataset_DINO(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 data_transform=None,
                 seed=0
                 ):
        super(Fibrosis_Mask_Dataset_DINO, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath = maskimgpath
        self.txtpath = txtpath

        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform
        # if self.train:
        #     self.data_aug = transforms.Compose([
        #         # xrv.datasets.ToPILImage(),
        #         # transforms.RandomAffine(45,
        #         #                         translate=(0.15, 0.15),
        #         #                         scale=(1.0 - 0.15, 1.0 + 0.15)),
        #         transforms.ToTensor()
        #     ])

        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()

        ####### pathology masks ########
        # Get our classes.

        # self.tr = transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(DSIZE)])

    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample = {}
        labels = []
        sample["idx"] = idx
        line=self.lines[idx]
        imgname=line.split('\t')[0]
        labelname = line
        if '.png' in imgname:
            img_path = os.path.join(self.imgpath, imgname)
            mask_path = os.path.join(self.maskimgpath, imgname)
        else:
            img_path = os.path.join(self.imgpath, imgname + '.png')
            mask_path = os.path.join(self.maskimgpath, imgname + '.png')
        image = Image.open(img_path).convert('RGB')
        # image = self.transforms(image)

        mask_image = Image.open(mask_path)
        # aa=np.array(mask_image)
        # image, mask_image = self.transforms(image, mask_image, imgname)
        # image = self.transforms(image)
        label = []
        label.append('Fibrosis' in labelname)
        label.append('No Finding' in labelname)

        # choosen_index = 0
        # while (choosen_index) == 0:
        #     randomidx = random.randint(0, len(self.lines)-1)
        #     line = self.lines[randomidx]
        #     if 'No Finding' in line:
        #         choosen_index = 1
        #         imgname2 = line.split('\t')[0]
        #         img_path = os.path.join(self.imgpath, imgname2.split('.png')[0] + '.png')
        #         mixed_image = Image.open(img_path).convert('RGB')
        #         mixed_mask_path = os.path.join(self.maskimgpath, imgname.split('.png')[0] + '.png')
        #         mixed_mask_image = Image.open(mixed_mask_path)

                # mixed_image = self.transforms(mixed_image)
        imgs= self.transforms(image, mask_image, imgname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        for ii in range(len(imgs)):
            labels.append(label)
        return imgs,labels,imgname
class Shanxi_Mixed_Mask_Dataset_DINO(torch.utils.data.Dataset):
    def __init__(self,
                 imgpath,
                 maskimgpath,
                 txtpath,
                 data_transform=None,
                 seed=0
                 ):
        super(Shanxi_Mixed_Mask_Dataset_DINO, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.maskimgpath = maskimgpath
        self.txtpath = txtpath

        if data_transform==None:
            self.transforms = transforms.Compose(
                [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])
        else:
            self.transforms=data_transform
        # Load data
        with open(txtpath, 'r', encoding='gbk') as file:
            self.lines = file.readlines()

    def __len__(self):
        return len(self.lines)

    def shuffle_list(self, list):
        random.shuffle(list)

    def __getitem__(self, idx):
        sample = {}
        labels = []
        sample["idx"] = idx
        line=self.lines[idx]
        imgname=line.split('\n')[0]
        labelname = imgname.split('_')[0]
        if '.png' in imgname:
            img_path = os.path.join(self.imgpath, imgname)
            mask_path = os.path.join(self.maskimgpath, imgname)
        else:
            img_path = os.path.join(self.imgpath, imgname + '.png')
            mask_path = os.path.join(self.maskimgpath, imgname + '.png')
        image = Image.open(img_path).convert('RGB')

        mask_image = Image.open(mask_path)
        label = []
        label.append('Sick' in labelname)
        label.append('Health' in labelname)

        choosen_index = 0
        while (choosen_index) == 0:
            randomidx = random.randint(0, len(self.lines)-1)
            line = self.lines[randomidx]
            if 'Health' in line:
                choosen_index = 1
                imgname2 = line.split('\n')[0]
                img_path = os.path.join(self.imgpath, imgname2.split('.png')[0] + '.png')
                mixed_image = Image.open(img_path).convert('RGB')
                mixed_mask_path = os.path.join(self.maskimgpath, imgname + '.png')
                mixed_mask_image = Image.open(mixed_mask_path)

                # mixed_image = self.transforms(mixed_image)
        mixed_imgs= self.transforms(image, mask_image, mixed_image, mixed_mask_image, imgname)
        label = np.asarray(label).T
        label = label.astype(np.float32)
        for ii in range(len(mixed_imgs)):
            labels.append(label)
        # sample["lab"] = label
        # sample["img"] = image
        # sample["img_name"]=imgname
        return mixed_imgs,labels,imgname