import os
import sys
import numpy as np
from PIL import Image
import random
import pyvips
import pandas as pd
import glob
import re
import pickle
import h5py

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch


class ImageDataset(Dataset):

    def __init__(self, data_path, img_size, mask_path, patch_size, df_path, mil, sp):
        self._data_path = data_path
        self.sp = sp
        self._img_size = img_size
        self.patch_size = patch_size
        self.mask_path = mask_path
        self.df_path = df_path
        self.mil = mil
        self._pre_process()
        mean = (0.5, 0.5, 0.5)
        std = (0.1, 0.1, 0.1)
        self.trans = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])

    def _pre_process(self):
        df = pd.read_csv(self.df_path)
        slide_names = df[self.sp].dropna()

        # make dataset
        self._items = []
        for root, _, fnames in sorted(os.walk(self._data_path)):
            name = os.path.basename(root)
            for fname in sorted(fnames):
                if fname.split('.')[-1] == 'png' and name in slide_names.values:
                    path = os.path.join(root, fname)
                    wsi_class = int(df[df[self.sp] == name][self.sp + '_label'])
                    img_mask_path = self.mask_path + name + '/' + fname.split('.')[0] + '.npy'
                    if not self.mil:
                        coarse_anno = int(df[df[self.sp] == name]['Anno'])
                    else:
                        coarse_anno = 1
                    item = (path, wsi_class, img_mask_path, coarse_anno)
                    self._items.append(item)

        # random.shuffle(self._items)

        self._num_images = len(self._items)

    def __len__(self):
        return self._num_images

    def __getitem__(self, idx):
        path, wsi_class, mask_path, coarse_anno = self._items[idx]

        if os.path.exists(mask_path):
            token_label = np.load(mask_path)
            token_label[token_label > 0] = wsi_class  # delete if not using background as class 0, need to +1 if in TCGA
        else:
            token_label = np.zeros((32, 32))

        if np.sum(token_label > 0) > 10:
            label = torch.tensor(int(wsi_class), dtype=torch.long)
        else:
            label = torch.tensor(0, dtype=torch.long)

        img = pyvips.Image.new_from_file(path, memory=False)

        img, token_label = self.rotate(img, token_label)
        img, token_label = self.random_flip(img, token_label)
        img = self.color_jitter(img)

        mask = torch.tensor(token_label, dtype=torch.long)
        tensor = np.ndarray(buffer=img.cast(pyvips.BandFormat.UCHAR).write_to_memory(),
                            dtype=np.uint8,
                            shape=[img.height, img.width, 3])
        tensor = self.trans(tensor)
        del img
        del token_label
        return tensor, label, mask, coarse_anno

    def color_jitter(self, image, delta=0.05):
        image = image.colourspace("lch")
        luminance_diff = random.uniform(1.0-delta, 1.0+delta)  # roughly brightness
        chroma_diff = random.uniform(1.0-delta, 1.0+delta)  # roughly saturation
        hue_diff = random.uniform(1.0-delta, 1.0+delta)  # hue
        image *= [luminance_diff, chroma_diff, hue_diff]
        image = image.colourspace("srgb")
        return image

    def random_flip(self, image, mask):
        if random.choice([True, False]):
            image = image.fliphor()
            mask = mask[::-1].reshape(-1)[::-1].reshape(mask.shape[0], mask.shape[1]).copy()
        if random.choice([True, False]):
            image = image.flipver()
            mask = mask[::-1].copy()
        return image, mask

    def rotate(self, img, mask):
        if random.choice([True, False]):
            img = img.rotate(90)
            mask = np.rot90(mask, k=-1).copy()
        elif random.choice([True, False]):
            img = img.rotate(180)
            mask = np.rot90(mask, k=-2).copy()
        elif random.choice([True, False]):
            img = img.rotate(270)
            mask = np.rot90(mask, k=-3).copy()
        return img, mask


class ValideImageDataset(Dataset):

    def __init__(self, args, df_path, ratio, epoch, split='train'):
        self._data_path = args.data_path
        self.df = pd.read_csv(df_path)
        self.split = split
        self.ratio = ratio
        self.epoch = epoch
        mean = (0.5, 0.5, 0.5)
        std = (0.1, 0.1, 0.1)
        self.trans = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
        self._pre_process()

    def _pre_process(self):
        random.seed(self.epoch)
        slide_names = self.df[self.split].dropna()

        # make dataset
        self._items = []
        for root, _, _ in sorted(os.walk(self._data_path)):
            name = os.path.basename(root)
            if name in slide_names.values:
                path = glob.glob(os.path.join(root, "*.png"))
                if len(path) < 2:
                    num = len(path)
                else:
                    num = int(len(path) * self.ratio)
                paths = random.sample(path, num)
                item = list(zip(paths, [name] * num))
                self._items += item

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        path, name = self._items[idx]

        img = pyvips.Image.new_from_file(path, memory=True)

        tensor = np.ndarray(buffer=img.cast(pyvips.BandFormat.UCHAR).write_to_memory(),
                            dtype=np.uint8,
                            shape=[img.height, img.width, 3])
        tensor = self.trans(tensor)
        return tensor, name


class TestImageDataset(Dataset):

    def __init__(self, slide_folder):
        self._data_path = slide_folder
        mean = (0.5, 0.5, 0.5)
        std = (0.1, 0.1, 0.1)
        self.trans = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
        self._pre_process()

    def _pre_process(self):
        pattern = re.compile(r"(\d+)_(\d+)\.png")

        # make dataset
        self._items = []
        for root, _, fnames in sorted(os.walk(self._data_path)):
            for fname in sorted(fnames):
                if fname.split('.')[-1] == 'png':
                    path = os.path.join(root, fname)

                    match = pattern.match(fname)
                    x1 = int(match.group(1))
                    x2 = int(match.group(2))
                    coord = [x1, x2]
                    item = (path, coord)
                    self._items.append(item)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        path, coord = self._items[idx]

        img = pyvips.Image.new_from_file(path, memory=True)

        tensor = np.ndarray(buffer=img.cast(pyvips.BandFormat.UCHAR).write_to_memory(),
                            dtype=np.uint8,
                            shape=[img.height, img.width, 3])
        tensor = self.trans(tensor)
        return tensor, coord


class WSIDataset(Dataset):
    def __init__(self, data_path, df, h5, splits):
        self._data_path = data_path
        self.df = df
        self.sp = splits
        self.h5 = h5
        self._pre_process()

    def _pre_process(self):
        slide_names = self.df[self.sp].dropna()

        self._items = []
        for slide_name in slide_names.values:
            if self.h5:
                slide = slide_name + '.h5'
            else:
                slide = slide_name + '.pkl'
            path = os.path.join(self._data_path, slide)
            wsi_class = int(self.df[self.df[self.sp] == slide_name][self.sp + '_label'])
            item = (path, wsi_class)
            self._items.append(item)

        self._num_slides = len(self._items)

    def __len__(self):
        return self._num_slides

    def __getitem__(self, idx):
        path, label = self._items[idx]
        if self.h5:
            with h5py.File(path, 'r') as hdf5_file:
                feature = hdf5_file['features'][:]
        else:
            with open(path, 'rb') as f:
                wsi = pickle.load(f)
                feature = wsi['feature']

        label = torch.tensor(int(label), dtype=torch.long)
        wsi = torch.tensor(feature.reshape(-1, feature.shape[-1])).type(torch.float32)
        return wsi, label
