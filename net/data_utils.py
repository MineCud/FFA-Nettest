import os
import random
import sys

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as tfs
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import functional as FF
from torchvision.utils import make_grid

sys.path.append('.')
sys.path.append('..')

from metrics import *
from option import opt

BS = opt.bs
print('batch size:', BS)

crop_size = 'whole_img'
if opt.crop:
    crop_size = opt.crop_size

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def tensorShow(tensors, titles=None):
    fig = plt.figure()
    for tensor, tit, i in zip(tensors, titles, range(len(tensors))):
        img = make_grid(tensor)
        npimg = img.numpy()
        ax = fig.add_subplot(211 + i)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(tit)
    plt.show()


def resolve_clear_dir(root):
    """Find clear/GT folder under a split root."""
    for name in ('clear', 'GT', 'gt'):
        clear_dir = os.path.join(root, name)
        if os.path.isdir(clear_dir):
            return clear_dir
    raise FileNotFoundError(
        f'No clear/GT folder under {root}. Expected one of: clear/, GT/, gt/'
    )


def resolve_split_root(data_dir, split):
    """
    split: 'train' or 'val'
    Priority:
      1) {data_dir}/{split}/hazy
      2) {data_dir}/hazy  (flat layout, train only)
    """
    split_dir = os.path.join(data_dir, split)
    if os.path.isdir(os.path.join(split_dir, 'hazy')):
        return split_dir
    if split == 'train' and os.path.isdir(os.path.join(data_dir, 'hazy')):
        print(f'[data] use flat layout for train: {data_dir}/hazy + GT|clear')
        return data_dir
    raise FileNotFoundError(
        f'Train/val split not found: {split_dir}/hazy\n'
        f'Expected:\n'
        f'  {data_dir}/train/hazy + {data_dir}/train/GT(or clear)\n'
        f'  {data_dir}/val/hazy   + {data_dir}/val/GT(or clear)\n'
        f'Or flat train layout:\n'
        f'  {data_dir}/hazy + {data_dir}/GT(or clear)'
    )


def resolve_val_root(data_dir):
    val_dir = os.path.join(data_dir, 'val')
    if os.path.isdir(os.path.join(val_dir, 'hazy')):
        return val_dir
    if os.path.isdir(os.path.join(data_dir, 'hazy')):
        print('[data] WARNING: val/ not found, using same hazy/GT for train and val')
        return data_dir
    raise FileNotFoundError(f'Validation set not found under {data_dir}/val/hazy')


def find_clear_path(clear_dir, haze_path, pair_mode, gt_format='.png'):
    name = os.path.basename(haze_path)
    stem, _ = os.path.splitext(name)

    if pair_mode == 'same_name':
        candidate = os.path.join(clear_dir, name)
        if os.path.isfile(candidate):
            return candidate
        for ext in IMG_EXTS:
            alt = os.path.join(clear_dir, stem + ext)
            if os.path.isfile(alt):
                return alt
        raise FileNotFoundError(f'GT not found for {name} in {clear_dir}')

    img_id = stem.split('_')[0]
    candidate = os.path.join(clear_dir, img_id + gt_format)
    if os.path.isfile(candidate):
        return candidate
    for ext in IMG_EXTS:
        alt = os.path.join(clear_dir, img_id + ext)
        if os.path.isfile(alt):
            return alt
    raise FileNotFoundError(f'GT not found for {name} -> {img_id}* in {clear_dir}')


class RESIDE_Dataset(data.Dataset):
    def __init__(self, path, train, size=crop_size, format='.png', pair_mode='sots_id'):
        super(RESIDE_Dataset, self).__init__()
        self.size = size
        self.train = train
        self.format = format
        self.pair_mode = pair_mode
        self.hazy_dir = os.path.join(path, 'hazy')
        self.clear_dir = resolve_clear_dir(path)

        haze_names = sorted(
            f for f in os.listdir(self.hazy_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        )
        self.pairs = []
        for name in haze_names:
            haze_path = os.path.join(self.hazy_dir, name)
            try:
                clear_path = find_clear_path(
                    self.clear_dir, haze_path, pair_mode, format
                )
                self.pairs.append((haze_path, clear_path))
            except FileNotFoundError as e:
                print(f'WARNING: skip {name}: {e}')

        print(f'[{path}] crop={size} pair_mode={pair_mode} pairs={len(self.pairs)}')

    def __getitem__(self, index):
        haze_path, clear_path = self.pairs[index]
        haze = Image.open(haze_path)

        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, len(self.pairs) - 1)
                haze_path, clear_path = self.pairs[index]
                haze = Image.open(haze_path)

        clear = Image.open(clear_path)
        clear = FF.center_crop(clear, haze.size[::-1])

        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)

        haze, clear = self.augData(haze.convert('RGB'), clear.convert('RGB'))
        return haze, clear

    def augData(self, data, target):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        data = tfs.ToTensor()(data)
        data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
        target = tfs.ToTensor()(target)
        return data, target

    def __len__(self):
        return len(self.pairs)


# --- path config ---
pwd = os.getcwd()
print('cwd:', pwd)

data_dir = os.path.abspath(opt.data_dir)
print('data_dir:', data_dir)

# RESIDE (original)
reside_root = os.path.join(data_dir, 'RESIDE')
if os.path.isdir(reside_root):
    ITS_train_loader = DataLoader(
        dataset=RESIDE_Dataset(os.path.join(reside_root, 'ITS'), train=True, size=crop_size, pair_mode='sots_id'),
        batch_size=BS, shuffle=True,
    )
    ITS_test_loader = DataLoader(
        dataset=RESIDE_Dataset(os.path.join(reside_root, 'SOTS', 'indoor'), train=False, size='whole img', pair_mode='sots_id'),
        batch_size=1, shuffle=False,
    )
    OTS_train_loader = DataLoader(
        dataset=RESIDE_Dataset(os.path.join(reside_root, 'OTS'), train=True, format='.jpg', size=crop_size, pair_mode='sots_id'),
        batch_size=BS, shuffle=True,
    )
    OTS_test_loader = DataLoader(
        dataset=RESIDE_Dataset(os.path.join(reside_root, 'SOTS', 'outdoor'), train=False, size='whole img', format='.png', pair_mode='sots_id'),
        batch_size=1, shuffle=False,
    )
else:
    ITS_train_loader = ITS_test_loader = None
    OTS_train_loader = OTS_test_loader = None
    print('[data] RESIDE not found, its/ots loaders disabled')

# Remote sensing / custom paired data
rs_train_root = resolve_split_root(data_dir, 'train')
rs_val_root = resolve_val_root(data_dir)

RS_train_loader = DataLoader(
    dataset=RESIDE_Dataset(
        rs_train_root, train=True, size=crop_size,
        pair_mode=opt.pair_mode,
    ),
    batch_size=BS, shuffle=True,
)
RS_test_loader = DataLoader(
    dataset=RESIDE_Dataset(
        rs_val_root, train=False, size='whole img',
        pair_mode=opt.pair_mode,
    ),
    batch_size=1, shuffle=False,
)

if __name__ == '__main__':
    print('RS train pairs:', len(RS_train_loader.dataset))
    print('RS val pairs:', len(RS_test_loader.dataset))
