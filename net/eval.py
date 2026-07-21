"""
Evaluate FFA-Net on SOTS or custom paired hazy/clear folders (e.g. remote sensing).

SOTS benchmark:
  python eval.py --task its --data_dir /path/to/data
  python eval.py --task ots --data_dir /path/to/data

Custom paired dataset (same filename in hazy/ and clear/):
  python eval.py --custom --hazy_dir /data/rs/hazy --clear_dir /data/rs/clear --task ots

Custom with SOTS-style id matching (1400_1.png -> 1400.png):
  python eval.py --custom --hazy_dir ... --clear_dir ... --pair_mode sots_id
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as tfs
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as FF

from metrics import psnr, ssim
from models import FFA

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def list_images(folder):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    )


def pair_paths(hazy_paths, clear_dir, pair_mode, clear_ext='.png'):
    pairs = []
    for haze_path in hazy_paths:
        name = os.path.basename(haze_path)
        stem, ext = os.path.splitext(name)
        if pair_mode == 'same_name':
            clear_path = os.path.join(clear_dir, name)
            if not os.path.isfile(clear_path):
                for alt in IMG_EXTS:
                    alt_path = os.path.join(clear_dir, stem + alt)
                    if os.path.isfile(alt_path):
                        clear_path = alt_path
                        break
        elif pair_mode == 'sots_id':
            img_id = stem.split('_')[0]
            clear_path = os.path.join(clear_dir, img_id + clear_ext)
            if not os.path.isfile(clear_path):
                for alt in IMG_EXTS:
                    alt_path = os.path.join(clear_dir, img_id + alt)
                    if os.path.isfile(alt_path):
                        clear_path = alt_path
                        break
        else:
            raise ValueError(f'unknown pair_mode: {pair_mode}')

        if not os.path.isfile(clear_path):
            print(f'WARNING: skip (no GT): {name}')
            continue
        pairs.append((haze_path, clear_path))
    return pairs


class PairedDehazeDataset(data.Dataset):
    """Paired hazy/clear images for PSNR evaluation."""

    def __init__(self, pairs, max_size=0):
        self.pairs = pairs
        self.max_size = max_size

    def _load_rgb(self, path):
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def _resize_if_needed(self, haze, clear):
        if self.max_size <= 0:
            return haze, clear
        w, h = haze.size
        long_side = max(w, h)
        if long_side <= self.max_size:
            return haze, clear
        scale = self.max_size / long_side
        new_w, new_h = int(w * scale), int(h * scale)
        haze = haze.resize((new_w, new_h), Image.BILINEAR)
        clear = clear.resize((new_w, new_h), Image.BILINEAR)
        return haze, clear

    def __getitem__(self, index):
        haze_path, clear_path = self.pairs[index]
        haze = self._load_rgb(haze_path)
        clear = self._load_rgb(clear_path)
        if clear.size != haze.size:
            clear = FF.center_crop(clear, haze.size[::-1])
        haze, clear = self._resize_if_needed(haze, clear)

        haze_t = tfs.Normalize(
            mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152]
        )(tfs.ToTensor()(haze))
        clear_t = tfs.ToTensor()(clear)
        return haze_t, clear_t, os.path.basename(haze_path)

    def __len__(self):
        return len(self.pairs)


def run_eval(net, loader, device, save_dir=''):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    ssims, psnrs, names = [], [], []
    with torch.no_grad():
        for i, (inputs, targets, batch_names) in enumerate(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            pred = net(inputs)
            ssim_v = ssim(pred, targets).item()
            psnr_v = psnr(pred, targets)
            name = batch_names[0]
            ssims.append(ssim_v)
            psnrs.append(psnr_v)
            names.append(name)

            if save_dir:
                stem = os.path.splitext(name)[0]
                vutils.save_image(pred.clamp(0, 1).cpu(), os.path.join(save_dir, f'{stem}_FFA.png'))

            if (i + 1) % 20 == 0 or i == 0:
                print(f'  [{i + 1}/{len(loader.dataset)}] {name}  psnr={psnr_v:.4f}  ssim={ssim_v:.4f}')

    return names, psnrs, ssims


parser = argparse.ArgumentParser()
parser.add_argument('--custom', action='store_true',
                    help='Use custom hazy/clear folders instead of SOTS')
parser.add_argument('--hazy_dir', type=str, default='',
                    help='Folder of hazy images (required with --custom)')
parser.add_argument('--clear_dir', type=str, default='',
                    help='Folder of clear GT images (required with --custom)')
parser.add_argument('--pair_mode', type=str, default='same_name',
                    choices=['same_name', 'sots_id'],
                    help='same_name: match by filename; sots_id: 1400_1.png -> 1400.png')
parser.add_argument('--clear_ext', type=str, default='.png',
                    help='GT extension when pair_mode=sots_id')
parser.add_argument('--max_size', type=int, default=0,
                    help='Resize long side to this limit (0=keep original). Useful for large RS tiles.')
parser.add_argument('--save_dir', type=str, default='',
                    help='Save dehazed predictions to this folder')
parser.add_argument('--task', type=str, default='ots', choices=['its', 'ots'],
                    help='Weight to load: its (indoor) or ots (outdoor, recommended for RS)')
parser.add_argument('--rrshid', action='store_true',
                    help='Evaluate on RRSHID (reports PSNR per density: TN/M/TK)')
parser.add_argument('--density', type=str, default='all',
                    help='Fog level: all, thin/浅, moderate/中, thick/浓')
parser.add_argument('--split', type=str, default='test', choices=['test', 'val'],
                    help='Which split to evaluate: test (default) or val')
parser.add_argument('--data_dir', type=str, default='',
                    help='Dataset root (default: ../dataset)')
parser.add_argument('--gps', type=int, default=3)
parser.add_argument('--blocks', type=int, default=19)
parser.add_argument('--model_dir', type=str, default='')
opt = parser.parse_args()

default_model = f'trained_models/{opt.task}_train_ffa_{opt.gps}_{opt.blocks}.pk'
model_path = opt.model_dir or default_model
if not os.path.isfile(model_path):
    print(f'ERROR: checkpoint not found: {model_path}')
    print('Download from Google Drive / Baidu (see trained_models/readme.md)')
    sys.exit(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {device}')
print(f'model: {model_path}')
if opt.max_size > 0:
    print(f'max_size: {opt.max_size} (long side)')

ckp = torch.load(model_path, map_location=device)
net = FFA(gps=opt.gps, blocks=opt.blocks)
if device == 'cuda':
    net = nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net = net.to(device)
net.eval()


def eval_loader(loader, label, save_subdir=''):
    sub_save = os.path.join(opt.save_dir, save_subdir) if opt.save_dir and save_subdir else opt.save_dir
    names, psnrs, ssims = run_eval(net, loader, device, save_dir=sub_save)
    mean_psnr = np.mean(psnrs)
    mean_ssim = np.mean(ssims)
    print('-' * 50)
    print(f'{label}: PSNR = {mean_psnr:.4f}  SSIM = {mean_ssim:.4f}  (n={len(psnrs)})')
    return mean_psnr, mean_ssim


if opt.rrshid:
    from rrshid_data import (
        FOG_LEVELS, collect_pairs, discover_rrshid, normalize_density, split_dirs,
    )
    from torch.utils.data import DataLoader

    if opt.data_dir:
        data_dir = os.path.abspath(opt.data_dir)
    else:
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))

    density = normalize_density(opt.density)
    discover_rrshid(data_dir)

    if density == 'all':
        targets = [
            ('thin', 'thin_fog/Thin/浅'),
            ('moderate', 'moderate_fog/Moderate/中'),
            ('thick', 'thick_fog/Thick/浓'),
        ]
    else:
        folder, label = FOG_LEVELS[density]
        targets = [(density, f'{folder}/{label}')]

    all_psnr, all_ssim = [], []
    for key, label in targets:
        hazy_dir, clear_dir = split_dirs(data_dir, key, opt.split)
        pairs = collect_pairs(hazy_dir, clear_dir)
        loader = DataLoader(
            PairedDehazeDataset(pairs, max_size=opt.max_size),
            batch_size=1, shuffle=False,
        )
        psnr_v, ssim_v = eval_loader(loader, f'{label} [{opt.split}]', save_subdir=f'{key}_{opt.split}')
        all_psnr.append(psnr_v)
        all_ssim.append(ssim_v)

    if len(targets) > 1:
        print('=' * 50)
        print(f'Overall [{opt.split}] (mean of 3 levels): PSNR = {np.mean(all_psnr):.4f}  SSIM = {np.mean(all_ssim):.4f}')
    sys.exit(0)

if opt.custom:
    if not opt.hazy_dir or not opt.clear_dir:
        print('ERROR: --custom requires --hazy_dir and --clear_dir')
        sys.exit(1)
    hazy_dir = os.path.abspath(opt.hazy_dir)
    clear_dir = os.path.abspath(opt.clear_dir)
    if not os.path.isdir(hazy_dir) or not os.path.isdir(clear_dir):
        print(f'ERROR: directory not found:\n  hazy: {hazy_dir}\n  clear: {clear_dir}')
        sys.exit(1)
    pairs = pair_paths(list_images(hazy_dir), clear_dir, opt.pair_mode, opt.clear_ext)
    if not pairs:
        print('ERROR: no valid hazy/clear pairs found')
        sys.exit(1)
    dataset = PairedDehazeDataset(pairs, max_size=opt.max_size)
    dataset_label = f'custom ({len(pairs)} pairs)'
    data_info = f'hazy={hazy_dir}\nclear={clear_dir}'
else:
    if not opt.data_dir:
        print('ERROR: SOTS mode requires --data_dir, or use --custom for RS data')
        sys.exit(1)
    test_path = os.path.join(os.path.abspath(opt.data_dir), 'RESIDE', 'SOTS',
                             'indoor' if opt.task == 'its' else 'outdoor')
    if not os.path.isdir(test_path):
        print(f'ERROR: SOTS not found: {test_path}')
        sys.exit(1)
    hazy_paths = list_images(os.path.join(test_path, 'hazy'))
    pairs = pair_paths(hazy_paths, os.path.join(test_path, 'clear'), 'sots_id', '.png')
    dataset = PairedDehazeDataset(pairs)
    dataset_label = f'SOTS {opt.task}'
    data_info = test_path

loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

print(f'dataset: {dataset_label}')
print(data_info)

names, psnrs, ssims = run_eval(net, loader, device, save_dir=opt.save_dir)

mean_psnr = np.mean(psnrs)
mean_ssim = np.mean(ssims)
print('-' * 50)
print(f'{dataset_label}: PSNR = {mean_psnr:.4f}  SSIM = {mean_ssim:.4f}  (n={len(psnrs)})')

if opt.custom:
    print('\nPer-image results:')
    for name, p, s in zip(names, psnrs, ssims):
        print(f'  {name:40s}  PSNR={p:.4f}  SSIM={s:.4f}')
else:
    print('Paper reference — Indoor: 36.39/0.9886  Outdoor: 33.57/0.9840')
