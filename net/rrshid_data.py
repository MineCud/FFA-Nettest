"""RRSHID-style fog dataset: thin_fog / moderate_fog / thick_fog (浅/中/浓).

Expected layout under data_dir:
  thin_fog/train/hazy + train/{gt|GT|clear}
  thin_fog/val/hazy   + val/{gt|GT|clear}
  thin_fog/test/hazy  + test/{gt|GT|clear}
  (same for moderate_fog, thick_fog)
"""
import os
import random

from PIL import Image
import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

# key -> (folder name, display)
FOG_LEVELS = {
    'thin': ('thin_fog', 'Thin/浅'),
    'moderate': ('moderate_fog', 'Moderate/中'),
    'thick': ('thick_fog', 'Thick/浓'),
    'tn': ('thin_fog', 'Thin/浅'),
    'm': ('moderate_fog', 'Moderate/中'),
    'tk': ('thick_fog', 'Thick/浓'),
}

DENSITY_ALIASES = {
    'thin': 'thin', 'light': 'thin', '浅': 'thin', 'tn': 'thin',
    'moderate': 'moderate', 'mid': 'moderate', 'medium': 'moderate',
    '中': 'moderate', 'm': 'moderate',
    'thick': 'thick', 'heavy': 'thick', 'dense': 'thick', '浓': 'thick', 'tk': 'thick',
    'all': 'all',
}

CLEAR_DIR_NAMES = ('clear', 'GT', 'gt', 'Clear')


def normalize_density(name):
    key = name.lower().strip()
    if key not in DENSITY_ALIASES:
        raise ValueError(f'unknown density "{name}", use thin/moderate/thick/all or 浅/中/浓')
    return DENSITY_ALIASES[key]


def resolve_clear_dir(split_root):
    for name in CLEAR_DIR_NAMES:
        path = os.path.join(split_root, name)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f'No GT folder under {split_root}. Expected one of: {", ".join(CLEAR_DIR_NAMES)}'
    )


def list_images(folder):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    )


def collect_pairs(hazy_dir, clear_dir):
    pairs = []
    for haze_path in list_images(hazy_dir):
        name = os.path.basename(haze_path)
        stem, _ = os.path.splitext(name)
        clear_path = os.path.join(clear_dir, name)
        if not os.path.isfile(clear_path):
            for ext in IMG_EXTS:
                alt = os.path.join(clear_dir, stem + ext)
                if os.path.isfile(alt):
                    clear_path = alt
                    break
        if os.path.isfile(clear_path):
            pairs.append((haze_path, clear_path))
        else:
            print(f'WARNING: skip (no GT): {name}')
    return pairs


def split_dirs(data_dir, fog_key, split):
    """Return (hazy_dir, clear_dir) for one fog level and split."""
    fog_folder = FOG_LEVELS[fog_key][0]
    split_root = os.path.join(os.path.abspath(data_dir), fog_folder, split)
    hazy_dir = os.path.join(split_root, 'hazy')
    if not os.path.isdir(hazy_dir):
        raise FileNotFoundError(f'hazy dir not found: {hazy_dir}')
    clear_dir = resolve_clear_dir(split_root)
    return hazy_dir, clear_dir


def discover_rrshid(data_dir):
    """Verify all 3 fog levels exist. Return dict key -> fog_folder name."""
    data_dir = os.path.abspath(data_dir)
    found = {}
    for key in ('thin', 'moderate', 'thick'):
        fog_folder = FOG_LEVELS[key][0]
        fog_path = os.path.join(data_dir, fog_folder)
        if not os.path.isdir(fog_path):
            raise FileNotFoundError(f'missing fog folder: {fog_path}')
        for split in ('train', 'val', 'test'):
            split_dirs(data_dir, key, split)
        found[key] = fog_folder
        print(f'[RRSHID] OK {fog_folder} ({FOG_LEVELS[key][1]})')
    return found


def merge_split_pairs(data_dir, split, densities=('thin', 'moderate', 'thick')):
    pairs = []
    for key in densities:
        hazy_dir, clear_dir = split_dirs(data_dir, key, split)
        level_pairs = collect_pairs(hazy_dir, clear_dir)
        print(f'[RRSHID] {FOG_LEVELS[key][1]} {split}: {len(level_pairs)} pairs')
        pairs.extend(level_pairs)
    return pairs


class PairedFolderDataset(data.Dataset):
    def __init__(self, pairs, train, size='whole img'):
        self.pairs = pairs
        self.train = train
        self.size = size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        haze_path, clear_path = self.pairs[index]
        haze = Image.open(haze_path).convert('RGB')
        clear = Image.open(clear_path).convert('RGB')

        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, len(self.pairs) - 1)
                haze_path, clear_path = self.pairs[index]
                haze = Image.open(haze_path).convert('RGB')
                clear = Image.open(clear_path).convert('RGB')

        clear = FF.center_crop(clear, haze.size[::-1])

        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)

        if self.train:
            if random.randint(0, 1):
                haze = FF.hflip(haze)
                clear = FF.hflip(clear)
            rot = random.randint(0, 3)
            if rot:
                haze = FF.rotate(haze, 90 * rot)
                clear = FF.rotate(clear, 90 * rot)

        haze = tfs.Normalize(
            mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152]
        )(tfs.ToTensor()(haze))
        clear = tfs.ToTensor()(clear)
        return haze, clear


def _make_loader(pairs, train, size, batch_size, shuffle):
    from torch.utils.data import DataLoader
    return DataLoader(
        PairedFolderDataset(pairs, train=train, size=size),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def build_rrshid_loaders(data_dir, crop_size=240, batch_size=16, crop=False):
    discover_rrshid(data_dir)
    train_size = crop_size if crop else 'whole img'

    loaders = {}
    for split, train_flag, bs, shuffle in (
        ('train', True, batch_size, True),
        ('val', False, 1, False),
        ('test', False, 1, False),
    ):
        pairs = merge_split_pairs(data_dir, split)
        loaders[f'rrshid_{split}'] = _make_loader(
            pairs, train=train_flag,
            size=train_size if train_flag else 'whole img',
            batch_size=bs, shuffle=shuffle,
        )

    for key in ('thin', 'moderate', 'thick'):
        short = {'thin': 'tn', 'moderate': 'm', 'thick': 'tk'}[key]
        for split, train_flag, bs, shuffle in (
            ('train', True, batch_size, True),
            ('val', False, 1, False),
            ('test', False, 1, False),
        ):
            hazy_dir, clear_dir = split_dirs(data_dir, key, split)
            pairs = collect_pairs(hazy_dir, clear_dir)
            loader = _make_loader(
                pairs, train=train_flag,
                size=train_size if train_flag else 'whole img',
                batch_size=bs, shuffle=shuffle,
            )
            loaders[f'rrshid_{short}_{split}'] = loader
            loaders[f'rrshid_{key}_{split}'] = loader

    total_train = len(loaders['rrshid_train'].dataset)
    total_val = len(loaders['rrshid_val'].dataset)
    total_test = len(loaders['rrshid_test'].dataset)
    print(f'[RRSHID] total train={total_train} val={total_val} test={total_test}')
    return loaders
