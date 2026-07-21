"""RRSHID dataset helpers (TN/M/TK = thin/moderate/thick haze)."""
import os
import random

from PIL import Image
import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}

# density key -> (hazy folder, GT folder, display name)
RRSHID_LEVELS = {
    'tn': ('RRSHID-TN', 'RRSHID-TN-GT', 'Thin/浅'),
    'm': ('RRSHID-M', 'RRSHID-M-GT', 'Moderate/中'),
    'tk': ('RRSHID-TK', 'RRSHID-TK-GT', 'Thick/浓'),
}

DENSITY_ALIASES = {
    'thin': 'tn', 'light': 'tn', '浅': 'tn', 'tn': 'tn',
    'moderate': 'm', 'mid': 'm', 'medium': 'm', '中': 'm', 'm': 'm',
    'thick': 'tk', 'heavy': 'tk', 'dense': 'tk', '浓': 'tk', 'tk': 'tk',
    'all': 'all',
}


def normalize_density(name):
    key = name.lower().strip()
    if key not in DENSITY_ALIASES:
        raise ValueError(f'unknown density "{name}", use tn/m/tk/all or 浅/中/浓')
    return DENSITY_ALIASES[key]


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


def _join_if_exists(base, name):
    path = os.path.join(base, name)
    return path if os.path.isdir(path) else None


def discover_rrshid(data_dir):
    """
    Find RRSHID hazy/GT folder pairs under data_dir.
    Supports official flat layout and survey Restored/Ground-truth layout.
    Returns: { 'tn': (hazy_dir, clear_dir), 'm': ..., 'tk': ... }
    """
    data_dir = os.path.abspath(data_dir)
    search_roots = [
        (data_dir, data_dir),
        (os.path.join(data_dir, 'Restored'), os.path.join(data_dir, 'Ground-truth')),
        (os.path.join(data_dir, 'haze'), os.path.join(data_dir, 'GT')),
        (os.path.join(data_dir, 'hazy'), os.path.join(data_dir, 'GT')),
    ]

    for hazy_base, gt_base in search_roots:
        found = {}
        for key, (hazy_name, gt_name, _) in RRSHID_LEVELS.items():
            hazy_dir = _join_if_exists(hazy_base, hazy_name)
            clear_dir = _join_if_exists(gt_base, gt_name)
            if hazy_dir and clear_dir:
                found[key] = (hazy_dir, clear_dir)
        if len(found) == 3:
            print(f'[RRSHID] found all 3 densities under {data_dir}')
            for key, (h, c) in found.items():
                print(f'  {RRSHID_LEVELS[key][2]}: {h}  <->  {c}')
            return found

    raise FileNotFoundError(
        f'RRSHID folders not found under {data_dir}\n'
        'Expected one of:\n'
        '  {data_dir}/RRSHID-TN + {data_dir}/RRSHID-TN-GT  (and M/TK)\n'
        '  {data_dir}/Restored/RRSHID-TN + {data_dir}/Ground-truth/RRSHID-TN-GT\n'
    )


def split_pairs(pairs, val_ratio=0.1, seed=42):
    pairs = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_ratio)) if pairs else 0
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    return train_pairs, val_pairs


def merge_pairs(level_map, densities=('tn', 'm', 'tk')):
    all_pairs = []
    for key in densities:
        hazy_dir, clear_dir = level_map[key]
        pairs = collect_pairs(hazy_dir, clear_dir)
        print(f'[RRSHID] {RRSHID_LEVELS[key][2]}: {len(pairs)} pairs')
        all_pairs.extend(pairs)
    return all_pairs


class PairedFolderDataset(data.Dataset):
    """Load paired images from two flat folders (RRSHID: 1.png <-> 1.png)."""

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


def build_rrshid_loaders(data_dir, val_ratio=0.1, crop_size=240, batch_size=16, crop=False):
    """Build train/val loaders for all densities + per-density test loaders."""
    level_map = discover_rrshid(data_dir)
    all_pairs = merge_pairs(level_map)
    train_pairs, val_pairs = split_pairs(all_pairs, val_ratio=val_ratio)

    train_size = crop_size if crop else 'whole img'

    from torch.utils.data import DataLoader

    loaders = {
        'rrshid_train': DataLoader(
            PairedFolderDataset(train_pairs, train=True, size=train_size),
            batch_size=batch_size, shuffle=True,
        ),
        'rrshid_test': DataLoader(
            PairedFolderDataset(val_pairs, train=False, size='whole img'),
            batch_size=1, shuffle=False,
        ),
    }

    for key in ('tn', 'm', 'tk'):
        pairs = collect_pairs(level_map[key][0], level_map[key][1])
        loaders[f'rrshid_{key}_test'] = DataLoader(
            PairedFolderDataset(pairs, train=False, size='whole img'),
            batch_size=1, shuffle=False,
        )
        loaders[f'rrshid_{key}_train'] = DataLoader(
            PairedFolderDataset(pairs, train=True, size=train_size),
            batch_size=batch_size, shuffle=True,
        )

    print(f'[RRSHID] train={len(train_pairs)} val={len(val_pairs)} (val_ratio={val_ratio})')
    return loaders
