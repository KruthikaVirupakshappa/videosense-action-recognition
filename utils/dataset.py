"""Dataset classes and video I/O helpers for VideoSense (Parts 1-4).

Tensor convention used everywhere in this project: a clip is a float tensor
of shape (C, T, H, W) -- channels first, then time -- which is what
``nn.Conv3d`` expects once a batch dimension is added: (B, C, T, H, W).

Two fixes relative to the reference code in the assignment handout:

1. ``testlist01.txt`` has NO label column (only ``Class/video.avi``), so the
   handout's ``if len(parts) >= 2`` silently drops every test video. Labels
   are therefore derived from the class folder in the path instead.
2. ``int(parts[1]) - 1`` is the 101-class index; on the 10-class mini subset
   that produces labels like 7 or 93 that do not match the 10 output units.
   Deriving the label from ``class_to_idx`` works for any class subset.
"""
import os
import random
from multiprocessing import Pool

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# ImageNet statistics (handout, Section 5.2) -- used for C3D / Siamese / fusion
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# Kinetics-400 statistics -- what torchvision's pretrained R(2+1)D expects
KINETICS_MEAN = [0.43216, 0.394666, 0.37645]
KINETICS_STD = [0.22803, 0.22145, 0.216989]

# 10-class mini subset from Section 4.2.3 of the handout
MINI_CLASSES = ['WalkingWithDog', 'Basketball', 'PushUps', 'Fencing',
                'HandstandWalking', 'Surfing', 'BabyCrawling', 'Archery',
                'PullUps', 'YoYo']

# Frames are cached at the classic C3D resolution (128x171 keeps the 4:3
# aspect ratio of 320x240 UCF101 videos); a 112x112 crop is taken at load time.
CACHE_HW = (128, 171)
CACHE_FRAMES = 32


# --------------------------------------------------------------------------
# Low-level video decoding
# --------------------------------------------------------------------------
def read_all_frames(path):
    """Decode every frame of a video sequentially (BGR, uint8).

    Sequential reading is used instead of ``cap.set(CAP_PROP_POS_FRAMES)``
    because seeking in XviD-encoded AVIs is frame-inaccurate and slow.
    UCF101 clips are short (~7 s), so holding them in memory is cheap.
    """
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def uniform_indices(total, num_frames):
    """``num_frames`` indices spread uniformly over ``[0, total - 1]``."""
    return np.linspace(0, max(total - 1, 0), num_frames).astype(int)


def segment_indices(total, num_frames):
    """TSN-style random sampling: one random index inside each of
    ``num_frames`` equal segments. Acts as temporal jitter during training."""
    edges = np.linspace(0, total, num_frames + 1)
    idx = [random.randint(int(lo), max(int(lo), int(np.ceil(hi)) - 1))
           for lo, hi in zip(edges[:-1], edges[1:])]
    return np.clip(np.array(idx), 0, total - 1)


def load_video_frames(path, num_frames=16, size_hw=CACHE_HW):
    """Uniformly sample ``num_frames`` RGB frames -> uint8 (T, H, W, 3)."""
    frames = read_all_frames(path)
    h, w = size_hw
    if not frames:
        return np.zeros((num_frames, h, w, 3), np.uint8)
    out = []
    for i in uniform_indices(len(frames), num_frames):
        rgb = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
        out.append(cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA))
    return np.stack(out)


# --------------------------------------------------------------------------
# Split files
# --------------------------------------------------------------------------
def list_classes(video_root):
    """Sorted class folder names under ``video_root``."""
    return sorted(d for d in os.listdir(video_root)
                  if os.path.isdir(os.path.join(video_root, d)))


def read_split(split_file, classes):
    """Return ``[(relative_path, class_name), ...]`` for videos in ``classes``.

    Works for both trainlist (``path label``) and testlist (``path``) files.
    """
    wanted = set(classes)
    entries = []
    with open(split_file) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            rel = parts[0].replace('\\', '/')
            cls = rel.split('/')[0]
            if cls in wanted:
                entries.append((rel, cls))
    return entries


# --------------------------------------------------------------------------
# Frame cache (one-time preprocessing, makes training I/O-cheap)
# --------------------------------------------------------------------------
def cache_path_for(cache_root, video_path):
    cls = os.path.basename(os.path.dirname(video_path))
    name = os.path.splitext(os.path.basename(video_path))[0] + '.npy'
    return os.path.join(cache_root, cls, name)


def _cache_one(job):
    video_path, out_path, num_frames, size_hw = job
    if os.path.exists(out_path):
        return
    cv2.setNumThreads(1)
    frames = load_video_frames(video_path, num_frames, size_hw)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, frames)


def cache_dataset_frames(video_root, cache_root, classes=None,
                         num_frames=CACHE_FRAMES, size_hw=CACHE_HW,
                         workers=8):
    """Decode every video of ``classes`` once and store uint8 frame stacks.

    Storing 32 frames (more than the 16 a clip uses) lets the training loader
    apply temporal jitter and lets evaluation run test-time augmentation.
    """
    classes = classes or list_classes(video_root)
    jobs = []
    for cls in classes:
        cls_dir = os.path.join(video_root, cls)
        for vid in sorted(os.listdir(cls_dir)):
            if vid.endswith('.avi'):
                vpath = os.path.join(cls_dir, vid)
                jobs.append((vpath, cache_path_for(cache_root, vpath),
                             num_frames, size_hw))
    with Pool(workers) as pool:
        for _ in tqdm(pool.imap_unordered(_cache_one, jobs, chunksize=4),
                      total=len(jobs), desc='Caching frames'):
            pass
    print(f'Frame cache ready: {len(jobs)} videos -> {cache_root}')


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------
def normalize_video(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Per-channel normalisation of a (3, T, H, W) tensor in [0, 1]."""
    mean = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1, 1)
    return (tensor - mean) / std


class VideoTransform:
    """uint8 (T, H, W, 3) frames -> normalised float tensor (3, T, S, S).

    Training: random spatial crop + random horizontal flip (the same crop and
    flip are applied to every frame so motion stays coherent).
    Evaluation: deterministic centre crop.
    """

    def __init__(self, crop_size=112, train=True,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.crop_size = crop_size
        self.train = train
        self.mean, self.std = mean, std

    def crop(self, frames):
        t, h, w, _ = frames.shape
        s = self.crop_size
        if h < s or w < s:
            frames = np.stack([cv2.resize(f, (max(w, s), max(h, s)))
                               for f in frames])
            t, h, w, _ = frames.shape
        if self.train:
            y, x = random.randint(0, h - s), random.randint(0, w - s)
        else:
            y, x = (h - s) // 2, (w - s) // 2
        frames = frames[:, y:y + s, x:x + s]
        if self.train and random.random() < 0.5:
            frames = frames[:, :, ::-1]
        return np.ascontiguousarray(frames)

    def __call__(self, frames):
        frames = self.crop(frames)
        # (T, H, W, 3) -> (3, T, H, W): the permute the handout highlights
        tensor = torch.from_numpy(frames).float().permute(3, 0, 1, 2) / 255.0
        return normalize_video(tensor, self.mean, self.std)


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
class UCF101Dataset(Dataset):
    """PyTorch Dataset for UCF101 clips.

    Args:
        root_dir: folder of class sub-folders containing ``.avi`` files.
        split_file: ``trainlist01.txt`` or ``testlist01.txt``.
        num_frames: frames per clip (T).
        spatial_size: crop size (H = W).
        transform: callable on uint8 (T, H, W, 3); defaults to VideoTransform.
        train: enables random temporal sampling + spatial augmentation.
        cache_dir: optional frame cache from ``cache_dataset_frames``.
        classes: optional explicit class list (defines label order).
    """

    def __init__(self, root_dir, split_file, num_frames=16, spatial_size=112,
                 transform=None, train=False, cache_dir=None, classes=None,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.spatial_size = spatial_size
        self.train = train
        self.cache_dir = cache_dir
        self.transform = transform or VideoTransform(spatial_size, train,
                                                     mean, std)
        self.classes = sorted(classes) if classes else list_classes(root_dir)
        self.samples, self.class_to_idx = self._load_split(split_file)
        # label -> list of sample indices (used by Pair/Triplet datasets and
        # by the few-shot evaluation protocol)
        self.class_idx = {}
        for i, (_, lbl) in enumerate(self.samples):
            self.class_idx.setdefault(lbl, []).append(i)

    def _load_split(self, split_file):
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        samples = []
        for rel, cls in read_split(split_file, self.classes):
            path = os.path.join(self.root_dir, rel)
            cached = (self.cache_dir and
                      os.path.exists(cache_path_for(self.cache_dir, path)))
            if cached or os.path.exists(path):
                samples.append((path, class_to_idx[cls]))
        return samples, class_to_idx

    def _load_video_frames(self, path, train=None):
        """Return uint8 (T, H, W, 3) frames for one clip."""
        train = self.train if train is None else train
        if self.cache_dir:
            cpath = cache_path_for(self.cache_dir, path)
            if os.path.exists(cpath):
                stack = np.load(cpath, mmap_mode='r')
                sampler = segment_indices if train else uniform_indices
                return np.asarray(stack[sampler(len(stack), self.num_frames)])
        return load_video_frames(path, self.num_frames, CACHE_HW)

    def raw_frames(self, idx):
        """Centre-cropped uint8 frames (T, S, S, 3) for visualisation."""
        frames = self._load_video_frames(self.samples[idx][0], train=False)
        return VideoTransform(self.spatial_size, train=False).crop(frames)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = self._load_video_frames(path)
        return self.transform(frames), label


class PairDataset(Dataset):
    """Pairs of clips with binary labels for contrastive training.

    label 0 = same action class (positive pair)
    label 1 = different action classes (negative pair)
    Positives and negatives are drawn 50/50.
    """

    def __init__(self, base_dataset, pairs_per_epoch=10000):
        self.ds = base_dataset
        self.pairs_per_epoch = pairs_per_epoch
        self.class_idx = base_dataset.class_idx
        self.classes = [c for c, idx in self.class_idx.items() if len(idx) > 1]

    def __len__(self):
        return self.pairs_per_epoch

    def __getitem__(self, _):
        if random.random() < 0.5:
            cls = random.choice(self.classes)
            i1, i2 = random.sample(self.class_idx[cls], 2)
            label = 0
        else:
            cls1, cls2 = random.sample(self.classes, 2)
            i1 = random.choice(self.class_idx[cls1])
            i2 = random.choice(self.class_idx[cls2])
            label = 1
        x1, _ = self.ds[i1]
        x2, _ = self.ds[i2]
        return x1, x2, torch.tensor(label, dtype=torch.float32)


class TripletDataset(Dataset):
    """(anchor, positive, negative) clip triplets for Triplet Loss."""

    def __init__(self, base_dataset, triplets_per_epoch=10000):
        self.ds = base_dataset
        self.n = triplets_per_epoch
        self.class_idx = base_dataset.class_idx
        self.classes = [c for c, idx in self.class_idx.items() if len(idx) > 1]

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        cls_a, cls_n = random.sample(self.classes, 2)
        ia, ip = random.sample(self.class_idx[cls_a], 2)
        ineg = random.choice(self.class_idx[cls_n])
        return self.ds[ia][0], self.ds[ip][0], self.ds[ineg][0]


class MultimodalDataset(Dataset):
    """Returns ``(rgb_clip, flow_clip, label)`` tuples.

    Flow clips are the pre-computed Farneback stacks of shape (2, T, H, W)
    produced by ``utils/optical_flow.py``.
    """

    def __init__(self, rgb_dataset, flow_root, normalize_flow=True,
                 train=None):
        self.rgb_ds = rgb_dataset
        self.flow_root = flow_root
        self.normalize_flow = normalize_flow
        self.train = rgb_dataset.train if train is None else train
        # Keep only videos whose flow has been computed
        self.indices = [i for i, (p, _) in enumerate(rgb_dataset.samples)
                        if os.path.exists(self.flow_path(p))]
        missing = len(rgb_dataset) - len(self.indices)
        if missing:
            print(f'[MultimodalDataset] {missing} videos have no flow file '
                  f'under {flow_root} and are skipped.')

    def flow_path(self, video_path):
        return cache_path_for(self.flow_root, video_path)

    def load_flow(self, video_path):
        flow = torch.from_numpy(np.load(self.flow_path(video_path)).copy())
        flow = flow.float()  # (2, T, H, W)
        if self.train and random.random() < 0.5:
            # Mirroring a flow field flips W and negates horizontal motion u
            flow = torch.flip(flow, dims=[3])
            flow[0] = -flow[0]
        if self.normalize_flow:
            # Normalise flow to [-1, 1] per clip
            flow = flow / (flow.abs().max() + 1e-6)
        return flow

    @property
    def samples(self):
        return [self.rgb_ds.samples[i] for i in self.indices]

    @property
    def classes(self):
        return self.rgb_ds.classes

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        rgb_clip, label = self.rgb_ds[real_idx]
        video_path, _ = self.rgb_ds.samples[real_idx]
        return rgb_clip, self.load_flow(video_path), label
