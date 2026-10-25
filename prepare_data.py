"""One-time preprocessing: verify the dataset, cache frames, compute flow.

    python prepare_data.py --verify                 # count videos / splits
    python prepare_data.py --frames mini base       # frame cache
    python prepare_data.py --flow                   # optical flow (mini)

``mini`` = the 10 classes of data/UCF101_mini (Parts 2, 4, 5).
``base`` = the Siamese base classes (results/base_classes.txt, created here
with the same seed as train_siamese.py if it does not exist yet).
"""
import argparse
import os
import random

from utils.dataset import (MINI_CLASSES, cache_dataset_frames, list_classes,
                           read_split)
from utils.optical_flow import preprocess_all_flows


def base_classes(video_root, n=30, seed=42,
                 path='results/base_classes.txt'):
    """Same selection rule as ``train_siamese.choose_base_classes``."""
    if os.path.exists(path):
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    pool = [c for c in list_classes(video_root) if c not in set(MINI_CLASSES)]
    base = sorted(random.Random(seed).sample(pool, min(n, len(pool))))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(base) + '\n')
    return base


def verify(video_root, split_dir):
    classes = list_classes(video_root)
    n = sum(len([v for v in os.listdir(os.path.join(video_root, c))
                 if v.endswith('.avi')]) for c in classes)
    print(f'{video_root}: {len(classes)} classes, {n} videos '
          f'(full UCF101 = 101 classes, 13,320 videos)')
    for name in ('trainlist01.txt', 'testlist01.txt'):
        entries = read_split(os.path.join(split_dir, name), classes)
        print(f'  {name}: {len(entries)} entries')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--video_root', default='data/UCF101')
    p.add_argument('--mini_root', default='data/UCF101_mini')
    p.add_argument('--split_dir', default='data/splits/ucfTrainTestlist')
    p.add_argument('--cache_dir', default='data/frame_cache')
    p.add_argument('--flow_root', default='data/optical_flow')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--frames', nargs='*', choices=['mini', 'base', 'all'])
    p.add_argument('--flow', action='store_true',
                   help='optical flow for the mini classes')
    p.add_argument('--num_base_classes', type=int, default=30)
    p.add_argument('--workers', type=int, default=os.cpu_count())
    args = p.parse_args()

    if args.verify:
        verify(args.video_root, args.split_dir)
        if os.path.isdir(args.mini_root):
            verify(args.mini_root, args.split_dir)
    for group in args.frames or []:
        if group == 'mini':
            root, classes = args.mini_root, MINI_CLASSES
        elif group == 'base':
            root = args.video_root
            classes = base_classes(root, args.num_base_classes)
        else:
            root, classes = args.video_root, None
        cache_dataset_frames(root, args.cache_dir, classes,
                             workers=args.workers)
    if args.flow:
        preprocess_all_flows(args.mini_root, args.flow_root,
                             classes=MINI_CLASSES, workers=args.workers)


if __name__ == '__main__':
    main()
