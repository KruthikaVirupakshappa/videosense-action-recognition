"""Part 1 figures (Exercises 4-5 and Reflection Q2/Q3).

    python part1_visualize.py --video_root data/UCF101

Produces:
  figures/part1_sampled_frames.png   8 uniformly sampled frames
  figures/part1_flow_components.png  u, v and HSV flow for 4 consecutive pairs
  figures/part1_flow_comparison.png  walking vs running-style motion
  figures/part1_camera_motion.png    raw vs median-compensated flow
  figures/part1_tensor_check.txt     shape / dtype check of dataset tensors
"""
import argparse
import glob
import os

import cv2
import numpy as np

from utils.dataset import read_all_frames, uniform_indices
from utils.optical_flow import farneback
from utils import viz


def first_video(video_root, cls):
    vids = sorted(glob.glob(os.path.join(video_root, cls, '*.avi')))
    if not vids:
        raise FileNotFoundError(f'no videos for class {cls} in {video_root}')
    return vids[0]


def consecutive_flows(frames_bgr, start, n=4):
    """Full-resolution Farneback flow for frames start..start+n."""
    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]
    flows = [farneback(gray[i], gray[i + 1]) for i in range(start, start + n)]
    return rgb[start:start + n], flows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--video_root', default='data/UCF101')
    p.add_argument('--clip_class', default='Basketball')
    p.add_argument('--compare', nargs='+',
                   default=['WalkingWithDog', 'LongJump', 'HandstandWalking'],
                   help='classes for the walking-vs-running comparison '
                        '(UCF101 has no "Running" class; the LongJump run-up '
                        'is the closest sprint)')
    p.add_argument('--pan_class', default='Surfing',
                   help='class with strong camera motion')
    args = p.parse_args()

    # Exercise 4: 8 uniformly sampled frames
    path = first_video(args.video_root, args.clip_class)
    frames = read_all_frames(path)
    idx = uniform_indices(len(frames), 8)
    viz.plot_sampled_frames(
        [cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in idx], idx,
        f'{os.path.basename(path)}: 8 uniformly sampled frames of '
        f'{len(frames)}', 'figures/part1_sampled_frames.png')

    # Exercise 5: u / v / HSV for 4 consecutive frame pairs
    mid = len(frames) // 2
    rgb, flows = consecutive_flows(frames, mid)
    viz.plot_flow_components(
        rgb, flows, f'{os.path.basename(path)}: Farneback flow, frames '
        f'{mid}-{mid + 4}', 'figures/part1_flow_components.png')

    # Reflection Q3: compare motion statistics across classes
    clips = {}
    for cls in args.compare:
        try:
            fr = read_all_frames(first_video(args.video_root, cls))
        except FileNotFoundError as err:
            print(f'[skip] {err}')
            continue
        step = max(1, len(fr) // 8)
        gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in fr]
        fl = [farneback(gray[i], gray[i + 1])
              for i in range(0, len(fr) - 1, step)][:8]
        clips[cls] = (fr, fl)
    if clips:
        viz.plot_flow_comparison(clips, 'figures/part1_flow_comparison.png')

    # Reflection Q2: camera pan vs subject motion
    try:
        fr = read_all_frames(first_video(args.video_root, args.pan_class))
        i = len(fr) // 2
        g1, g2 = (cv2.cvtColor(fr[j], cv2.COLOR_BGR2GRAY) for j in (i, i + 1))
        viz.plot_camera_motion(cv2.cvtColor(fr[i], cv2.COLOR_BGR2RGB),
                               farneback(g1, g2),
                               'figures/part1_camera_motion.png')
    except FileNotFoundError as err:
        print(f'[skip] {err}')

    # Tensor sanity check: what the DataLoader will hand the network
    from utils.dataset import UCF101Dataset
    split = 'data/splits/ucfTrainTestlist/trainlist01.txt'
    if os.path.exists(split) and os.path.isdir('data/UCF101_mini'):
        ds = UCF101Dataset('data/UCF101_mini', split, train=True,
                           cache_dir='data/frame_cache')
        x, y = ds[0]
        flow_files = glob.glob('data/optical_flow/*/*.npy')
        msg = (f'RGB clip tensor: shape {tuple(x.shape)} (C, T, H, W), '
               f'dtype {x.dtype}, mean {x.mean():.3f}, std {x.std():.3f}, '
               f'label {y} ({ds.classes[y]})\n')
        if flow_files:
            f = np.load(flow_files[0])
            msg += (f'Flow stack: shape {f.shape} (2, T, H, W), dtype '
                    f'{f.dtype}, |flow| max {np.abs(f).max():.2f} px\n')
        print(msg)
        with open('figures/part1_tensor_check.txt', 'w') as fh:
            fh.write(msg)


if __name__ == '__main__':
    main()
