"""Dense optical flow (Farneback) pre-computation and colourisation (Part 1).

Run once before Part 4:
    python -m utils.optical_flow --video_root data/UCF101_mini \
        --flow_root data/optical_flow --workers 8
"""
import argparse
import os
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm

FARNEBACK_PARAMS = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                        poly_n=5, poly_sigma=1.2, flags=0)


def farneback(gray1, gray2):
    """Dense flow between two grayscale frames -> (H, W, 2) = (u, v)."""
    return cv2.calcOpticalFlowFarneback(gray1, gray2, None, **FARNEBACK_PARAMS)


def compute_flow_for_video(video_path, out_path, num_frames=16, size=112):
    """Compute Farneback flow at ``num_frames`` uniformly spaced positions.

    For every sampled index ``i`` the flow between frame ``i`` and ``i + 1``
    is computed, giving a stack saved as float32 (2, T, H, W).
    """
    cap = cv2.VideoCapture(video_path)
    grays = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        grays.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                (size, size), interpolation=cv2.INTER_AREA))
    cap.release()

    flows = []
    if len(grays) >= 2:
        for idx in np.linspace(0, len(grays) - 2, num_frames).astype(int):
            flows.append(farneback(grays[idx], grays[idx + 1]))
    else:
        flows = [np.zeros((size, size, 2), np.float32)] * num_frames
    flow_stack = np.stack(flows).astype(np.float32)  # (T, H, W, 2)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, flow_stack.transpose(3, 0, 1, 2))  # (2, T, H, W)


def _flow_job(job):
    cv2.setNumThreads(1)  # one process per core is faster than cv2 threads
    vpath, opath, num_frames, size = job
    if not os.path.exists(opath):
        compute_flow_for_video(vpath, opath, num_frames, size)


def preprocess_all_flows(ucf_root, flow_root, num_frames=16, size=112,
                         classes=None, workers=8):
    """Batch-process every video under ``ucf_root``. Run once before training.

    Already-computed files are skipped, so the job can be resumed.
    """
    classes = classes or sorted(os.listdir(ucf_root))
    jobs = []
    for cls in classes:
        cls_dir = os.path.join(ucf_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for vid in sorted(os.listdir(cls_dir)):
            if vid.endswith('.avi'):
                jobs.append((os.path.join(cls_dir, vid),
                             os.path.join(flow_root, cls,
                                          vid.replace('.avi', '.npy')),
                             num_frames, size))
    os.makedirs(flow_root, exist_ok=True)
    with Pool(workers) as pool:
        for _ in tqdm(pool.imap_unordered(_flow_job, jobs, chunksize=4),
                      total=len(jobs), desc='Computing optical flow'):
            pass
    print(f'Optical flow preprocessing complete! ({len(jobs)} videos)')


def flow_to_hsv(flow, max_mag=None):
    """Colour-code a flow field with the standard HSV scheme.

    Hue = direction (``cv2.cartToPolar`` angle), Value = magnitude,
    Saturation = 255. Accepts (H, W, 2) or (2, H, W). Returns RGB uint8.
    """
    if flow.shape[0] == 2 and flow.ndim == 3:
        flow = flow.transpose(1, 2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0].astype(np.float32),
                               flow[..., 1].astype(np.float32))
    hsv = np.zeros((*flow.shape[:2], 3), np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2          # OpenCV hue range is 0-179
    hsv[..., 1] = 255
    max_mag = max_mag if max_mag is not None else mag.max() + 1e-6
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video_root', default='data/UCF101_mini')
    parser.add_argument('--flow_root', default='data/optical_flow')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--size', type=int, default=112)
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    args = parser.parse_args()
    preprocess_all_flows(args.video_root, args.flow_root, args.num_frames,
                         args.size, workers=args.workers)


if __name__ == '__main__':
    main()
