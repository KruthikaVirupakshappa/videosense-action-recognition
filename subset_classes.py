"""Create the 10-class mini-UCF101 (Section 4.2.3)."""
import os
import shutil

from utils.dataset import MINI_CLASSES as CLASSES

SRC = 'data/UCF101'
DST = 'data/UCF101_mini'

if __name__ == '__main__':
    for cls in CLASSES:
        shutil.copytree(f'{SRC}/{cls}', f'{DST}/{cls}', dirs_exist_ok=True)
    n = sum(len(os.listdir(f'{DST}/{c}')) for c in CLASSES)
    print(f'Mini dataset ready: {len(CLASSES)} classes, {n} videos')
