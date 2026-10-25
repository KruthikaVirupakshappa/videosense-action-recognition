"""Create the project directory structure from Section 3.4 of the handout."""
import os

DIRS = ['data/UCF101', 'data/optical_flow', 'data/splits', 'data/frame_cache',
        'models', 'utils', 'results', 'figures', 'figures/gradcam']

if __name__ == '__main__':
    for d in DIRS:
        os.makedirs(d, exist_ok=True)
        print(f'ok  {d}/')
