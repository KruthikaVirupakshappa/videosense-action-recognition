"""Draw the architecture diagrams used in the report's Methods section.

    python make_diagrams.py   # -> figures/arch_c3d.png, arch_siamese.png,
                              #    arch_twostream.png
"""
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

COLORS = {'in': '#dbe9f6', 'conv': '#9ecae1', 'pool': '#fdd0a2',
          'fc': '#c7e9c0', 'loss': '#fcbba1', 'op': '#e5e5e5'}


def box(ax, x, y, w, h, text, kind, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02',
                                fc=COLORS[kind], ec='#444444', lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', (x2, y2), (x1, y1),
                arrowprops=dict(arrowstyle='->', lw=0.9, color='#333333'))


def chain(ax, items, y, x0=0.0, w=1.0, gap=0.25, h=0.8):
    """Draw boxes left-to-right connected by arrows; returns right edge x."""
    x = x0
    for i, (text, kind) in enumerate(items):
        box(ax, x, y, w, h, text, kind)
        if i:
            arrow(ax, x - gap, y + h / 2, x, y + h / 2)
        x += w + gap
    return x - gap


def save(fig, path):
    os.makedirs('figures', exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {path}')


C3D_ITEMS = [('clip\n3x16x112x112', 'in'),
             ('conv1a 64\n3x3x3', 'conv'),
             ('pool1 1x2x2\n64x16x56x56', 'pool'),
             ('conv2a 128', 'conv'), ('pool2 2x2x2\n128x8x28x28', 'pool'),
             ('conv3a,b\n256', 'conv'), ('pool3\n256x4x14x14', 'pool'),
             ('conv4a,b\n512', 'conv'), ('pool4\n512x2x7x7', 'pool'),
             ('conv5a,b\n512', 'conv'), ('pool5\n512x1x3x3', 'pool')]


def c3d():
    fig, ax = plt.subplots(figsize=(17, 2.2))
    right = chain(ax, C3D_ITEMS + [('fc6 4096\n(embedding)', 'fc'),
                                   ('fc7 4096', 'fc'),
                                   ('fc8\nnum_classes', 'fc')], 0)
    ax.text(0, 1.05, 'C3D: every conv = Conv3d 3x3x3 (pad 1) + BatchNorm3d '
            '+ ReLU; dropout 0.5 after fc6 / fc7', fontsize=9)
    ax.set_xlim(-0.2, right + 0.2)
    ax.set_ylim(-0.2, 1.3)
    ax.axis('off')
    save(fig, 'figures/arch_c3d.png')


def siamese():
    fig, ax = plt.subplots(figsize=(12, 3.6))
    for y, name in ((2.0, 'clip x1'), (0.0, 'clip x2')):
        box(ax, 0, y, 1.2, 0.8, name, 'in')
        arrow(ax, 1.2, y + 0.4, 2.0, 1.4)
    box(ax, 2.0, 0.9, 2.6, 1.0, 'SHARED C3D encoder\n(conv1a ... pool5)\n'
        'one set of weights', 'conv', fs=9)
    box(ax, 5.0, 0.9, 2.2, 1.0, 'projector\n4608 -> 1024 -> 512\n'
        '+ L2 normalise', 'fc', fs=9)
    arrow(ax, 4.6, 1.4, 5.0, 1.4)
    for y, name in ((2.0, 'z1 (unit sphere)'), (0.0, 'z2 (unit sphere)')):
        box(ax, 7.7, y, 1.6, 0.8, name, 'op')
        arrow(ax, 7.2, 1.4, 7.7, y + 0.4)
        arrow(ax, 9.3, y + 0.4, 9.8, 1.4)
    box(ax, 9.8, 0.9, 1.6, 1.0, 'd = ||z1 - z2||', 'op', fs=9)
    box(ax, 11.8, 0.9, 2.4, 1.0, 'contrastive loss\n(1-y) d^2/2 +\n'
        'y max(0, m-d)^2/2', 'loss', fs=9)
    arrow(ax, 11.4, 1.4, 11.8, 1.4)
    ax.set_xlim(-0.2, 14.4)
    ax.set_ylim(-0.3, 3.0)
    ax.axis('off')
    ax.set_title('Siamese network (Part 3)')
    save(fig, 'figures/arch_siamese.png')


def twostream():
    fig, ax = plt.subplots(figsize=(13, 3.8))
    for y, inp, enc in ((2.2, 'RGB clip\n3x16x112x112', 'RGB C3D encoder'),
                        (0.0, 'flow clip (u, v)\n2x16x112x112',
                         'Flow C3D encoder\n(conv1a: 2 in-channels)')):
        box(ax, 0, y, 1.8, 0.9, inp, 'in')
        box(ax, 2.3, y, 2.4, 0.9, enc, 'conv')
        box(ax, 5.2, y, 1.9, 0.9, 'fc 4608 -> 512\n+ ReLU', 'fc')
        arrow(ax, 1.8, y + 0.45, 2.3, y + 0.45)
        arrow(ax, 4.7, y + 0.45, 5.2, y + 0.45)
        arrow(ax, 7.1, y + 0.45, 7.7, 1.55)
    box(ax, 7.7, 1.0, 2.8, 1.1, 'fusion\nconcat [z_rgb; z_flow]\n'
        'or a z_rgb + (1-a) z_flow\nor average of logits', 'op', fs=8)
    box(ax, 11.0, 1.1, 2.0, 0.9, 'dropout + linear\nnum_classes', 'fc')
    arrow(ax, 10.5, 1.55, 11.0, 1.55)
    ax.set_xlim(-0.2, 13.2)
    ax.set_ylim(-0.3, 3.3)
    ax.axis('off')
    ax.set_title('Two-stream late-fusion network (Part 4)')
    save(fig, 'figures/arch_twostream.png')


if __name__ == '__main__':
    c3d()
    siamese()
    twostream()
