"""Plotting helpers for every figure in the report."""
import os

import cv2
import matplotlib
import numpy as np

matplotlib.use('Agg')  # headless-safe; notebooks display the saved PNGs
import matplotlib.pyplot as plt  # noqa: E402

from utils.optical_flow import flow_to_hsv  # noqa: E402


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {path}')


# ----------------------------------------------------------------- Part 1
def plot_sampled_frames(frames, indices, title, path):
    """Show uniformly sampled RGB frames side by side."""
    fig, axes = plt.subplots(1, len(frames), figsize=(2.2 * len(frames), 2.6))
    for ax, frame, idx in zip(axes, frames, indices):
        ax.imshow(frame)
        ax.set_title(f'frame {idx}', fontsize=9)
        ax.axis('off')
    fig.suptitle(title)
    _save(fig, path)


def plot_flow_components(frames, flows, title, path):
    """RGB frame | u | v | HSV colourisation for consecutive frame pairs."""
    n = len(flows)
    # 99th percentile instead of max: a few outlier pixels would otherwise
    # compress everything else into near-black / near-white
    lim = max(np.percentile(np.abs(f), 99) for f in flows) + 1e-6
    max_mag = max(np.percentile(np.linalg.norm(f, axis=-1), 99)
                  for f in flows) + 1e-6
    fig, axes = plt.subplots(4, n, figsize=(3.2 * n, 10))
    rows = ['RGB frame t', 'u (horizontal)', 'v (vertical)',
            'HSV colour\n(hue=dir, bright=speed)']
    for j, (frame, flow) in enumerate(zip(frames, flows)):
        axes[0, j].imshow(frame)
        im_u = axes[1, j].imshow(flow[..., 0], cmap='coolwarm',
                                 vmin=-lim, vmax=lim)
        axes[2, j].imshow(flow[..., 1], cmap='coolwarm', vmin=-lim, vmax=lim)
        axes[3, j].imshow(flow_to_hsv(flow, max_mag))
        axes[0, j].set_title(f't = {j} -> {j + 1}')
    for i, name in enumerate(rows):
        axes[i, 0].set_ylabel(name, fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im_u, ax=axes[1:3, :].ravel().tolist(), shrink=0.6,
                 label='pixels / frame')
    fig.suptitle(title)
    _save(fig, path)


def plot_flow_comparison(clips, path):
    """Side-by-side HSV flow + magnitude histograms for several classes.

    ``clips``: dict name -> (frames list, flows list)
    """
    names = list(clips)
    n_show = 4
    fig, axes = plt.subplots(len(names), n_show + 2,
                             figsize=(3 * (n_show + 2), 3 * len(names)))
    axes = np.atleast_2d(axes)
    all_mags = np.concatenate([np.linalg.norm(f, axis=-1).ravel()
                               for _, fl in clips.values() for f in fl])
    # shared scale so colours are comparable across classes; 99th percentile
    # so rare outliers (motion blur) do not turn every image black
    scale = np.percentile(all_mags, 99) + 1e-6
    hist_max = np.percentile(all_mags, 99.9)
    for i, name in enumerate(names):
        frames, flows = clips[name]
        axes[i, 0].imshow(cv2.cvtColor(frames[len(frames) // 2],
                                       cv2.COLOR_BGR2RGB))
        axes[i, 0].axis('off')
        sel = np.linspace(0, len(flows) - 1, n_show).astype(int)
        for j, k in enumerate(sel, start=1):
            axes[i, j].imshow(flow_to_hsv(flows[k], scale))
            axes[i, j].axis('off')
        mags = np.concatenate([np.linalg.norm(f, axis=-1).ravel()
                               for f in flows])
        axes[i, n_show + 1].hist(mags, bins=60, range=(0, hist_max),
                                 log=True)
        axes[i, n_show + 1].set_xlabel('|flow| (px / frame)', fontsize=8)
        axes[i, n_show + 1].set_title(
            f'mean |flow| = {mags.mean():.2f}, p95 = '
            f'{np.percentile(mags, 95):.2f}', fontsize=9)
        axes[i, 0].set_title(name, loc='left', fontsize=11)
    fig.suptitle('Optical flow comparison: frame, HSV flow at 4 times '
                 '(shared colour scale), magnitude histogram')
    fig.tight_layout()
    _save(fig, path)


def plot_camera_motion(frame, flow, path):
    """Raw flow vs. flow after subtracting the median (global) motion."""
    # letterbox bars (constant black rows) carry no motion: exclude them from
    # the global-motion estimate and blank them in the compensated view
    content = frame.reshape(-1, 3).max(1).reshape(frame.shape[:2]) > 16
    median = np.median(flow[content], axis=0)
    residual = (flow - median) * content[..., None]
    max_mag = np.percentile(np.linalg.norm(flow, axis=-1), 99) + 1e-6
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(frame)
    axes[0].set_title('frame', fontsize=10)
    axes[1].imshow(flow_to_hsv(flow, max_mag))
    axes[1].set_title(f'raw flow\nmedian (u, v) = ({median[0]:.2f}, '
                      f'{median[1]:.2f}) px', fontsize=10)
    axes[2].imshow(flow_to_hsv(residual, max_mag))
    axes[2].set_title('flow minus median\n(camera motion removed)',
                      fontsize=10)
    for ax in axes:
        ax.axis('off')
    fig.tight_layout()
    _save(fig, path)


# ---------------------------------------------------------- Training curves
def plot_training_curves(history, title, path):
    """Loss and accuracy curves from a history dict of per-epoch lists."""
    epochs = np.arange(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1.plot(epochs, history['train_loss'], label='train')
    ax1.plot(epochs, history['val_loss'], label='validation')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax2.plot(epochs, history['train_top1'], label='train top-1')
    ax2.plot(epochs, history['val_top1'], label='val top-1')
    ax2.plot(epochs, history['val_top5'], '--', label='val top-5')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    ax2.set_ylim(0, 1.02)
    ax2.set_title('Accuracy')
    ax2.legend()
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    _save(fig, path)


def plot_siamese_training(history, title, path):
    """Loss curve + mean positive / negative distance per epoch."""
    epochs = np.arange(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1.plot(epochs, history['train_loss'], label='train')
    ax1.plot(epochs, history['val_loss'], label='validation')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_title('Loss')
    ax1.legend()
    for key, lab in (('val_pos_mean', 'positive (same class)'),
                     ('val_neg_mean', 'negative (different class)')):
        ax2.plot(epochs, history[key], label=lab)
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('mean Euclidean distance')
    ax2.set_title('Validation pair distances')
    ax2.legend()
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    _save(fig, path)


def plot_distance_histograms_per_epoch(dist_log, title, path, max_panels=6):
    """Positive / negative distance histograms at selected epochs.

    ``dist_log``: list of dicts {'epoch', 'pos': [...], 'neg': [...]}
    """
    sel = np.unique(np.linspace(0, len(dist_log) - 1,
                                min(max_panels, len(dist_log))).astype(int))
    fig, axes = plt.subplots(1, len(sel), figsize=(3.3 * len(sel), 3.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, sel):
        entry = dist_log[i]
        ax.hist(entry['pos'], bins=30, range=(0, 2), alpha=0.6,
                label='positive', density=True)
        ax.hist(entry['neg'], bins=30, range=(0, 2), alpha=0.6,
                label='negative', density=True)
        ax.set_title(f"epoch {entry['epoch']}")
        ax.set_xlabel('distance')
    axes[0].legend()
    fig.suptitle(title)
    _save(fig, path)


def plot_margin_comparison(runs, path):
    """``runs``: dict label -> (pos distances, neg distances)."""
    fig, axes = plt.subplots(1, len(runs), figsize=(4 * len(runs), 3.4),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (label, (pos, neg)) in zip(axes, runs.items()):
        ax.hist(pos, bins=30, range=(0, 2), alpha=0.6, density=True,
                label='positive')
        ax.hist(neg, bins=30, range=(0, 2), alpha=0.6, density=True,
                label='negative')
        ax.set_title(label)
        ax.set_xlabel('distance (unit-sphere embeddings, max = 2)')
    axes[0].legend()
    fig.suptitle('Effect of the contrastive margin on pair distances')
    _save(fig, path)


# ------------------------------------------------------------------ Part 5
def plot_confusion_matrix(cm, class_names, path, top_n=20,
                          title='Normalised confusion matrix'):
    """Heat-map of the ``top_n`` classes with the highest error rate."""
    import seaborn as sns

    top_n = min(top_n, len(class_names))
    err = 1 - np.diag(cm)
    top_idx = np.sort(np.argsort(err)[-top_n:])
    cm_sub = cm[np.ix_(top_idx, top_idx)]
    names = [class_names[i] for i in top_idx]
    size = max(6, 0.55 * top_n + 3)
    fig, ax = plt.subplots(figsize=(size + 1, size))
    sns.heatmap(cm_sub, xticklabels=names, yticklabels=names, annot=True,
                fmt='.2f', cmap='Blues', vmin=0, vmax=1, ax=ax,
                cbar_kws={'label': 'fraction of true class'})
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    _save(fig, path)


def plot_confused_pair_evidence(pairs, path):
    """Visual evidence for confused class pairs.

    ``pairs``: list of dicts with keys 'a', 'b', 'rate_ab', 'rate_ba',
    'frames_a', 'frames_b' (4 uint8 frames each), 'flow_a', 'flow_b' (HSV
    images), 'motion_a', 'motion_b' (mean flow magnitude of the class).
    """
    fig, axes = plt.subplots(len(pairs), 1, figsize=(20, 2.6 * len(pairs)))
    axes = np.atleast_1d(axes)
    for ax, (r, p) in zip(axes, enumerate(pairs)):
        gap = np.full((p['frames_a'].shape[1], 12, 3), 255, np.uint8)
        tiles = (list(p['frames_a']) + [p['flow_a'], gap] +
                 list(p['frames_b']) + [p['flow_b']])
        ax.imshow(_strip(tiles))
        ax.axis('off')
        ax.set_title(
            f"Pair {r + 1}:  {p['a']} (mean |flow| {p['motion_a']:.2f})  vs  "
            f"{p['b']} (mean |flow| {p['motion_b']:.2f})   |   "
            f"{p['a']}->{p['b']} {p['rate_ab']:.0%},  "
            f"{p['b']}->{p['a']} {p['rate_ba']:.0%}   "
            '(4 frames + HSV flow per class)', fontsize=11)
    _save(fig, path)


def _strip(tiles, pad=3):
    """Concatenate images of equal height horizontally with white padding."""
    h = tiles[0].shape[0]
    out = []
    for t in tiles:
        if t.shape[0] != h:
            t = cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h))
        out += [t, np.full((h, pad, 3), 255, np.uint8)]
    return np.concatenate(out[:-1], axis=1)


def plot_tsne_panels(panels, class_names, path):
    """``panels``: list of (title, X_2d, labels, silhouette_hd, sil_2d)."""
    cmap = plt.get_cmap('tab20' if len(class_names) > 10 else 'tab10')
    fig, axes = plt.subplots(1, len(panels), figsize=(6.3 * len(panels), 6))
    axes = np.atleast_1d(axes)
    for ax, (title, x2d, labels, sil, sil2d) in zip(axes, panels):
        labels = np.asarray(labels)
        for c, name in enumerate(class_names):
            m = labels == c
            ax.scatter(x2d[m, 0], x2d[m, 1], s=14, alpha=0.75,
                       color=cmap(c % cmap.N), label=name)
        ax.set_title(f'{title}\nsilhouette: embedding {sil:.3f} | '
                     f't-SNE 2D {sil2d:.3f}', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[-1].legend(fontsize=8, markerscale=1.5, loc='center left',
                    bbox_to_anchor=(1.0, 0.5))
    fig.suptitle('t-SNE of video embeddings (test split)')
    _save(fig, path)


def overlay_cam(frames, cam, alpha=0.45):
    """Blend a (T, H, W) CAM in [0, 1] onto uint8 frames (T, H, W, 3)."""
    out = []
    for frame, c in zip(frames, cam):
        heat = cv2.applyColorMap((c * 255).astype(np.uint8),
                                 cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        out.append((frame * (1 - alpha) + heat * alpha).astype(np.uint8))
    return np.stack(out)


def save_gif(frames, path, fps=6, scale=2):
    """Write uint8 frames (T, H, W, 3) to an animated GIF."""
    from PIL import Image

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    imgs = [Image.fromarray(f).resize((f.shape[1] * scale,
                                       f.shape[0] * scale)) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    print(f'saved {path}')


def plot_gradcam_grid(rows, path, n_frames=8):
    """``rows``: list of dicts with 'frames', 'overlay', 'title'."""
    fig, axes = plt.subplots(len(rows), 1, figsize=(16, 4.4 * len(rows)))
    axes = np.atleast_1d(axes)
    for ax, row in zip(axes, rows):
        idx = np.linspace(0, len(row['frames']) - 1, n_frames).astype(int)
        top = _strip([row['frames'][t] for t in idx])
        bottom = _strip([row['overlay'][t] for t in idx])
        sep = np.full((3, top.shape[1], 3), 255, np.uint8)
        ax.imshow(np.concatenate([top, sep, bottom]))
        ax.axis('off')
        ax.set_title(f"{row['title']}\nframes t = {', '.join(map(str, idx))}"
                     ' (top: input, bottom: Grad-CAM overlay)', fontsize=10)
    _save(fig, path)


def plot_per_class_comparison(class_names, acc_a, acc_b, label_a, label_b,
                              path, highlight=()):
    """Grouped bar chart of per-class accuracy for two models."""
    x = np.arange(len(class_names))
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(class_names)), 4.5))
    ax.bar(x - w / 2, acc_a, w, label=label_a)
    ax.bar(x + w / 2, acc_b, w, label=label_b)
    for i in highlight:
        ax.annotate(f'+{acc_b[i] - acc_a[i]:.0%}', (x[i] + w / 2, acc_b[i]),
                    ha='center', va='bottom', fontsize=8, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('per-class accuracy')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(f'Per-class accuracy: {label_a} vs {label_b}')
    _save(fig, path)
