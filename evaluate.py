"""Part 5 -- evaluation, visualisation and interpretation.

Runs any subset of these tasks on the mini test split (testlist01):

  report      top-1/top-5, per-class precision/recall, macro-F1 (Ex. 28)
  confusion   normalised confusion matrices + 3 most-confused pairs (Ex. 29)
  tsne        t-SNE of C3D / Siamese / two-stream embeddings +
              silhouette scores (Ex. 30)
  gradcam     Grad-CAM GIFs + frame overlays for 5 clips (Ex. 31)
  tta         test-time augmentation with 5 temporal crops (Ex. 32, bonus)
  complexity  torchinfo parameter / FLOP table for every model (Ex. 10)
  per_class   per-class accuracy, fusion vs RGB-only (Ex. 24)
  margins     pos/neg distance histograms across Siamese margins (Ex. 16)
  tables      collect every results/*.json into results/results_tables.md

Example:
    python evaluate.py --tasks report confusion tsne gradcam tta \
        complexity per_class tables
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, silhouette_score)
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.cnn3d import C3D, build_r2plus1d_18, r2plus1d_embedding
from models.fusion import TwoStreamFusion
from models.siamese import VideoSiamese
from utils.dataset import (IMAGENET_MEAN, IMAGENET_STD, KINETICS_MEAN,
                           KINETICS_STD, MultimodalDataset, UCF101Dataset,
                           segment_indices, VideoTransform)
from utils.optical_flow import farneback, flow_to_hsv
from utils.train_utils import (autocast, get_device, load_json,
                               model_complexity, save_json, set_seed,
                               top_k_accuracy)
from utils import viz

TASKS = ('report', 'confusion', 'tsne', 'gradcam', 'tta', 'complexity',
         'per_class', 'margins', 'tables')
DISPLAY = {'c3d': 'C3D (scratch)', 'r2plus1d': 'R(2+1)D-18 (pretrained)',
           'fusion': 'Two-stream (concat)', 'siamese': 'Siamese (C3D)'}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--tasks', nargs='+', default=list(TASKS), choices=TASKS)
    p.add_argument('--data_root', default='data/UCF101_mini')
    p.add_argument('--split_dir', default='data/splits/ucfTrainTestlist')
    p.add_argument('--cache_dir', default='data/frame_cache')
    p.add_argument('--flow_root', default='data/optical_flow')
    p.add_argument('--c3d_ckpt', default='models/c3d_ucf101.pt')
    p.add_argument('--r2plus1d_ckpt', default='models/r2plus1d.pt')
    p.add_argument('--fusion_ckpt', default='models/twostream_ucf101.pt')
    p.add_argument('--siamese_ckpt', default='models/siamese_ucf101.pt')
    p.add_argument('--cm_model', default='c3d',
                   choices=['c3d', 'r2plus1d', 'fusion'],
                   help='model whose confused pairs are analysed')
    p.add_argument('--gradcam_models', nargs='+', default=['c3d', 'r2plus1d'])
    p.add_argument('--n_gradcam', type=int, default=5)
    p.add_argument('--tta_crops', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


# ------------------------------------------------------------------ loading
class Ctx:
    """Lazily builds datasets and models so each task loads only what it
    needs (and missing checkpoints just skip that model)."""

    def __init__(self, args):
        self.args = args
        self.device = get_device()
        self._models, self._ds = {}, {}

    def dataset(self, kind):
        if kind not in self._ds:
            a = self.args
            mean, std = ((KINETICS_MEAN, KINETICS_STD) if kind == 'r2plus1d'
                         else (IMAGENET_MEAN, IMAGENET_STD))
            ds = UCF101Dataset(a.data_root,
                               os.path.join(a.split_dir, 'testlist01.txt'),
                               train=False, cache_dir=a.cache_dir or None,
                               mean=mean, std=std)
            if kind == 'fusion':
                ds = MultimodalDataset(ds, a.flow_root)
            self._ds[kind] = ds
        return self._ds[kind]

    @property
    def classes(self):
        return self.dataset('c3d').classes

    def loader(self, kind):
        return DataLoader(self.dataset(kind), batch_size=self.args.batch_size,
                          shuffle=False, num_workers=self.args.num_workers,
                          pin_memory=True)

    def model(self, kind):
        if kind in self._models:
            return self._models[kind]
        ckpt = getattr(self.args, f'{kind}_ckpt')
        if not os.path.exists(ckpt):
            print(f'[skip] {kind}: checkpoint {ckpt} not found')
            self._models[kind] = None
            return None
        n = len(self.classes)
        state = torch.load(ckpt, map_location='cpu')
        if kind == 'c3d':
            model = C3D(num_classes=n,
                        batch_norm=any('running_mean' in k for k in state))
        elif kind == 'r2plus1d':
            model = build_r2plus1d_18(n, pretrained=False,
                                      freeze_backbone=False)
        elif kind == 'fusion':
            model = TwoStreamFusion(num_classes=n, fusion='concat')
        else:
            model = VideoSiamese()
        model.load_state_dict(state)
        model = model.to(self.device).eval()
        self._models[kind] = model
        return model


def forward(kind, model, batch, device):
    if kind == 'fusion':
        rgb, flow, labels = batch
        return model(rgb.to(device), flow.to(device)), labels
    clips, labels = batch
    return model(clips.to(device)), labels


def embed(kind, model, batch, device):
    if kind == 'fusion':
        rgb, flow, labels = batch
        return model.get_embedding(rgb.to(device), flow.to(device)), labels
    clips, labels = batch
    clips = clips.to(device)
    if kind == 'r2plus1d':
        return r2plus1d_embedding(model, clips), labels
    return model.get_embedding(clips), labels


@torch.no_grad()
def predict(ctx, kind):
    model = ctx.model(kind)
    logits, labels = [], []
    for batch in tqdm(ctx.loader(kind), desc=f'predict {kind}', leave=False):
        with autocast(ctx.device):
            out, lbl = forward(kind, model, batch, ctx.device)
        logits.append(out.float().cpu())
        labels.append(lbl)
    return torch.cat(logits), torch.cat(labels)


def available(ctx, kinds=('c3d', 'r2plus1d', 'fusion')):
    return [k for k in kinds if ctx.model(k) is not None]


# ------------------------------------------------------------------- tasks
def task_report(ctx):
    """Exercise 28: full evaluation report for every trained classifier."""
    summary = {}
    for kind in available(ctx):
        logits, labels = predict(ctx, kind)
        preds = logits.argmax(1).numpy()
        y = labels.numpy()
        top1 = float((preds == y).mean())
        top5 = top_k_accuracy(logits, labels, k=5)
        rep = classification_report(y, preds, target_names=ctx.classes,
                                    digits=3, zero_division=0)
        macro_f1 = float(f1_score(y, preds, average='macro'))
        print(f'\n=== {DISPLAY[kind]} ===\nTop-1 Accuracy: {top1:.4f} | '
              f'Top-5 Accuracy: {top5:.4f} | macro-F1: {macro_f1:.4f}\n{rep}')
        with open(f'results/eval_report_{kind}.txt', 'w') as f:
            f.write(f'{DISPLAY[kind]}\nTop-1 {top1:.4f}  Top-5 {top5:.4f}  '
                    f'macro-F1 {macro_f1:.4f}\n\n{rep}')
        summary[kind] = {
            'top1': top1, 'top5': top5, 'macro_f1': macro_f1,
            'report': classification_report(y, preds, output_dict=True,
                                            target_names=ctx.classes,
                                            zero_division=0)}
    save_json(summary, 'results/eval_summary.json')


def _class_motion(ctx):
    """Mean optical-flow magnitude per class (test split)."""
    ds = ctx.dataset('c3d')
    mags = {c: [] for c in range(len(ctx.classes))}
    for path, lbl in ds.samples:
        fpath = os.path.join(ctx.args.flow_root,
                             os.path.basename(os.path.dirname(path)),
                             os.path.basename(path).replace('.avi', '.npy'))
        if os.path.exists(fpath):
            flow = np.load(fpath)
            mags[lbl].append(float(np.sqrt((flow ** 2).sum(0)).mean()))
    return {c: (float(np.mean(v)) if v else float('nan'))
            for c, v in mags.items()}


def _flow_tile(ctx, sample_path):
    fpath = os.path.join(ctx.args.flow_root,
                         os.path.basename(os.path.dirname(sample_path)),
                         os.path.basename(sample_path).replace('.avi', '.npy'))
    if not os.path.exists(fpath):
        return np.zeros((112, 112, 3), np.uint8)
    flow = np.load(fpath)                       # (2, T, H, W)
    return flow_to_hsv(flow[:, flow.shape[1] // 2])


def task_confusion(ctx):
    """Exercise 29: normalised confusion matrix + 3 confused pairs."""
    names = ctx.classes
    motion = _class_motion(ctx)
    for kind in available(ctx):
        logits, labels = predict(ctx, kind)
        preds, y = logits.argmax(1).numpy(), labels.numpy()
        cm = confusion_matrix(y, preds, labels=range(len(names)),
                              normalize='true')
        np.save(f'results/confusion_{kind}.npy', cm)
        viz.plot_confusion_matrix(
            cm, names, f'figures/confusion_matrix_{kind}.png',
            title=f'{DISPLAY[kind]}: normalised confusion matrix '
                  f'(top-{min(20, len(names))} most-confused classes)')
        if kind != ctx.args.cm_model:
            continue
        # symmetric confusion score for every unordered pair
        scores = [((cm[i, j] + cm[j, i]), i, j)
                  for i in range(len(names)) for j in range(i + 1, len(names))]
        scores.sort(reverse=True)
        ds = ctx.dataset('c3d')
        pairs = []
        for _, i, j in scores[:3]:
            # prefer a clip of class i that was actually predicted as j
            mis = np.flatnonzero((y == i) & (preds == j))
            ia = int(mis[0]) if len(mis) else int(np.flatnonzero(y == i)[0])
            mis_b = np.flatnonzero((y == j) & (preds == i))
            ib = (int(mis_b[0]) if len(mis_b)
                  else int(np.flatnonzero(y == j)[0]))
            fa, fb = ds.raw_frames(ia), ds.raw_frames(ib)
            sel = np.linspace(0, len(fa) - 1, 4).astype(int)
            pairs.append({
                'a': names[i], 'b': names[j],
                'rate_ab': float(cm[i, j]), 'rate_ba': float(cm[j, i]),
                'frames_a': fa[sel], 'frames_b': fb[sel],
                'flow_a': _flow_tile(ctx, ds.samples[ia][0]),
                'flow_b': _flow_tile(ctx, ds.samples[ib][0]),
                'motion_a': motion[i], 'motion_b': motion[j],
                'example_a': os.path.basename(ds.samples[ia][0]),
                'example_b': os.path.basename(ds.samples[ib][0])})
        viz.plot_confused_pair_evidence(pairs,
                                        f'figures/confused_pairs_{kind}.png')
        save_json([{k: v for k, v in p.items()
                    if not k.startswith(('frames', 'flow'))} for p in pairs],
                  f'results/confused_pairs_{kind}.json')
        for p in pairs:
            print(f"{p['a']} <-> {p['b']}: {p['rate_ab']:.2f} / "
                  f"{p['rate_ba']:.2f} | mean |flow| {p['motion_a']:.2f} vs "
                  f"{p['motion_b']:.2f}")


@torch.no_grad()
def _embeddings(ctx, kind):
    model = ctx.model(kind)
    loader_kind = 'c3d' if kind == 'siamese' else kind
    embs, labels = [], []
    for batch in tqdm(ctx.loader(loader_kind), desc=f'embed {kind}',
                      leave=False):
        with autocast(ctx.device):
            z, lbl = embed(kind, model, batch, ctx.device)
        embs.append(z.float().cpu())
        labels.append(lbl)
    return torch.cat(embs).numpy(), torch.cat(labels).numpy()


def task_tsne(ctx):
    """Exercise 30: t-SNE for C3D, Siamese and two-stream + silhouette."""
    panels, scores = [], {}
    for kind in ('c3d', 'siamese', 'fusion', 'r2plus1d'):
        if ctx.model(kind) is None:
            continue
        x, y = _embeddings(ctx, kind)
        x_2d = TSNE(n_components=2, perplexity=30, init='pca',
                    random_state=42).fit_transform(x)
        sil = float(silhouette_score(x, y))                  # Euclidean, HD
        sil_cos = float(silhouette_score(x, y, metric='cosine'))
        sil_2d = float(silhouette_score(x_2d, y))
        scores[kind] = {'silhouette_euclidean': sil,
                        'silhouette_cosine': sil_cos,
                        'silhouette_tsne2d': sil_2d, 'dim': int(x.shape[1])}
        title = DISPLAY[kind] + (' - classes UNSEEN in training'
                                 if kind == 'siamese' else '')
        panels.append((title, x_2d, y, sil, sil_2d))
        viz.plot_tsne_panels([panels[-1]], ctx.classes,
                             f'figures/tsne_{kind}.png')
        print(f'{kind}: silhouette {sil:.3f} (cosine {sil_cos:.3f}, '
              f't-SNE 2D {sil_2d:.3f})')
    # the three models required by Exercise 30 side by side
    core = [p for p, k in zip(panels, scores) if k != 'r2plus1d']
    viz.plot_tsne_panels(core, ctx.classes, 'figures/tsne_comparison.png')
    save_json(scores, 'results/silhouette_scores.json')


class VideoGradCAM:
    """Grad-CAM for 3D CNNs (Selvaraju et al. 2017, extended to time).

    Channel weights are the gradients averaged over (T, H, W); the CAM is
    ReLU(sum_c w_c * A_c), upsampled trilinearly to the clip size.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = self.gradients = None
        self.handle = target_layer.register_forward_hook(
            self._save_activation)

    def remove(self):
        """Detach the hook so later no-grad inference is unaffected."""
        self.handle.remove()

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()
        # tensor hook is more robust than module backward hooks
        if output.requires_grad:
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def generate_cam(self, clip, class_idx=None, per_frame_norm=False):
        self.model.zero_grad()
        with torch.enable_grad():
            logits = self.model(clip)
            if class_idx is None:
                class_idx = int(logits.argmax(1))
            logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=clip.shape[2:], mode='trilinear',
                            align_corners=False)[0, 0]       # (T, H, W)
        if per_frame_norm:   # handout variant: hides temporal differences
            for t in range(cam.shape[0]):
                lo, hi = cam[t].min(), cam[t].max()
                cam[t] = (cam[t] - lo) / (hi - lo + 1e-8)
        else:                # global normalisation keeps temporal saliency
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        conf = logits.detach().softmax(1)[0, class_idx].item()
        return cam.cpu().numpy(), class_idx, conf


def _motion_magnitude(frames):
    """Farneback magnitude between consecutive sampled frames (T, H, W)."""
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    mags = [np.linalg.norm(farneback(g1, g2), axis=-1)
            for g1, g2 in zip(grays[:-1], grays[1:])]
    return np.stack(mags + [mags[-1]])


def task_gradcam(ctx):
    """Exercise 31: Grad-CAM on 5 clips from 5 different classes."""
    ds = ctx.dataset('c3d')
    rng = random.Random(ctx.args.seed)
    picks = []
    for c in rng.sample(sorted(ds.class_idx), min(ctx.args.n_gradcam,
                                                  len(ds.class_idx))):
        picks.append(rng.choice(ds.class_idx[c]))
    stats = {}
    for kind in ctx.args.gradcam_models:
        model = ctx.model(kind)
        if model is None:
            continue
        model.float().eval()
        layer = model.gradcam_layer() if kind == 'c3d' else model.layer4
        cam_engine = VideoGradCAM(model, layer)
        kds = ctx.dataset(kind)
        rows, stats[kind] = [], []
        for idx in picks:
            clip, label = kds[idx]
            frames = ds.raw_frames(idx)
            cam, pred, conf = cam_engine.generate_cam(
                clip.unsqueeze(0).to(ctx.device))
            overlay = viz.overlay_cam(frames, cam)
            name = os.path.splitext(os.path.basename(ds.samples[idx][0]))[0]
            viz.save_gif(overlay, f'figures/gradcam/{kind}_{name}.gif')
            # Quantitative proxies for "content-relevant" attention:
            motion = _motion_magnitude(frames)
            hot = motion >= np.quantile(motion, 0.8)        # top-20% motion
            h, w = cam.shape[1:]
            centre = np.zeros_like(cam, bool)
            centre[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = True
            mass = cam.sum() + 1e-8
            entry = {
                'video': name, 'true': ctx.classes[label],
                'pred': ctx.classes[pred], 'confidence': float(conf),
                'corr_cam_motion': float(np.corrcoef(cam.ravel(),
                                                     motion.ravel())[0, 1]),
                'cam_mass_in_top20pct_motion': float(cam[hot].sum() / mass),
                'cam_mass_in_centre': float(cam[centre].sum() / mass),
                'temporal_profile': cam.mean((1, 2)).round(3).tolist()}
            stats[kind].append(entry)
            ok = 'OK' if entry['true'] == entry['pred'] else 'WRONG'
            rows.append({'frames': frames, 'overlay': overlay,
                         'title': f"{kind}: true {entry['true']} | pred "
                                  f"{entry['pred']} ({conf:.2f}, {ok}) | "
                                  f"CAM-motion r="
                                  f"{entry['corr_cam_motion']:.2f}"})
            print(f"[{kind}] {name}: pred {entry['pred']} | r(cam,motion)="
                  f"{entry['corr_cam_motion']:.2f} | mass@motion "
                  f"{entry['cam_mass_in_top20pct_motion']:.2f} (chance 0.20)"
                  f" | mass@centre {entry['cam_mass_in_centre']:.2f} "
                  f"(chance 0.25)")
        cam_engine.remove()
        viz.plot_gradcam_grid(rows, f'figures/gradcam_{kind}.png')
    save_json(stats, 'results/gradcam_stats.json')


@torch.no_grad()
def task_tta(ctx):
    """Exercise 32 (bonus): average logits over random temporal crops."""
    out = {}
    for kind in ('c3d', 'r2plus1d'):
        model = ctx.model(kind)
        if model is None:
            continue
        ds = ctx.dataset(kind)
        tf = VideoTransform(ds.spatial_size, train=False,
                            mean=ds.transform.mean, std=ds.transform.std)
        random.seed(ctx.args.seed)
        single, tta, labels = [], [], []
        for path, label in tqdm(ds.samples, desc=f'TTA {kind}', leave=False):
            stack = np.load(os.path.join(
                ds.cache_dir, os.path.basename(os.path.dirname(path)),
                os.path.basename(path).replace('.avi', '.npy')),
                mmap_mode='r')
            views = [np.asarray(stack[np.linspace(0, len(stack) - 1,
                                                  ds.num_frames).astype(int)])]
            views += [np.asarray(stack[segment_indices(len(stack),
                                                       ds.num_frames)])
                      for _ in range(ctx.args.tta_crops)]
            batch = torch.stack([tf(v) for v in views]).to(ctx.device)
            with autocast(ctx.device):
                logits = model(batch).float().cpu()
            single.append(logits[0])
            tta.append(logits[1:].softmax(1).mean(0))
            labels.append(label)
        labels = torch.tensor(labels)
        single, tta = torch.stack(single), torch.stack(tta)
        out[kind] = {'single_top1': top_k_accuracy(single, labels, 1),
                     'tta_top1': top_k_accuracy(tta, labels, 1),
                     'single_top5': top_k_accuracy(single, labels, 5),
                     'tta_top5': top_k_accuracy(tta, labels, 5),
                     'crops': ctx.args.tta_crops}
        print(f"{kind}: single-clip {out[kind]['single_top1']:.4f} -> "
              f"TTA({ctx.args.tta_crops}) {out[kind]['tta_top1']:.4f}")
    save_json(out, 'results/tta.json')


def task_complexity(ctx):
    """Exercise 10: parameter and FLOP counts via torchinfo."""
    dev = ctx.device
    rgb = torch.zeros(1, 3, 16, 112, 112, device=dev)
    flow = torch.zeros(1, 2, 16, 112, 112, device=dev)
    n = len(ctx.classes)
    builders = {
        'c3d': (lambda: C3D(num_classes=n), (rgb,)),
        'r2plus1d': (lambda: build_r2plus1d_18(n, pretrained=False,
                                               freeze_backbone=True), (rgb,)),
        'siamese': (lambda: VideoSiamese(), (rgb, rgb)),
        'fusion': (lambda: TwoStreamFusion(n, fusion='concat'), (rgb, flow)),
    }
    out = {}
    for kind, (build, inputs) in builders.items():
        out[kind] = model_complexity(build().to(dev), inputs,
                                     path=f'results/{kind}_torchinfo.txt')
        print(f"{kind:9s} params {out[kind]['params']:>12,}  trainable "
              f"{out[kind]['trainable_params']:>12,}  "
              f"GFLOPs {out[kind]['flops_G']:7.1f}")
    save_json(out, 'results/complexity.json')


def task_per_class(ctx):
    """Exercise 24: where does motion help the most?"""
    rgb_path = 'results/fusion_rgb.json'
    fus_path = 'results/fusion_concat.json'
    if not (os.path.exists(rgb_path) and os.path.exists(fus_path)):
        print('[skip] per_class: run train_fusion.py --fusion rgb / concat')
        return
    rgb, fus = load_json(rgb_path), load_json(fus_path)
    names = fus['classes']
    a, b = np.array(rgb['per_class_acc']), np.array(fus['per_class_acc'])
    gain = b - a
    top5 = np.argsort(-gain)[:5]
    viz.plot_per_class_comparison(names, a, b, 'RGB-only', 'RGB+Flow (concat)',
                                  'figures/per_class_fusion_vs_rgb.png',
                                  highlight=top5)
    res = [{'class': names[i], 'rgb_acc': float(a[i]),
            'fusion_acc': float(b[i]), 'gain': float(gain[i])} for i in top5]
    save_json(res, 'results/per_class_gain.json')
    for r in res:
        print(f"{r['class']:18s} {r['rgb_acc']:.2f} -> {r['fusion_acc']:.2f} "
              f"({r['gain']:+.2f})")


def task_margins(ctx):
    """Exercise 16: final-epoch distance histograms for each Siamese run."""
    runs = {}
    for path in sorted(glob.glob('results/siamese*.json')):
        r = load_json(path)
        if 'dist_log' not in r:
            continue
        m = r['margin'] if r['loss'] == 'contrastive' else r['triplet_margin']
        last = r['dist_log'][-1]
        runs[f"{r['loss']} m={m}"] = (last['pos'], last['neg'])
    if runs:
        viz.plot_margin_comparison(runs, 'figures/siamese_margin_hist.png')
    else:
        print('[skip] margins: no Siamese results found')


def _fmt(x, pct=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '-'
    return f'{100 * x:.1f}%' if pct else f'{x:.3f}'


def task_tables(ctx):
    """Collect every results JSON into one Markdown file for the report."""
    lines = ['# VideoSense results tables', '']
    runs = {os.path.basename(p)[:-5]: load_json(p)
            for p in glob.glob('results/*.json')}
    runs = {k: v for k, v in runs.items() if isinstance(v, dict)}

    lines += ['## Classification (mini UCF101, testlist01)', '',
              '| Model | Top-1 | Top-5 | Best-epoch top-1 | Train top-1 |'
              ' Params | GFLOPs |', '|---|---|---|---|---|---|---|']
    fusion_runs = sorted(k for k in runs if k.startswith('fusion_'))
    for key in ['c3d', 'r2plus1d'] + fusion_runs:
        r = runs.get(key)
        if not r or 'final_top1' not in r:
            continue
        c = r.get('complexity', {})
        lines.append(f"| {key} | {_fmt(r['final_top1'])} | "
                     f"{_fmt(r['final_top5'])} | {_fmt(r.get('best_top1'))} | "
                     f"{_fmt(r.get('train_top1_final'))} | "
                     f"{c.get('params', 0):,} | {c.get('flops_G', 0):.1f} |")

    ls = sorted((r['label_smoothing'], k) for k, r in runs.items()
                if r.get('model') == 'c3d' and 'label_smoothing' in r)
    if ls:
        lines += ['', '## Label smoothing (C3D)', '',
                  '| label_smoothing | run | Top-1 | Top-5 | Train top-1 |'
                  ' train-val gap |', '|---|---|---|---|---|---|']
        for s, k in ls:
            r = runs[k]
            gap = r['train_top1_final'] - r['final_top1']
            lines.append(f"| {s} | {k} | {_fmt(r['final_top1'])} | "
                         f"{_fmt(r['final_top5'])} | "
                         f"{_fmt(r['train_top1_final'])} | {_fmt(gap)} |")

    abl = [k for k in ('fusion_rgb', 'fusion_flow', 'fusion_concat',
                       'fusion_average', 'fusion_attention',
                       'fusion_cross_attention') if k in runs]
    if abl:
        lines += ['', '## Two-stream ablation', '',
                  '| Variant | Top-1 | Top-5 | learned alpha (RGB) |',
                  '|---|---|---|---|']
        for k in abl:
            r = runs[k]
            alpha = r.get('alpha_final')
            lines.append(f"| {k.replace('fusion_', '')} | "
                         f"{_fmt(r['final_top1'])} | {_fmt(r['final_top5'])} |"
                         f" {'-' if alpha is None else f'{alpha:.3f}'} |")

    sia = sorted(k for k, r in runs.items() if 'fewshot' in r)
    if sia:
        lines += ['', '## Few-shot (5-way, 200 episodes, mean +/- std)', '',
                  '| Run | Loss | Margin | 1-shot unseen | 5-shot unseen |'
                  ' 5-shot unseen (prototype) | 1-shot seen-base |',
                  '|---|---|---|---|---|---|---|']
        for k in sia:
            r, fs = runs[k], runs[k]['fewshot']
            m = r['margin'] if r['loss'] == 'contrastive' else \
                r['triplet_margin']

            def cell(key, proto=False):
                e = fs.get(key)
                if not e:
                    return '-'
                pre = 'proto_' if proto else ''
                return (f"{100 * e[pre + 'acc_mean']:.1f} +/- "
                        f"{100 * e[pre + 'acc_std']:.1f}")
            lines.append(f"| {k} | {r['loss']} | {m} | "
                         f"{cell('unseen_5way_1shot')} | "
                         f"{cell('unseen_5way_5shot')} | "
                         f"{cell('unseen_5way_5shot', True)} | "
                         f"{cell('seen_base_5way_1shot')} |")

    if 'silhouette_scores' in runs:
        lines += ['', '## Embedding quality (silhouette, test split)', '',
                  '| Model | dim | Euclidean | Cosine | t-SNE 2D |',
                  '|---|---|---|---|---|']
        for k, s in runs['silhouette_scores'].items():
            lines.append(f"| {DISPLAY.get(k, k)} | {s['dim']} | "
                         f"{s['silhouette_euclidean']:.3f} | "
                         f"{s['silhouette_cosine']:.3f} | "
                         f"{s['silhouette_tsne2d']:.3f} |")
    if 'eval_summary' in runs:
        lines += ['', '## Evaluation report summary', '',
                  '| Model | Top-1 | Top-5 | macro-F1 |', '|---|---|---|---|']
        for k, s in runs['eval_summary'].items():
            lines.append(f"| {DISPLAY.get(k, k)} | {_fmt(s['top1'])} | "
                         f"{_fmt(s['top5'])} | {s['macro_f1']:.3f} |")
    if 'tta' in runs:
        lines += ['', '## Test-time augmentation (bonus)', '',
                  '| Model | single clip | TTA | gain |', '|---|---|---|---|']
        for k, s in runs['tta'].items():
            lines.append(f"| {k} | {_fmt(s['single_top1'])} | "
                         f"{_fmt(s['tta_top1'])} | "
                         f"{_fmt(s['tta_top1'] - s['single_top1'])} |")
    if 'complexity' in runs:
        lines += ['', '## Model complexity (torchinfo, 1x3x16x112x112)', '',
                  '| Model | Params | Trainable | GMACs | GFLOPs |',
                  '|---|---|---|---|---|']
        note = {'siamese': ' - per clip pair', 'fusion': ' - RGB + flow'}
        for k, c in runs['complexity'].items():
            lines.append(f"| {DISPLAY.get(k, k)}{note.get(k, '')} | "
                         f"{c['params']:,} | "
                         f"{c['trainable_params']:,} | {c['mult_adds_G']:.1f}"
                         f" | {c['flops_G']:.1f} |")
    with open('results/results_tables.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    ctx = Ctx(args)
    for task in TASKS:  # fixed order: 'tables' last
        if task in args.tasks:
            print(f'\n##### {task} #####')
            globals()[f'task_{task}'](ctx)


if __name__ == '__main__':
    main()
