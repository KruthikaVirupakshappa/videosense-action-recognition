"""Part 4 -- train two-stream (RGB + optical flow) models.

Ablation study (Exercise 22) + attention fusion (Exercise 23):
    python train_fusion.py --fusion rgb
    python train_fusion.py --fusion flow
    python train_fusion.py --fusion concat    # -> models/twostream_ucf101.pt
    python train_fusion.py --fusion average
    python train_fusion.py --fusion attention
    python train_fusion.py --fusion cross_attention

All variants share the same recipe (from scratch, 30 epochs) so the table is
an apples-to-apples comparison. ``--init_c3d models/c3d_ucf101.pt`` optionally
initialises both streams from the Part 2 C3D.
"""
import argparse
import os

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.fusion import FUSION_MODES, TwoStreamFusion
from utils.dataset import MultimodalDataset, UCF101Dataset
from utils.train_utils import (collect_logits, fusion_forward, get_device,
                               model_complexity, per_class_accuracy,
                               run_epoch, save_json, set_seed)
from utils.viz import plot_training_curves


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--fusion', choices=FUSION_MODES, default='concat')
    p.add_argument('--data_root', default='data/UCF101_mini')
    p.add_argument('--split_dir', default='data/splits/ucfTrainTestlist')
    p.add_argument('--cache_dir', default='data/frame_cache')
    p.add_argument('--flow_root', default='data/optical_flow')
    p.add_argument('--embed_dim', type=int, default=512)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--alpha_lr', type=float, default=1e-2,
                   help='learning rate of the attention-fusion alpha')
    p.add_argument('--label_smoothing', type=float, default=0.1)
    p.add_argument('--init_c3d', default=None,
                   help='optional C3D checkpoint to initialise both streams')
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--run_name', default=None)
    p.add_argument('--out', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    amp = not args.no_amp
    run = args.run_name or f'fusion_{args.fusion}'
    out = args.out or ('models/twostream_ucf101.pt'
                       if args.fusion == 'concat' and not args.run_name
                       else f'models/{run}.pt')

    cache = args.cache_dir or None
    train_ds = MultimodalDataset(
        UCF101Dataset(args.data_root,
                      os.path.join(args.split_dir, 'trainlist01.txt'),
                      train=True, cache_dir=cache), args.flow_root)
    val_ds = MultimodalDataset(
        UCF101Dataset(args.data_root,
                      os.path.join(args.split_dir, 'testlist01.txt'),
                      train=False, cache_dir=cache), args.flow_root)
    classes = train_ds.classes
    print(f'fusion={args.fusion} | {len(classes)} classes | train '
          f'{len(train_ds)} | val {len(val_ds)}')
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     pin_memory=True, persistent_workers=args.num_workers > 0)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kw)

    model = TwoStreamFusion(num_classes=len(classes),
                            embed_dim=args.embed_dim, fusion=args.fusion)
    if args.init_c3d:
        model.load_c3d_backbone(torch.load(args.init_c3d, map_location='cpu'))
        print(f'initialised streams from {args.init_c3d}')
    model = model.to(device)
    complexity = model_complexity(
        model, (torch.zeros(1, 3, 16, 112, 112, device=device),
                torch.zeros(1, 2, 16, 112, 112, device=device)),
        path=f'results/{run}_torchinfo.txt')

    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing)
    # alpha is a single scalar: Adam moves each parameter by ~lr per step, so
    # at lr=3e-4 it could barely leave 0.5 in 30 epochs. Give it its own group
    # (higher lr, no weight decay pulling it back to 0.5).
    alpha_params = [p for n, p in model.named_parameters()
                    if n == 'alpha_logit']
    other_params = [p for n, p in model.named_parameters()
                    if n != 'alpha_logit']
    groups = [{'params': other_params, 'weight_decay': 1e-4}]
    if alpha_params:
        groups.append({'params': alpha_params, 'lr': args.alpha_lr,
                       'weight_decay': 0.0})
    optimizer = Adam(groups, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {k: [] for k in ('train_loss', 'train_top1', 'val_loss',
                               'val_top1', 'val_top5', 'alpha')}
    for epoch in range(args.epochs):
        tr = run_epoch(model, train_dl, criterion, device, optimizer,
                       forward_fn=fusion_forward, amp=amp,
                       desc=f'train {epoch + 1}')
        va = run_epoch(model, val_dl, criterion, device,
                       forward_fn=fusion_forward, amp=amp, desc='val')
        scheduler.step()
        for k, v in (('train_loss', tr['loss']), ('train_top1', tr['top1']),
                     ('val_loss', va['loss']), ('val_top1', va['top1']),
                     ('val_top5', va['top5'])):
            history[k].append(v)
        alpha_msg = ''
        if args.fusion == 'attention':
            history['alpha'].append(model.alpha.item())
            alpha_msg = f' | alpha(rgb) {model.alpha.item():.3f}'
        print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {tr['loss']:.4f} | "
              f"Train Acc: {tr['top1']:.3f} | Val Top-1: {va['top1']:.3f} | "
              f"Val Top-5: {va['top5']:.3f}{alpha_msg}")

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    torch.save(model.state_dict(), out)
    logits, labels = collect_logits(model, val_dl, device,
                                    forward_fn=fusion_forward, amp=amp)
    result = {
        'run': run, 'fusion': args.fusion, 'checkpoint': out,
        'init_c3d': args.init_c3d, 'classes': classes, 'epochs': args.epochs,
        'final_top1': history['val_top1'][-1],
        'final_top5': history['val_top5'][-1],
        'best_top1': max(history['val_top1']),
        'train_top1_final': history['train_top1'][-1],
        'per_class_acc': per_class_accuracy(logits.argmax(1), labels,
                                            len(classes)),
        'alpha_final': history['alpha'][-1] if history['alpha'] else None,
        'complexity': complexity, 'history': history,
    }
    save_json(result, f'results/{run}.json')
    plot_training_curves(history, f'Two-stream ({args.fusion}): training '
                         'curves', f'figures/{run}_curves.png')
    print(f"Done. top-1 {result['final_top1']:.3f} top-5 "
          f"{result['final_top5']:.3f}" +
          (f" alpha {result['alpha_final']:.3f}"
           if result['alpha_final'] is not None else ''))


if __name__ == '__main__':
    main()
