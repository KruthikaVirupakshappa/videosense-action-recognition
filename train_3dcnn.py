"""Part 2 -- train C3D from scratch or fine-tune pretrained R(2+1)D-18.

Examples:
    # C3D, 30 epochs, label smoothing 0.1 (Exercise 7)
    python train_3dcnn.py --model c3d --label_smoothing 0.1
    # label-smoothing sweep (Exercise 8)
    python train_3dcnn.py --model c3d --label_smoothing 0.0 \
        --run_name c3d_ls0.0
    # R(2+1)D-18, head-only fine-tuning for 10 epochs (Exercise 9)
    python train_3dcnn.py --model r2plus1d --epochs 10 --lr 1e-3
"""
import argparse
import os

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.cnn3d import C3D, build_r2plus1d_18, freeze_batchnorm
from utils.dataset import (IMAGENET_MEAN, IMAGENET_STD, KINETICS_MEAN,
                           KINETICS_STD, UCF101Dataset)
from utils.train_utils import (collect_logits, get_device, model_complexity,
                               per_class_accuracy, run_epoch, save_json,
                               set_seed)
from utils.viz import plot_training_curves


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--model', choices=['c3d', 'r2plus1d'], default='c3d')
    p.add_argument('--data_root', default='data/UCF101_mini')
    p.add_argument('--split_dir', default='data/splits/ucfTrainTestlist')
    p.add_argument('--cache_dir', default='data/frame_cache',
                   help="frame cache from prepare_data.py ('' to decode "
                        'videos on the fly)')
    p.add_argument('--num_frames', type=int, default=16)
    p.add_argument('--spatial', type=int, default=112)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=None,
                   help='default: 3e-4 for C3D, 1e-3 for the R(2+1)D head')
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--label_smoothing', type=float, default=0.1)
    p.add_argument('--no_bn', action='store_true',
                   help='C3D without BatchNorm (paper-exact)')
    p.add_argument('--finetune_all', action='store_true',
                   help='R(2+1)D: also update the backbone')
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--run_name', default=None)
    p.add_argument('--out', default=None, help='checkpoint path')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f'Training on: {device}')

    is_c3d = args.model == 'c3d'
    run = args.run_name or ('c3d' if is_c3d else 'r2plus1d')
    out = args.out or ('models/c3d_ucf101.pt' if is_c3d and not args.run_name
                       else f'models/{run}.pt')
    lr = args.lr or (3e-4 if is_c3d else 1e-3)
    mean, std = ((IMAGENET_MEAN, IMAGENET_STD) if is_c3d
                 else (KINETICS_MEAN, KINETICS_STD))

    # -- Data --
    cache = args.cache_dir or None
    common = dict(num_frames=args.num_frames, spatial_size=args.spatial,
                  cache_dir=cache, mean=mean, std=std)
    train_ds = UCF101Dataset(args.data_root,
                             os.path.join(args.split_dir, 'trainlist01.txt'),
                             train=True, **common)
    val_ds = UCF101Dataset(args.data_root,
                           os.path.join(args.split_dir, 'testlist01.txt'),
                           train=False, **common)
    num_classes = len(train_ds.classes)
    print(f'{num_classes} classes | train {len(train_ds)} | '
          f'val {len(val_ds)} clips')
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     pin_memory=True, persistent_workers=args.num_workers > 0)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kw)

    # -- Model, Loss, Optimizer --
    if is_c3d:
        model = C3D(num_classes=num_classes, batch_norm=not args.no_bn)
        train_hook = None
    else:
        model = build_r2plus1d_18(num_classes, pretrained=True,
                                  freeze_backbone=not args.finetune_all)
        # frozen backbone -> keep its Kinetics BatchNorm statistics fixed
        train_hook = None if args.finetune_all else freeze_batchnorm
    model = model.to(device)
    complexity = model_complexity(
        model, torch.zeros(1, 3, args.num_frames, args.spatial, args.spatial,
                           device=device),
        path=f'results/{run}_torchinfo.txt')
    print(f"params {complexity['params']:,} (trainable "
          f"{complexity['trainable_params']:,}) | "
          f"{complexity['flops_G']:.1f} GFLOPs / clip")

    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(params, lr=lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # -- Training loop --
    history = {k: [] for k in ('train_loss', 'train_top1', 'val_loss',
                               'val_top1', 'val_top5', 'lr')}
    best = 0.0
    for epoch in range(args.epochs):
        tr = run_epoch(model, train_dl, criterion, device, optimizer,
                       amp=not args.no_amp, train_hook=train_hook,
                       desc=f'train {epoch + 1}')
        va = run_epoch(model, val_dl, criterion, device,
                       amp=not args.no_amp, desc='val')
        history['lr'].append(scheduler.get_last_lr()[0])
        scheduler.step()
        for k, v in (('train_loss', tr['loss']), ('train_top1', tr['top1']),
                     ('val_loss', va['loss']), ('val_top1', va['top1']),
                     ('val_top5', va['top5'])):
            history[k].append(v)
        print(f"Epoch {epoch + 1}/{args.epochs} | Loss: {tr['loss']:.4f} | "
              f"Train Acc: {tr['top1']:.3f} | Val Loss: {va['loss']:.4f} | "
              f"Val Top-1: {va['top1']:.3f} | Val Top-5: {va['top5']:.3f}")
        if va['top1'] >= best:
            best = va['top1']
            os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
            torch.save(model.state_dict(), out.replace('.pt', '_best.pt'))
    # Final-epoch weights are the reported model (no test-set model selection)
    torch.save(model.state_dict(), out)

    logits, labels = collect_logits(model, val_dl, device,
                                    amp=not args.no_amp)
    preds = logits.argmax(1)
    result = {
        'run': run, 'model': args.model, 'checkpoint': out,
        'label_smoothing': args.label_smoothing, 'lr': lr,
        'epochs': args.epochs, 'classes': train_ds.classes,
        'final_top1': history['val_top1'][-1],
        'final_top5': history['val_top5'][-1],
        'best_top1': best,
        'train_top1_final': history['train_top1'][-1],
        'per_class_acc': per_class_accuracy(preds, labels, num_classes),
        'complexity': complexity, 'history': history,
    }
    save_json(result, f'results/{run}.json')
    plot_training_curves(
        history, f'{run}: training curves (label smoothing '
        f'{args.label_smoothing})', f'figures/{run}_curves.png')
    print(f"Done. final top-1 {result['final_top1']:.3f}, top-5 "
          f"{result['final_top5']:.3f} -> {out}")


if __name__ == '__main__':
    main()
