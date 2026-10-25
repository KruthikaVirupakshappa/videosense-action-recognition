"""Part 3 -- train the Siamese network and run N-way K-shot evaluation.

Protocol: the Siamese encoder is trained on BASE classes only (by default
30 UCF101 classes disjoint from the 10 mini classes). Few-shot evaluation is
then done on the 10 mini classes, which the network has never seen -- this
is what "select N unseen action classes" in Section 7.5 requires.

Examples:
    python train_siamese.py                                  # Exercises 14-15
    python train_siamese.py --margin 1.0 --run_name siamese_m1.0 --epochs 10
    python train_siamese.py --loss triplet --run_name siamese_triplet
"""
import argparse
import os
import random

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.siamese import (ContrastiveLoss, TripletLoss, VideoSiamese,
                            verify_contrastive_loss)
from utils.dataset import (MINI_CLASSES, PairDataset, TripletDataset,
                           UCF101Dataset, list_classes)
from utils.fewshot import compute_embeddings, few_shot_episodes
from utils.train_utils import (autocast, get_device, model_complexity,
                               save_json, set_seed)
from utils.viz import plot_distance_histograms_per_epoch, plot_siamese_training


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--video_root', default='data/UCF101',
                   help='FULL UCF101 (base classes come from here)')
    p.add_argument('--split_dir', default='data/splits/ucfTrainTestlist')
    p.add_argument('--cache_dir', default='data/frame_cache')
    p.add_argument('--eval_classes', nargs='+', default=MINI_CLASSES,
                   help='held-out classes for few-shot evaluation')
    p.add_argument('--num_base_classes', type=int, default=30)
    p.add_argument('--base_classes_file', default='results/base_classes.txt',
                   help='re-used if it exists so every run trains on the '
                        'same base classes')
    p.add_argument('--loss', choices=['contrastive', 'triplet'],
                   default='contrastive')
    p.add_argument('--margin', type=float, default=2.0,
                   help='contrastive margin')
    p.add_argument('--triplet_margin', type=float, default=0.5)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--pairs_per_epoch', type=int, default=4000)
    p.add_argument('--val_triplets', type=int, default=600)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--embed_dim', type=int, default=512)
    p.add_argument('--episodes', type=int, default=200)
    p.add_argument('--n_query', type=int, default=15)
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--run_name', default=None)
    p.add_argument('--out', default=None)
    return p.parse_args()


def choose_base_classes(args):
    """Pick (and persist) base classes disjoint from the evaluation classes."""
    if os.path.exists(args.base_classes_file):
        with open(args.base_classes_file) as f:
            return [line.strip() for line in f if line.strip()]
    pool = [c for c in list_classes(args.video_root)
            if c not in set(args.eval_classes)]
    base = sorted(random.Random(args.seed).sample(
        pool, min(args.num_base_classes, len(pool))))
    os.makedirs(os.path.dirname(args.base_classes_file) or '.', exist_ok=True)
    with open(args.base_classes_file, 'w') as f:
        f.write('\n'.join(base) + '\n')
    return base


class FixedTriplets(Dataset):
    """A fixed (seeded) set of validation triplets so that distance
    histograms are comparable across epochs and across runs."""

    def __init__(self, base_dataset, n, seed=0):
        rng = random.Random(seed)
        self.ds = base_dataset
        cls_idx = {c: i for c, i in base_dataset.class_idx.items()
                   if len(i) > 1}
        classes = list(cls_idx)
        self.items = []
        for _ in range(n):
            ca, cn = rng.sample(classes, 2)
            a, p = rng.sample(cls_idx[ca], 2)
            self.items.append((a, p, rng.choice(cls_idx[cn])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        a, p, n = self.items[i]
        return self.ds[a][0], self.ds[p][0], self.ds[n][0]


def embed_triplet(model, xa, xp, xn):
    """Encode three clip batches with the ONE shared encoder."""
    return model.encode(torch.cat([xa, xp, xn])).chunk(3)


@torch.no_grad()
def validate(model, loader, device, args, amp):
    """Positive (anchor-positive) and negative (anchor-negative) distances
    on the fixed validation triplets, plus the validation loss."""
    model.eval()
    pos, neg = [], []
    for xa, xp, xn in tqdm(loader, desc='val', leave=False):
        with autocast(device, amp):
            za, zp, zn = embed_triplet(model, xa.to(device), xp.to(device),
                                       xn.to(device))
        pos.append((za - zp).norm(dim=1).cpu())
        neg.append((za - zn).norm(dim=1).cpu())
    pos, neg = torch.cat(pos), torch.cat(neg)
    if args.loss == 'contrastive':
        # identical to ContrastiveLoss on a 50/50 set of pairs
        loss = 0.5 * (0.5 * pos.pow(2).mean() +
                      0.5 * (args.margin - neg).clamp(min=0).pow(2).mean())
    else:
        loss = (pos - neg + args.triplet_margin).clamp(min=0).mean()
    return loss.item(), pos.numpy(), neg.numpy()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    amp = not args.no_amp
    run = args.run_name or 'siamese'
    out = args.out or ('models/siamese_ucf101.pt' if run == 'siamese'
                       else f'models/{run}.pt')
    verify_contrastive_loss()

    base_classes = choose_base_classes(args)
    overlap = set(base_classes) & set(args.eval_classes)
    assert not overlap, f'base/eval overlap: {overlap}'
    print(f'{len(base_classes)} base classes (training), '
          f'{len(args.eval_classes)} unseen classes (few-shot evaluation)')

    split = lambda name: os.path.join(args.split_dir, name)  # noqa: E731
    cache = args.cache_dir or None
    train_base = UCF101Dataset(args.video_root, split('trainlist01.txt'),
                               train=True, cache_dir=cache,
                               classes=base_classes)
    val_base = UCF101Dataset(args.video_root, split('testlist01.txt'),
                             train=False, cache_dir=cache,
                             classes=base_classes)
    eval_unseen = UCF101Dataset(args.video_root, split('testlist01.txt'),
                                train=False, cache_dir=cache,
                                classes=args.eval_classes)
    print(f'train {len(train_base)} | val {len(val_base)} | '
          f'unseen eval {len(eval_unseen)} clips')

    if args.loss == 'contrastive':
        train_set = PairDataset(train_base, args.pairs_per_epoch)
        criterion = ContrastiveLoss(margin=args.margin)
    else:
        train_set = TripletDataset(train_base, args.pairs_per_epoch)
        criterion = TripletLoss(margin=args.triplet_margin)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     pin_memory=True, persistent_workers=args.num_workers > 0)
    train_dl = DataLoader(train_set, shuffle=True, drop_last=True,
                          **loader_kw)
    val_dl = DataLoader(FixedTriplets(val_base, args.val_triplets),
                        shuffle=False, **loader_kw)

    model = VideoSiamese(embed_dim=args.embed_dim).to(device)
    x = torch.zeros(1, 3, 16, 112, 112, device=device)
    complexity = model_complexity(model, (x, x),
                                  path=f'results/{run}_torchinfo.txt')
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {k: [] for k in ('train_loss', 'val_loss', 'train_pos_mean',
                               'train_neg_mean', 'val_pos_mean',
                               'val_neg_mean')}
    dist_log = []
    for epoch in range(args.epochs):
        model.train()
        tot, n, tr_pos, tr_neg = 0.0, 0, [], []
        for batch in tqdm(train_dl, desc=f'train {epoch + 1}', leave=False):
            batch = [b.to(device, non_blocking=True) for b in batch]
            with autocast(device, amp):
                if args.loss == 'contrastive':
                    x1, x2, y = batch
                    dist = model(x1, x2)          # shared-weight forward
                    loss = criterion(dist, y)
                    tr_pos.append(dist[y == 0].detach().float().cpu())
                    tr_neg.append(dist[y == 1].detach().float().cpu())
                else:
                    za, zp, zn = embed_triplet(model, *batch)
                    loss, d_ap, d_an = criterion(za, zp, zn)
                    tr_pos.append(d_ap.detach().float().cpu())
                    tr_neg.append(d_an.detach().float().cpu())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            tot += loss.item() * batch[0].size(0)
            n += batch[0].size(0)
        scheduler.step()

        val_loss, vpos, vneg = validate(model, val_dl, device, args, amp)
        history['train_loss'].append(tot / n)
        history['val_loss'].append(val_loss)
        history['train_pos_mean'].append(float(torch.cat(tr_pos).mean()))
        history['train_neg_mean'].append(float(torch.cat(tr_neg).mean()))
        history['val_pos_mean'].append(float(vpos.mean()))
        history['val_neg_mean'].append(float(vneg.mean()))
        dist_log.append({'epoch': epoch + 1,
                         'pos': np.round(vpos, 4).tolist(),
                         'neg': np.round(vneg, 4).tolist()})
        print(f'Epoch {epoch + 1}/{args.epochs} | train loss {tot / n:.4f} | '
              f'val loss {val_loss:.4f} | val d+ {vpos.mean():.3f} | '
              f'val d- {vneg.mean():.3f}')

    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    torch.save(model.state_dict(), out)

    # -- N-way K-shot evaluation (Exercise 15) --
    fewshot = {}
    for split_name, ds in (('unseen', eval_unseen), ('seen_base', val_base)):
        embs, labels = compute_embeddings(model, ds, device,
                                          num_workers=args.num_workers,
                                          amp=amp)
        for k in (1, 5):
            res = few_shot_episodes(embs, labels, n_way=5, k_shot=k,
                                    n_query=args.n_query,
                                    n_episodes=args.episodes, seed=args.seed)
            fewshot[f'{split_name}_5way_{k}shot'] = res
            print(f"[{split_name}] 5-way {k}-shot: {res['acc_mean']:.3f} "
                  f"+/- {res['acc_std']:.3f} (prototype "
                  f"{res['proto_acc_mean']:.3f})")

    result = {'run': run, 'loss': args.loss, 'margin': args.margin,
              'triplet_margin': args.triplet_margin, 'epochs': args.epochs,
              'checkpoint': out, 'base_classes': base_classes,
              'eval_classes': args.eval_classes, 'fewshot': fewshot,
              'complexity': complexity, 'history': history,
              'dist_log': dist_log}
    save_json(result, f'results/{run}.json')
    margin = (args.margin if args.loss == 'contrastive'
              else args.triplet_margin)
    title = f'{run} ({args.loss}, margin {margin})'
    plot_siamese_training(history, title, f'figures/{run}_curves.png')
    plot_distance_histograms_per_epoch(
        dist_log, f'{title}: validation pair distances per epoch',
        f'figures/{run}_distance_hist.png')


if __name__ == '__main__':
    main()
