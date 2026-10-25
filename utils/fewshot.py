"""N-way K-shot evaluation protocol for embedding models (Part 3)."""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.train_utils import autocast


@torch.no_grad()
def compute_embeddings(model, dataset, device, batch_size=32, num_workers=8,
                       amp=True):
    """Embed every clip of ``dataset`` once -> (N, D) tensor, (N,) labels.

    The evaluation transform is deterministic, so embedding each clip once
    and re-using it across episodes is equivalent to the handout's
    per-episode re-encoding, and ~100x cheaper.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    embs, labels = [], []
    for clips, lbls in tqdm(loader, desc='Embedding', leave=False):
        with autocast(device, amp):
            z = model.encode(clips.to(device))
        embs.append(z.float().cpu())
        labels.append(lbls)
    return torch.cat(embs), torch.cat(labels)


def few_shot_episodes(embs, labels, n_way=5, k_shot=1, n_query=15,
                      n_episodes=200, seed=0):
    """Run N-way K-shot episodes on pre-computed embeddings.

    For every episode: sample N classes, K support + n_query query clips per
    class, and classify each query by its nearest support embedding (1-NN,
    as in the handout). The prototype rule (nearest class-mean, Snell et al.
    2017) is reported as well; for K=1 both rules coincide.

    Returns dict with mean/std accuracy for both rules.
    """
    rng = random.Random(seed)
    labels = labels.numpy()
    class_idx = {c: np.flatnonzero(labels == c).tolist()
                 for c in np.unique(labels)}
    eligible = [c for c, idx in class_idx.items()
                if len(idx) >= k_shot + 1]
    assert len(eligible) >= n_way, 'not enough classes for this episode size'
    acc_nn, acc_proto = [], []
    for _ in range(n_episodes):
        classes = rng.sample(eligible, n_way)
        s_emb, s_lbl, q_emb, q_lbl = [], [], [], []
        for label_idx, cls in enumerate(classes):
            n_q = min(n_query, len(class_idx[cls]) - k_shot)
            idxs = rng.sample(class_idx[cls], k_shot + n_q)
            s_emb.append(embs[idxs[:k_shot]])
            s_lbl += [label_idx] * k_shot
            q_emb.append(embs[idxs[k_shot:]])
            q_lbl += [label_idx] * n_q
        s_emb, q_emb = torch.cat(s_emb), torch.cat(q_emb)
        s_lbl, q_lbl = torch.tensor(s_lbl), torch.tensor(q_lbl)
        # 1-NN in embedding space
        dists = torch.cdist(q_emb, s_emb)                     # (Q, N*K)
        pred = s_lbl[dists.argmin(1)]
        acc_nn.append((pred == q_lbl).float().mean().item())
        # nearest prototype (class mean of the K supports)
        protos = s_emb.view(n_way, k_shot, -1).mean(1)
        pred_p = torch.cdist(q_emb, protos).argmin(1)
        acc_proto.append((pred_p == q_lbl).float().mean().item())
    return {'n_way': n_way, 'k_shot': k_shot, 'episodes': n_episodes,
            'acc_mean': float(np.mean(acc_nn)),
            'acc_std': float(np.std(acc_nn)),
            'proto_acc_mean': float(np.mean(acc_proto)),
            'proto_acc_std': float(np.std(acc_proto))}


def few_shot_eval(model, dataset, device, n_way=5, k_shot=1, n_query=15,
                  n_episodes=200, seed=0):
    """Handout-compatible entry point: returns (mean, std) 1-NN accuracy."""
    embs, labels = compute_embeddings(model, dataset, device)
    res = few_shot_episodes(embs, labels, n_way, k_shot, n_query,
                            n_episodes, seed)
    return res['acc_mean'], res['acc_std']
