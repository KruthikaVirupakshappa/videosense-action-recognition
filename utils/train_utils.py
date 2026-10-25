"""Training / evaluation helpers shared by the train_*.py scripts."""
import json
import os
import random
from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm


def get_device():
    """CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast(device, enabled=True):
    """bfloat16 autocast on CUDA (A100 supports it natively; no GradScaler
    needed). Falls back to full precision elsewhere."""
    if enabled and device.type == 'cuda' and torch.cuda.is_bf16_supported():
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return nullcontext()


def top_k_accuracy(logits, labels, k=5):
    """Top-k accuracy: correct if the true label is among the top-k logits."""
    k = min(k, logits.shape[1])
    top_k = logits.topk(k, dim=1).indices                       # (B, k)
    correct = top_k.eq(labels.unsqueeze(1).expand_as(top_k))
    return correct.any(dim=1).float().mean().item()


def rgb_forward(model, batch, device):
    """Forward for single-input models; batch = (clips, labels)."""
    clips, labels = batch
    return model(clips.to(device, non_blocking=True)), labels.to(device)


def fusion_forward(model, batch, device):
    """Forward for the two-stream model; batch = (rgb, flow, labels)."""
    rgb, flow, labels = batch
    return (model(rgb.to(device, non_blocking=True),
                  flow.to(device, non_blocking=True)), labels.to(device))


def run_epoch(model, loader, criterion, device, optimizer=None,
              forward_fn=rgb_forward, amp=True, train_hook=None, desc=''):
    """One pass over ``loader``. Trains if ``optimizer`` is given.

    Returns dict(loss, top1, top5) averaged over samples.
    """
    training = optimizer is not None
    model.train(training)
    if training and train_hook:
        train_hook(model)  # e.g. freeze BN of a frozen backbone
    total_loss, n = 0.0, 0
    all_logits, all_labels = [], []
    grad_ctx = nullcontext() if training else torch.no_grad()
    with grad_ctx:
        for batch in tqdm(loader, desc=desc, leave=False):
            with autocast(device, amp):
                logits, labels = forward_fn(model, batch, device)
                loss = criterion(logits.float(), labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            bs = labels.size(0)
            total_loss += loss.item() * bs
            n += bs
            all_logits.append(logits.detach().float().cpu())
            all_labels.append(labels.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    return {'loss': total_loss / n,
            'top1': top_k_accuracy(logits, labels, 1),
            'top5': top_k_accuracy(logits, labels, 5)}


@torch.no_grad()
def collect_logits(model, loader, device, forward_fn=rgb_forward, amp=True):
    """Return (logits, labels) for the whole loader."""
    model.eval()
    all_logits, all_labels = [], []
    for batch in tqdm(loader, desc='Predict', leave=False):
        with autocast(device, amp):
            logits, labels = forward_fn(model, batch, device)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def per_class_accuracy(preds, labels, num_classes):
    preds, labels = np.asarray(preds), np.asarray(labels)
    return [float((preds[labels == c] == c).mean()) if (labels == c).any()
            else float('nan') for c in range(num_classes)]


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def model_complexity(model, input_data, path=None):
    """torchinfo summary -> dict(params, trainable, mult_adds, flops).

    torchinfo reports multiply-accumulates; FLOPs ~= 2 x MACs.
    """
    from torchinfo import summary

    info = summary(model, input_data=input_data, verbose=0, depth=3,
                   col_names=('output_size', 'num_params', 'mult_adds'))
    if path:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            f.write(str(info))
    return {'params': int(info.total_params),
            'trainable_params': int(info.trainable_params),
            'mult_adds_G': info.total_mult_adds / 1e9,
            'flops_G': 2 * info.total_mult_adds / 1e9}
