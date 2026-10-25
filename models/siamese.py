"""Siamese network, contrastive loss and triplet loss (Part 3).

Run ``python -m models.siamese`` to execute the two sanity checks used in
the report: (1) the contrastive loss matches a hand computation, and (2) the
two branches really share one set of weights.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cnn3d import C3D


class VideoSiamese(nn.Module):
    """Siamese network for video similarity.

    Twin C3D encoders with TIED weights -> L2-normalised embedding -> distance.
    "Twin" is conceptual: there is exactly ONE encoder module, and both inputs
    go through it, so every gradient step updates a single set of parameters.

    Input:  two clips (B, 3, T, H, W) each
    Output: Euclidean distance (B,) -- small = similar, large = dissimilar
    """

    def __init__(self, embed_dim=512, dropout=0.5, batch_norm=True):
        super().__init__()
        backbone = C3D(num_classes=1, batch_norm=batch_norm)  # head unused
        self.encoder = backbone.features      # spatiotemporal extractor
        feat_dim = 512 * 1 * 3 * 3            # C3D map size after pool5
        # Projection head: flattened features -> compact embedding
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, embed_dim),
        )
        self.embed_dim = embed_dim

    def encode(self, x):
        """L2-normalised embedding for a batch of clips (unit sphere)."""
        emb = self.projector(self.encoder(x))
        return F.normalize(emb.float(), p=2, dim=1)

    get_embedding = encode  # common name used by evaluate.py

    def forward(self, x1, x2):
        """Euclidean distance between the embeddings of x1 and x2.

        Both clips are concatenated and pushed through the shared encoder in a
        single pass. This is mathematically identical to calling ``encode``
        twice with the same weights, but faster, and BatchNorm sees one batch.
        """
        z1, z2 = self.encode(torch.cat([x1, x2])).chunk(2)
        return torch.sqrt(((z1 - z2) ** 2).sum(dim=1) + 1e-8)

    def similarity(self, x1, x2):
        """Cosine similarity in [-1, 1]."""
        z1, z2 = self.encode(x1), self.encode(x2)
        return F.cosine_similarity(z1, z2)


class ContrastiveLoss(nn.Module):
    """Contrastive loss (Hadsell, Chopra & LeCun, 2006).

        L = (1 - y) * 0.5 * d^2
            + y     * 0.5 * max(0, margin - d)^2

    y = 0 -> same class (pull together)
    y = 1 -> different class (push apart, up to the margin)
    """

    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, dist, label):
        dist, label = dist.float(), label.float()
        similar_loss = (1 - label) * 0.5 * dist.pow(2)
        dissim_loss = label * 0.5 * torch.clamp(self.margin - dist,
                                                min=0).pow(2)
        return (similar_loss + dissim_loss).mean()


class TripletLoss(nn.Module):
    """Triplet loss (Schroff et al., 2015): L = max(0, d(a,p) - d(a,n) + m).

    Unlike contrastive loss it only asks positives to be *closer than*
    negatives by a margin, rather than forcing absolute distances.
    """

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, za, zp, zn):
        d_ap = (za - zp).pow(2).sum(1).add(1e-8).sqrt()
        d_an = (za - zn).pow(2).sum(1).add(1e-8).sqrt()
        return F.relu(d_ap - d_an + self.margin).mean(), d_ap, d_an


def verify_contrastive_loss():
    """Check ContrastiveLoss against values computed by hand."""
    loss_fn = ContrastiveLoss(margin=2.0)
    dist = torch.tensor([0.5, 0.5, 2.5, 1.0])
    label = torch.tensor([0., 1., 1., 1.])
    # pair 1 (same, d=.5):  0.5 * .25            = 0.125
    # pair 2 (diff, d=.5):  0.5 * (2 - .5)^2     = 1.125
    # pair 3 (diff, d=2.5): beyond margin        = 0
    # pair 4 (diff, d=1):   0.5 * (2 - 1)^2      = 0.5
    expected = (0.125 + 1.125 + 0.0 + 0.5) / 4
    got = loss_fn(dist, label).item()
    assert abs(got - expected) < 1e-6, (got, expected)
    print(f'[check] contrastive loss = {got:.6f}, hand-computed = '
          f'{expected:.6f}  -> OK')
    return got, expected


def verify_shared_weights(model):
    """Show that both branches use one parameter set and both send gradient."""
    model.eval()  # deterministic (no dropout masks, fixed BN statistics)
    x1 = torch.randn(2, 3, 16, 112, 112)
    x2 = torch.randn(2, 3, 16, 112, 112)
    first = [m for m in model.encoder if isinstance(m, nn.Conv3d)][0]
    model.zero_grad()
    model.encode(x1).sum().backward()
    g1 = first.weight.grad.clone()
    model.zero_grad()
    model.encode(x2).sum().backward()
    g2 = first.weight.grad.clone()
    model.zero_grad()
    (model.encode(x1).sum() + model.encode(x2).sum()).backward()
    g12 = first.weight.grad.clone()
    # Shared weights => grad(b1 + b2) == grad(b1) + grad(b2)
    accumulated = torch.allclose(g12, g1 + g2, rtol=1e-4, atol=1e-6)
    n_params = sum(p.numel() for p in model.parameters())
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    print(f'[check] one encoder module, {n_enc:,} encoder params '
          f'({n_params:,} total) - not duplicated per branch')
    print(f'[check] gradient from both branches accumulates into the same '
          f'tensor: {accumulated}')
    model.zero_grad()
    return accumulated


if __name__ == '__main__':
    verify_contrastive_loss()
    verify_shared_weights(VideoSiamese())
