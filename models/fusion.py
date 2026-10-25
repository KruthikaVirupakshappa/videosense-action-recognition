"""Two-stream (RGB + optical flow) fusion network (Part 4).

One class covers every model in the ablation study via ``fusion``:

=============  ==============================================================
``rgb``        RGB stream only (appearance)
``flow``       Flow stream only (motion)
``concat``     late fusion: concatenate embeddings -> classifier (default)
``average``    late fusion: each stream has its own classifier; average logits
``attention``  learned scalar alpha: z = alpha*z_rgb + (1-alpha)*z_flow
``cross_attention``  flow tokens attend to RGB tokens and vice versa
=============  ==============================================================
"""
import torch
import torch.nn as nn

from models.cnn3d import C3D

FUSION_MODES = ('rgb', 'flow', 'concat', 'average', 'attention',
                'cross_attention')


class StreamEncoder(nn.Module):
    """C3D feature extractor + projection to an ``embed_dim`` vector."""

    def __init__(self, in_channels, embed_dim, batch_norm=True):
        super().__init__()
        backbone = C3D(num_classes=1, batch_norm=batch_norm,
                       in_channels=in_channels)
        self.encoder = backbone.features
        feat_dim = 512 * 1 * 3 * 3
        self.head = nn.Sequential(nn.Flatten(),
                                  nn.Linear(feat_dim, embed_dim), nn.ReLU())

    def forward(self, x):
        return self.head(self.encoder(x))

    def tokens(self, x):
        """pool5 feature map as a token sequence: (B, T'*H'*W' = 9, 512)."""
        return self.encoder(x).flatten(2).transpose(1, 2)

    def gradcam_layer(self):
        return [m for m in self.encoder if isinstance(m, nn.Conv3d)][-1]


class TwoStreamFusion(nn.Module):
    """Late-fusion two-stream network.

    Input:  rgb (B, 3, T, H, W), flow (B, 2, T, H, W)
    Output: logits (B, num_classes)
    """

    def __init__(self, num_classes=101, embed_dim=512, dropout=0.5,
                 fusion='concat', batch_norm=True):
        super().__init__()
        assert fusion in FUSION_MODES, fusion
        self.fusion = fusion
        self.use_rgb = fusion != 'flow'
        self.use_flow = fusion != 'rgb'
        if self.use_rgb:
            self.rgb_stream = StreamEncoder(3, embed_dim, batch_norm)
        if self.use_flow:
            # Flow stream: 2-channel input (u, v)
            self.flow_stream = StreamEncoder(2, embed_dim, batch_norm)

        if fusion == 'average':
            self.rgb_classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))
            self.flow_classifier = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(embed_dim, num_classes))
        else:
            in_dim = (embed_dim * 2 if fusion in ('concat', 'cross_attention')
                      else embed_dim)
            self.classifier = nn.Sequential(nn.Dropout(dropout),
                                            nn.Linear(in_dim, num_classes))
        if fusion == 'cross_attention':
            # Queries from one modality, keys/values from the other, over the
            # 9 spatiotemporal pool5 positions of each stream (512-d tokens).
            self.flow_to_rgb = nn.MultiheadAttention(512, num_heads=8,
                                                     batch_first=True)
            self.rgb_to_flow = nn.MultiheadAttention(512, num_heads=8,
                                                     batch_first=True)
            self.norm_rgb = nn.LayerNorm(512)
            self.norm_flow = nn.LayerNorm(512)
            self.proj = nn.Sequential(nn.Linear(1024, embed_dim * 2),
                                      nn.ReLU())
        if fusion == 'attention':
            # alpha = sigmoid(alpha_logit); logit 0 -> alpha initialised at 0.5
            self.alpha_logit = nn.Parameter(torch.zeros(1))

    @property
    def alpha(self):
        """Learned weight of the RGB stream (attention fusion only)."""
        return torch.sigmoid(self.alpha_logit)

    def get_stream_embeddings(self, rgb, flow):
        """Return both stream embeddings (useful for analysis)."""
        z_rgb = self.rgb_stream(rgb) if self.use_rgb else None
        z_flow = self.flow_stream(flow) if self.use_flow else None
        return z_rgb, z_flow

    def cross_attend(self, rgb, flow):
        """Cross-modal attention: each stream's tokens query the other's.

        Flow queries attend to RGB keys/values ("what does the moving region
        look like?") and RGB queries attend to flow ("how is this object
        moving?"). Residual + LayerNorm, mean-pool, concatenate.
        """
        t_rgb = self.rgb_stream.tokens(rgb)
        t_flow = self.flow_stream.tokens(flow)
        f2r, _ = self.flow_to_rgb(t_flow, t_rgb, t_rgb)
        r2f, _ = self.rgb_to_flow(t_rgb, t_flow, t_flow)
        z_flow = self.norm_flow(t_flow + f2r).mean(1)
        z_rgb = self.norm_rgb(t_rgb + r2f).mean(1)
        return self.proj(torch.cat([z_rgb, z_flow], dim=1))

    def get_embedding(self, rgb, flow):
        """Fused embedding fed to the classifier (used for t-SNE)."""
        if self.fusion == 'cross_attention':
            return self.cross_attend(rgb, flow)
        z_rgb, z_flow = self.get_stream_embeddings(rgb, flow)
        if self.fusion == 'rgb':
            return z_rgb
        if self.fusion == 'flow':
            return z_flow
        if self.fusion == 'attention':
            return self.alpha * z_rgb + (1 - self.alpha) * z_flow
        return torch.cat([z_rgb, z_flow], dim=1)  # concat and average

    def forward(self, rgb, flow):
        if self.fusion == 'average':
            z_rgb, z_flow = self.get_stream_embeddings(rgb, flow)
            return 0.5 * (self.rgb_classifier(z_rgb) +
                          self.flow_classifier(z_flow))
        return self.classifier(self.get_embedding(rgb, flow))

    def load_c3d_backbone(self, state_dict):
        """Initialise both streams from a trained C3D checkpoint (optional).

        The flow stream's first conv receives the RGB filters averaged over
        the colour channels (cross-modality initialisation, Wang et al. 2016).
        """
        feats = {k[len('features.'):]: v for k, v in state_dict.items()
                 if k.startswith('features.')}
        if self.use_rgb:
            self.rgb_stream.encoder.load_state_dict(feats)
        if self.use_flow:
            flow_feats = dict(feats)
            w = feats['0.weight']                     # (64, 3, 3, 3, 3)
            flow_feats['0.weight'] = w.mean(dim=1, keepdim=True).repeat(
                1, 2, 1, 1, 1)
            self.flow_stream.encoder.load_state_dict(flow_feats)
