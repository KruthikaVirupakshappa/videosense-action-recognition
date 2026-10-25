"""3D CNN architectures for video classification (Part 2).

* ``C3D`` -- Tran et al. (2015): VGG-style network with 3x3x3 convolutions.
* ``R2Plus1DBlock`` -- Tran et al. (2018): a 3D conv factorised into a
  1x3x3 spatial conv followed by a 3x1x1 temporal conv.
* ``build_r2plus1d_18`` -- torchvision R(2+1)D-18 pretrained on Kinetics-400.

Run ``python -m models.cnn3d`` to print the tensor shape after every
MaxPool3d (the shape trace requested in Section 6.1).
"""
import torch
import torch.nn as nn


def _conv_block(in_ch, out_ch, batch_norm):
    layers = [nn.Conv3d(in_ch, out_ch, kernel_size=(3, 3, 3),
                        padding=(1, 1, 1))]
    if batch_norm:
        layers.append(nn.BatchNorm3d(out_ch))
    layers.append(nn.ReLU())  # not in-place: keeps Grad-CAM hooks valid
    return layers


class C3D(nn.Module):
    """C3D architecture for video classification.

    Input:  (B, in_channels, T=16, H=112, W=112)
    Output: (B, num_classes)

    ``batch_norm=True`` inserts BatchNorm3d after every conv. The original
    C3D has no BN; with only ~950 training clips on the mini subset, BN makes
    training from scratch with Adam markedly more stable. Set it to False to
    reproduce the paper's exact layer list.
    """

    def __init__(self, num_classes=101, dropout=0.5, batch_norm=True,
                 in_channels=3):
        super().__init__()
        bn = batch_norm
        self.features = nn.Sequential(
            # Block 1: pool only spatially so early temporal info is kept
            *_conv_block(in_channels, 64, bn),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            # -> (B, 64, 16, 56, 56)
            # Block 2
            *_conv_block(64, 128, bn),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            # -> (B, 128, 8, 28, 28)
            # Block 3
            *_conv_block(128, 256, bn), *_conv_block(256, 256, bn),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            # -> (B, 256, 4, 14, 14)
            # Block 4
            *_conv_block(256, 512, bn), *_conv_block(512, 512, bn),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            # -> (B, 512, 2, 7, 7)
            # Block 5
            *_conv_block(512, 512, bn), *_conv_block(512, 512, bn),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            # -> (B, 512, 1, 3, 3)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 3 * 3, 4096), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        return self.classifier(self.features(x))

    def get_embedding(self, x):
        """Return the 4096-dim fc6 embedding (used for t-SNE)."""
        return self.classifier[:3](self.features(x))  # Flatten, fc6, ReLU

    def gradcam_layer(self):
        """Last Conv3d in the feature extractor (Grad-CAM target)."""
        return [m for m in self.features if isinstance(m, nn.Conv3d)][-1]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class R2Plus1DBlock(nn.Module):
    """One R(2+1)D residual block: spatial (1x3x3) then temporal (3x1x1)."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # Spatial: 1 x 3 x 3
        self.spatial = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, 1, 1),
                      bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        )
        # Temporal: 3 x 1 x 1
        self.temporal = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=(3, 1, 1),
                      stride=(stride, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1,
                          stride=(stride, stride, stride), bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        return torch.relu(self.temporal(self.spatial(x)) + self.shortcut(x))


def build_r2plus1d_18(num_classes, pretrained=True, freeze_backbone=True):
    """torchvision R(2+1)D-18 (Kinetics-400) with a new classifier head.

    With ``freeze_backbone=True`` only ``model.fc`` is trainable, which is the
    "fine-tune only the final classifier head" setting of Exercise 9.
    """
    from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

    weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
    model = r2plus1d_18(weights=weights)
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # trainable
    return model


def r2plus1d_embedding(model, x):
    """512-dim pooled feature of torchvision R(2+1)D (input to ``fc``)."""
    x = model.stem(x)
    x = model.layer4(model.layer3(model.layer2(model.layer1(x))))
    return model.avgpool(x).flatten(1)


def freeze_batchnorm(model):
    """Put every BatchNorm layer in eval mode (keeps pretrained statistics
    when the backbone is frozen but ``model.train()`` has been called)."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()


if __name__ == '__main__':
    # Shape trace through C3D: T shrinks 16 -> 16 -> 8 -> 4 -> 2 -> 1
    net = C3D(num_classes=10)
    x = torch.randn(1, 3, 16, 112, 112)
    print(f'input          {tuple(x.shape)}')
    for layer in net.features:
        x = layer(x)
        if isinstance(layer, nn.MaxPool3d):
            print(f'after MaxPool3d {tuple(x.shape)}')
    print(f'logits         {tuple(net.classifier(x).shape)}')
    blk = R2Plus1DBlock(64, 128, stride=2)
    print('R2Plus1DBlock  ', tuple(blk(torch.randn(1, 64, 8, 28, 28)).shape))
