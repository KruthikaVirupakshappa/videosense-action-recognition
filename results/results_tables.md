# VideoSense results tables

## Classification (mini UCF101, testlist01)

| Model | Top-1 | Top-5 | Best-epoch top-1 | Train top-1 | Params | GFLOPs |
|---|---|---|---|---|---|---|
| c3d | 51.8% | 87.7% | 52.3% | 89.7% | 63,362,186 | 77.1 |
| r2plus1d | 94.2% | 100.0% | 94.2% | 99.9% | 31,305,255 | 81.0 |
| fusion_attention | 61.7% | 92.1% | 62.6% | 99.2% | 60,045,899 | 153.4 |
| fusion_average | 66.7% | 94.4% | 67.0% | 99.4% | 60,051,028 | 153.4 |
| fusion_concat | 67.0% | 93.0% | 67.3% | 98.6% | 60,051,018 | 153.4 |
| fusion_cross_attention | 63.7% | 92.7% | 64.3% | 99.7% | 63,203,914 | 153.4 |
| fusion_flow | 51.5% | 90.9% | 55.6% | 94.8% | 30,024,650 | 76.4 |
| fusion_rgb | 57.3% | 90.4% | 59.1% | 91.8% | 30,026,378 | 77.0 |

## Label smoothing (C3D)

| label_smoothing | run | Top-1 | Top-5 | Train top-1 | train-val gap |
|---|---|---|---|---|---|
| 0.0 | c3d_ls0.0 | 51.5% | 90.6% | 86.2% | 34.8% |
| 0.1 | c3d | 51.8% | 87.7% | 89.7% | 37.9% |
| 0.2 | c3d_ls0.2 | 61.4% | 90.9% | 92.0% | 30.6% |

## Two-stream ablation

| Variant | Top-1 | Top-5 | learned alpha (RGB) |
|---|---|---|---|
| rgb | 57.3% | 90.4% | - |
| flow | 51.5% | 90.9% | - |
| concat | 67.0% | 93.0% | - |
| average | 66.7% | 94.4% | - |
| attention | 61.7% | 92.1% | 0.436 |
| cross_attention | 63.7% | 92.7% | - |

## Few-shot (5-way, 200 episodes, mean +/- std)

| Run | Loss | Margin | 1-shot unseen | 5-shot unseen | 5-shot unseen (prototype) | 1-shot seen-base |
|---|---|---|---|---|---|---|
| siamese | contrastive | 2.0 | 30.7 +/- 8.6 | 36.2 +/- 6.9 | 35.7 +/- 7.3 | 48.4 +/- 12.5 |
| siamese_m1.0 | contrastive | 1.0 | 26.7 +/- 6.0 | 33.8 +/- 6.5 | 28.2 +/- 6.2 | 43.9 +/- 12.1 |
| siamese_m4.0 | contrastive | 4.0 | 25.6 +/- 6.6 | 34.7 +/- 7.0 | 28.2 +/- 6.6 | 40.1 +/- 11.6 |
| siamese_triplet | triplet | 0.5 | 35.5 +/- 10.2 | 40.9 +/- 8.2 | 41.3 +/- 9.3 | 54.7 +/- 13.9 |

## Embedding quality (silhouette, test split)

| Model | dim | Euclidean | Cosine | t-SNE 2D |
|---|---|---|---|---|
| C3D (scratch) | 4096 | 0.084 | 0.106 | 0.021 |
| Siamese (C3D) | 512 | -0.191 | -0.290 | -0.187 |
| Two-stream (concat) | 1024 | 0.118 | 0.185 | 0.136 |
| R(2+1)D-18 (pretrained) | 512 | 0.165 | 0.285 | 0.453 |

## Evaluation report summary

| Model | Top-1 | Top-5 | macro-F1 |
|---|---|---|---|
| C3D (scratch) | 51.8% | 87.7% | 0.511 |
| R(2+1)D-18 (pretrained) | 93.9% | 100.0% | 0.939 |
| Two-stream (concat) | 67.0% | 93.0% | 0.672 |

## Test-time augmentation (bonus)

| Model | single clip | TTA | gain |
|---|---|---|---|
| c3d | 51.8% | 51.5% | -0.3% |
| r2plus1d | 93.9% | 93.9% | 0.0% |

## Model complexity (torchinfo, 1x3x16x112x112)

| Model | Params | Trainable | GMACs | GFLOPs |
|---|---|---|---|---|
| C3D (scratch) | 63,362,186 | 63,362,186 | 38.6 | 77.1 |
| R(2+1)D-18 (pretrained) | 31,305,255 | 5,130 | 40.5 | 81.0 |
| Siamese (C3D) - per clip pair | 32,905,856 | 32,905,856 | 77.1 | 154.1 |
| Two-stream (concat) - RGB + flow | 60,051,018 | 60,051,018 | 76.7 | 153.4 |
