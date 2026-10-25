# VideoSense: action recognition on UCF101 with 3D CNNs, Siamese networks and two-stream fusion

DATA 255 Extra Credit HW3. This repo holds the code for all five parts of the
assignment. It was developed for a Google Colab **A100** runtime and also runs
on CPU or Apple MPS (slowly).

The quickest route is to open **`VideoSense_Colab.ipynb`** in Colab and run
the cells top to bottom. The notebook calls the scripts below in order.

## Layout

```
videosense/
├── data/                    # created at run time (videos, splits, caches)
├── models/
│   ├── cnn3d.py             # C3D, R2Plus1DBlock, pretrained R(2+1)D-18 builder
│   ├── siamese.py           # VideoSiamese, ContrastiveLoss, TripletLoss, sanity checks
│   └── fusion.py            # TwoStreamFusion (rgb / flow / concat / average / attention / cross_attention)
├── utils/
│   ├── dataset.py           # UCF101Dataset, Pair/Triplet/Multimodal datasets, frame cache
│   ├── optical_flow.py      # Farneback flow pre-computation + HSV colourisation
│   ├── fewshot.py           # N-way K-shot episode evaluation
│   ├── train_utils.py       # epoch loop, top-k accuracy, AMP, torchinfo complexity
│   └── viz.py               # every plot in the report
├── setup_project.py         # creates the directory structure
├── subset_classes.py        # builds the 10-class mini-UCF101
├── prepare_data.py          # verify dataset, cache frames, compute optical flow
├── part1_visualize.py       # Part 1 figures
├── train_3dcnn.py           # Part 2: C3D from scratch / R(2+1)D fine-tuning
├── train_siamese.py         # Part 3: Siamese training + 5-way 1/5-shot evaluation
├── train_fusion.py          # Part 4: two-stream models + ablation
├── evaluate.py              # Part 5: report, confusion, t-SNE, Grad-CAM, TTA, tables
├── make_diagrams.py         # architecture diagrams for the report
└── VideoSense_Colab.ipynb   # end-to-end driver notebook
```

## Setup

```bash
pip install -r requirements.txt
python setup_project.py
```

## Data

```bash
cd data
wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
unrar x -idq UCF101.rar && mv UCF-101 UCF101       # the archive's top folder is UCF-101/
unzip -q UCF101TrainTestSplits-RecognitionTask.zip -d splits/   # -> splits/ucfTrainTestlist/
cd ..
python subset_classes.py                 # data/UCF101_mini (10 classes)
python prepare_data.py --verify          # expect 101 classes / 13,320 videos
python prepare_data.py --frames mini base --flow
```

`prepare_data.py` runs two one-time steps:

* **Frame cache.** Each video is decoded once into 32 uniformly spaced frames
  at 128x171 and stored as uint8 `.npy` in `data/frame_cache/`. Training then
  takes a random 112x112 crop with horizontal flip, and picks 16 of the 32
  frames with TSN-style random segment sampling, which adds temporal jitter.
  Evaluation uses the centre crop and 16 uniform frames.
* **Optical flow.** Farneback flow is computed at 16 positions per video at
  112x112 and saved as `(2, T, H, W)` float32 in `data/optical_flow/`. This
  is done for the mini classes only.

`base` caches the 30 Siamese training classes (see Part 3).

## Running each part from scratch

| Part | Command | Outputs |
|---|---|---|
| 1 | `python part1_visualize.py` | `figures/part1_*.png` |
| 2 | `python train_3dcnn.py --model c3d --label_smoothing 0.1` | `models/c3d_ucf101.pt`, `results/c3d.json`, `figures/c3d_curves.png` |
| 2 | `python train_3dcnn.py --model c3d --label_smoothing 0.0 --run_name c3d_ls0.0` (repeat with 0.2) | label-smoothing sweep |
| 2 | `python train_3dcnn.py --model r2plus1d --epochs 10` | `models/r2plus1d.pt` (head-only fine-tune) |
| 3 | `python train_siamese.py` | `models/siamese_ucf101.pt`, few-shot results, distance histograms |
| 3 | `python train_siamese.py --margin 1.0 --run_name siamese_m1.0` (and 4.0) | margin study |
| 3 | `python train_siamese.py --loss triplet --run_name siamese_triplet` | contrastive vs triplet |
| 4 | `python train_fusion.py --fusion {rgb,flow,concat,average,attention,cross_attention}` | `models/twostream_ucf101.pt` (concat) + ablation |
| 5 | `python evaluate.py` | confusion matrices, t-SNE, Grad-CAM GIFs, TTA, `results/results_tables.md` |
| - | `python make_diagrams.py` | `figures/arch_*.png` |

Every training script writes a JSON file to `results/` with the per-epoch
history, final top-1/top-5, per-class accuracy and torchinfo complexity, plus
a curves PNG in `figures/`. `python evaluate.py --tasks tables` collects them
all into `results/results_tables.md`.

## Design decisions and fixes to the handout code

1. **Split parsing.** `testlist01.txt` has no label column, so the handout's
   `if len(parts) >= 2` drops every test video. The 101-class index in
   `trainlist01.txt` is also wrong for the 10-class subset. Labels now come
   from the class folder name, using the dataset's own `class_to_idx`.
2. **BatchNorm in C3D.** `C3D(batch_norm=True)` adds BatchNorm3d after each
   conv, which keeps training from scratch on ~950 clips stable. Pass
   `--no_bn` for the paper-exact network. The default LR is 3e-4 (Adam)
   instead of 1e-3, because fc6 alone has 19M parameters.
3. **Siamese protocol.** The handout asks for few-shot evaluation on unseen
   classes. The encoder is trained on 30 UCF101 classes that do not overlap
   the mini set (`results/base_classes.txt`, seed 42) and is evaluated 5-way
   on the 10 mini classes. Embeddings are computed once and reused across the
   200 episodes. With a deterministic eval transform this gives the same
   result as re-encoding per episode, at a fraction of the cost. Both the
   1-NN rule from the handout and the prototype (class-mean) rule are
   reported.
4. **Shared weights.** `VideoSiamese` has exactly one encoder module. Both
   clips go through it in a single concatenated batch.
   `python -m models.siamese` checks this, and also checks the contrastive
   loss against a hand computation.
5. **Attention fusion.** `alpha = sigmoid(logit)` is initialised at 0.5 and
   has its own optimizer group (lr 1e-2, no weight decay). At lr 3e-4 a
   single scalar could barely move away from 0.5 in 30 epochs.
6. **Grad-CAM.** Hooks are attached to the last Conv3d. The CAM (2x7x7 for
   C3D) is upsampled trilinearly to 16x112x112 and normalised globally, so
   differences between frames stay visible. Two quantitative checks are
   reported: the correlation between the CAM and the optical-flow magnitude,
   and the share of CAM mass that falls on the 20% most-moving pixels (0.20
   would be chance).
7. **Mixed precision.** bfloat16 autocast is used on CUDA. It is native on
   the A100 and needs no GradScaler. Disable it with `--no_amp`.

## Reproducibility

Every script seeds Python, NumPy and PyTorch (`--seed`, default 42). The
reported numbers come from the last-epoch weights. A `*_best.pt` is also
saved for reference, but the test split is never used for model selection.
