# Inria Aerial Image Labeling — Semantic Segmentation (U-Net + ResNet34)

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, memory-efficient deep learning pipeline for binary building footprint segmentation (**Building vs. Not Building**) on the **Inria Aerial Image Labeling Dataset**. Built with PyTorch, `segmentation-models-pytorch`, `albumentations`, and `rasterio`.

---

## Project Overview

High-resolution remote sensing imagery presents two major challenges for deep learning:
1. **Memory Constraints**: Each tile in the Inria dataset is a `5000 x 5000` RGB GeoTIFF (~75 MB uncompressed). Loading all 180 training tiles into RAM simultaneously requires over 13 GB of raw memory.
2. **Cross-City Domain Shift**: The training set covers five cities (`austin`, `chicago`, `kitsap`, `tyrol-w`, `vienna`), while the test set evaluates zero-shot generalization on five completely unseen cities (`bellingham`, `bloomington`, `innsbruck`, `sfo`, `tyrol-e`) ranging from sparse alpine villages to dense metropolitan cores.

### Key Features of This Pipeline
- **On-the-Fly Windowed GeoTIFF Reading**: Uses `rasterio.windows.Window` to read `512 x 512` crops directly from disk without loading full `5000 x 5000` rasters into RAM.
- **Cross-City Validation Strategies**: Supports the official Inria validation split (tiles `1..5` per city held out) as well as a `leave_one_city_out` switch to benchmark unseen-city generalization.
- **Domain-Shift Augmentations**: Spatial dihedral transforms (`HorizontalFlip`, `VerticalFlip`, `RandomRotate90`) combined with radiometric perturbations (`ColorJitter`, `HueSaturationValue`, `GaussianBlur`) to simulate varying atmospheric and sensor conditions.
- **Mixed Precision (AMP) Training**: Uses `torch.amp` with `BCEDiceLoss`, `AdamW`, Cosine Annealing learning rate scheduling, and gradient clipping—optimized to run smoothly even on 4 GB VRAM laptop GPUs.
- **Gaussian-Blended Sliding-Window Inference**: Stitches `512 x 512` predictions with 50% overlap (`stride = 256`) using a 2D Gaussian weight kernel to eliminate tile boundary seams on full `5000 x 5000` tiles.
- **Global Pixel-Count Metric Accumulation**: Computes exact dataset-level and per-city Intersection-over-Union (IoU) and Pixel Accuracy by accumulating global `TP`, `FP`, `FN`, and `TN` pixel counts rather than averaging per-tile ratios.

---

## Dataset Setup

1. Download the official **Inria Aerial Image Labeling Dataset** from the [Inria project page](https://project.inria.fr/aerialimagelabeling/).
2. Extract the archive on your machine. The expected directory structure is[cite: 1]:

```text
AerialImageDataset/
├── train/
│   ├── images/      # 180 RGB GeoTIFFs (austin1.tif ... vienna36.tif, 5000x5000 px, 0.3m/px)
│   └── gt/          # 180 single-channel uint8 masks (0 = background, 255 = building)
└── test/
    └── images/      # 180 RGB GeoTIFFs (bellingham1.tif ... tyrol-e36.tif)