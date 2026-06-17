# Data Pipeline

This document outlines the data preprocessing pipeline for the MIMII machine sound dataset.

## Preprocessing Steps
1. Audio loading: Convert stereo to mono, resample to 16kHz, pad/trim to 10s.
2. Mel Spectrogram: Extract with 128 mel bins, 512 hop length, 2048 n_fft.
3. Scaling: Convert to log scale (dB).
4. Splits: 80/10/10 stratified split to preserve class distributions per machine type.

## MVTec AD Visual Pipeline
The visual pipeline processes anomaly detection datasets, focusing on `bottle`, `cable`, and `carpet` categories.

### Category List & Defect Types
| Category | Domain | Defect Types |
|----------|--------|--------------|
| Bottle   | Object | Contamination, broken, print error |
| Cable    | Object | Bent wire, missing cable, cut |
| Carpet   | Texture| Color, cut, hole, metal contamination |

### Dataset Statistics & Processing
- **Loading Strategy:** Mask-aware loading returns `(image, mask, label)`.
- **Labels:** 0 for Normal (good), 1 for Anomaly (defect).
- **Masks:** Normal images receive a synthetic 0-tensor mask. Anomalous images load ground-truth pixel masks.
- **Class Balance:** Severe class imbalance is handled using a `WeightedRandomSampler` during DataLoader creation.
- **Augmentations:** ImageNet normalizations, RandomCrop(224), ColorJitter for training. FiveCrop + Horizontal Flip for TTA.
