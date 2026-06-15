# Data Pipeline

This document outlines the data preprocessing pipeline for the MIMII machine sound dataset.

## Preprocessing Steps
1. Audio loading: Convert stereo to mono, resample to 16kHz, pad/trim to 10s.
2. Mel Spectrogram: Extract with 128 mel bins, 512 hop length, 2048 n_fft.
3. Scaling: Convert to log scale (dB).
4. Splits: 80/10/10 stratified split to preserve class distributions per machine type.
