# Retinal Vessel Segmentation

A Python implementation of the retinal vessel segmentation pipeline proposed by Abdushkour et al. (2023), with an added novelty: **Frangi vesselness filter fusion** to improve detection of fine capillaries. The pipeline was tested on the [DRIVE dataset](https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction).

> **Reference Paper:**
> Abdushkour H, Soomro TA, Ali A, Ali Jandan F, Jelinek H, Memon F, et al.
> *"Enhancing fine retinal vessel segmentation: Morphological reconstruction and double thresholds filtering strategy."*
> PLoS ONE 18(7): e0288792, 2023.
> https://doi.org/10.1371/journal.pone.0288792

---

## What This Implementation Does

This code faithfully reproduces the five-phase pipeline from the paper and extends it with a Hessian-based Frangi filter fused into the vessel coherence stage (Phase 3). This implementation was evaluated on the DRIVE dataset.

---

## Pipeline Overview

```
Fundus Image
     │
     ▼
Phase 1: Pre-processing & Light Reflex Removal
         (Green channel extraction, top-hat/black-hat morphology, CLAHE)
     │
     ▼
Phase 2: Homomorphic + Wiener Filtering
         (Illumination normalisation, adaptive noise removal)
     │
     ▼
Phase 3: Multi-scale LoG + Frangi Fusion + Anisotropic Diffusion  ← Novelty
         (2nd-order Gaussian detector, Hessian vesselness, Perona-Malik diffusion)
     │
     ▼
Phase 4 & 5: Double Thresholding + Morphological Reconstruction
             (Adaptive T_L / T_U, reconstruction by dilation, small-object removal)
     │
     ▼
Segmented Vessel Map + Evaluation Metrics
```

---

## Novelty Over the Original Paper

The original paper uses a multi-scale Laplacian-of-Gaussian (LoG) detector followed by anisotropic diffusion. While effective for large and medium vessels, the LoG kernel can miss very fine capillaries.

This implementation adds a **Frangi vesselness filter** (Frangi et al., 1998) operating on the same Wiener-filtered green channel. The Frangi response and the LoG response are fused via pixel-wise maximum before diffusion is applied. This lets the LoG handle the dominant vessel structure while the Hessian-based Frangi filter recovers thin tubular structures the LoG alone would suppress.

---

## Requirements

```
Python >= 3.8
opencv-python
numpy
matplotlib
Pillow
scipy
scikit-image
```

Install all dependencies:

```bash
pip install opencv-python numpy matplotlib Pillow scipy scikit-image
```

---

## Dataset

This implementation was tested on the **DRIVE** (Digital Retinal Images for Vessel Extraction) dataset, which contains 40 colour fundus images (768 × 584 px) split equally into training and test sets, each with a corresponding binary ground truth annotation.

Download: https://www.kaggle.com/datasets/andrewmvd/drive-digital-retinal-images-for-vessel-extraction

Expected file formats: fundus images as `.tif`, ground truth masks as `.gif`.

---

## Usage

Update the image paths at the bottom of `segmentation.py`:

```python
IMAGE_PATH        = "test/21_training.tif"
GROUND_TRUTH_PATH = "test/21_manual1.gif"
```

Then run:

```bash
python segmentation.py
```

Each phase produces a `matplotlib` visualisation of its intermediate outputs. If the ground truth file is not found, the script skips evaluation and outputs only the segmented vessel map.

---

## Pipeline Details

### Phase 1 — Pre-processing & Light Reflex Removal

Extracts the green channel (highest vessel-to-background contrast in RGB fundus images) and builds a binary retinal mask to exclude the black circular border. Top-hat and black-hat morphological operations are combined to suppress the centre-light-reflex artefact. CLAHE (Contrast Limited Adaptive Histogram Equalisation) is then applied to boost local contrast.

### Phase 2 — Enhancement

Homomorphic filtering separates the slow-varying illumination component from the high-frequency reflectance in the frequency domain, normalising uneven background brightness. Wiener filtering (adaptive, pixel-by-pixel) then removes residual Gaussian noise while preserving vessel edges.

### Phase 3 — Vessel Coherence (Novelty: Hybrid Hessian–Laplacian)

A 2nd-order oriented Gaussian derivative kernel (Eq. 13 of the paper) is convolved with the Wiener-filtered green channel across two width scales (`σ_v ∈ {4, 5}`), seven elongation factors (`0.5–3.5`), and 12 orientations. Scale-normalised responses are max-pooled to form the LoG vessel map.

The **Frangi vesselness filter** is applied in parallel on the same input and fused with the LoG map via pixel-wise maximum — the key addition of this implementation, targeting capillaries missed by LoG alone.

Anisotropic diffusion (Perona–Malik) is applied to the fused response, smoothing the background while preserving vessel-shaped ridges. Diffusion stops early using an entropy-based criterion (rate of change < 0.001).

### Phase 4 & 5 — Binarisation & Reconstruction

A lower threshold `T_L = mean + 0.7 × std` is derived from the image intensity histogram. An upper threshold `T_U` is derived from the mean intensity of Canny-detected edge pixels. Morphological reconstruction by dilation uses `T_U` as the marker and `T_L` as the mask to recover weakly connected vessel branches. Connected components smaller than 70 pixels are removed as noise.

---

## Evaluation Metrics

All metrics are computed within the retinal field of view only (inside the retinal mask). AUC follows the paper's definition of `(Se + Sp) / 2`.

| Metric | Formula |
|---|---|
| Accuracy (AC) | (TP + TN) / (TP + TN + FP + FN) |
| Sensitivity (Se) | TP / (TP + FN) |
| Specificity (Sp) | TN / (TN + FP) |
| AUC | (Se + Sp) / 2 |

---



## Citation

If you use or build upon this work, please cite the original paper:

```
Abdushkour H, Soomro TA, Ali A, Ali Jandan F, Jelinek H, Memon F, et al.
Enhancing fine retinal vessel segmentation: Morphological reconstruction and
double thresholds filtering strategy.
PLoS ONE 18(7): e0288792 (2023).
https://doi.org/10.1371/journal.pone.0288792
```