# Macenko Stain Normalization Protocol for ROSIE Fine-Tuning

## Table of Contents

1. [Background & Rationale](#1-background--rationale)
2. [Method Overview](#2-method-overview)
3. [Environment Setup](#3-environment-setup)
4. [Reference Slide Selection](#4-reference-slide-selection)
5. [Step-by-Step Pipeline](#5-step-by-step-pipeline)
6. [Reference Parameters (Locked)](#6-reference-parameters-locked)
7. [How to Normalize a New Slide](#7-how-to-normalize-a-new-slide)
8. [Validation & QC](#8-validation--qc)
9. [Troubleshooting](#9-troubleshooting)
10. [Appendix: Full Pipeline Script](#10-appendix-full-pipeline-script)

---

## 1. Background & Rationale

### Why stain normalization is needed

The ROSIE model (Wu et al., *Nature Communications* 2025) was trained on H&E slides that were:
- All stained **centrally on-site** using standard protocols
- All scanned on the **same scanner** (MoticEasyScan Pro 6N, 40x, default settings)
- **No stain normalization** was applied during training

The paper explicitly acknowledges this limitation:

> *"All data collected and imaged in our study was performed in-house on the same experimental setup (e.g., H&E scanner, PhenoCycler Fusion). Due to this uniformity, we have limited experimental evidence demonstrating our model's robustness on data sourced from significantly different environments."*

> *"Inter-batch effects due to staining technique, quality, and machinery are known to cause variations in the image statistics of H&E stains. Since predictions generated on H&E stains that significantly deviate from the training data are expected to perform worse."*

Since your fine-tuning and validation slides come from different labs/scanners than ROSIE's training data, stain normalization is necessary to bring all data into a consistent color space.

### Why Macenko

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **Macenko** | SVD-based stain vector decomposition in optical density space | Physically grounded in H&E absorption; fast; widely validated | Requires reference image |
| Reinhard | Color transfer in LAB space | Simple | Matches statistics, not stain biology |
| Vahadane | Sparse NMF decomposition | Structure-preserving | Slower; marginal gains over Macenko |

Macenko is the standard in computational pathology for cross-site normalization. It decomposes each pixel into hematoxylin and eosin concentrations via their optical density absorption spectra, then remaps those concentrations to match a reference slide's stain profile.

### Reference slide choice

Since ROSIE's raw training data is not publicly available, the fine-tuning slide is used as the reference:

```
Fine-tuning slide (REFERENCE)  -->  Model learns THIS stain distribution
         |                                         |
    Macenko fit                              Fine-tuned model
         |                                         |
New slides (TARGETS) --normalize-->  Input matches what model expects
```

---

## 2. Method Overview

The Macenko algorithm works as follows:

### Mathematical foundation

1. **RGB to Optical Density (OD):**
   ```
   OD = -log(RGB / 255)
   ```
   This transforms the Beer-Lambert light absorption into a linear space where stain concentrations are additive.

2. **Stain vector estimation via eigendecomposition:**
   - Compute covariance matrix of tissue-pixel OD values (background filtered out)
   - Take the two largest eigenvectors (principal stain directions)
   - Project all tissue pixels onto this 2D plane
   - Find the extreme angular percentiles (1st and 99th) to identify the pure H and E stain vectors

3. **Stain decomposition:**
   ```
   OD_image (N,3) = Concentrations (N,2) @ StainMatrix (2,3)
   ```
   Solve for concentrations using the pseudoinverse of the stain matrix.

4. **Normalization:**
   - Decompose the target image using its own stain vectors
   - Rescale concentrations to match the reference's 99th-percentile intensities
   - Reconstruct using the **reference** stain vectors

5. **OD back to RGB:**
   ```
   RGB = 255 * exp(-OD_normalized)
   ```

---

## 3. Environment Setup

### Required packages

All available in a standard Anaconda environment:

```bash
pip install numpy scipy scikit-image Pillow tifffile openslide-python
```

System-level dependency (macOS):
```bash
brew install openslide
```

### Format support

| Slide format | Library | Notes |
|---|---|---|
| `.qptiff` (OME-TIFF) | `tifffile` | Pyramid stored as SubIFDs; openslide reads only level 0 |
| `.ndpi` (Hamamatsu) | `openslide` | tifffile cannot parse NDPI; openslide handles natively |
| `.svs` (Aperio) | `openslide` | Full pyramid support |
| `.tiff` (generic) | `tifffile` or `openslide` | Depends on internal structure |
| `.png` / `.jpg` | `Pillow` or `skimage.io` | For pre-extracted patches |

---

## 4. Reference Slide Selection

### Current reference

| Property | Value |
|---|---|
| **File** | `SU17-19620 2J_HE_downsampled.qptiff` |
| **Role** | Fine-tuning paired H&E data |
| **Dimensions** | 30,868 x 47,025 px |
| **Resolution** | 0.4988 um/px (~20x equivalent) |
| **Format** | QPTIFF (OME-TIFF with 6 pyramid levels) |

### Selection criteria for a good reference region

When the normalizer extracts patches from the reference slide, it looks for:
- **Tissue fraction >= 70%** (avoids background-dominated patches)
- **Mix of tissue types**: nuclei (hematoxylin-rich), stroma/cytoplasm (eosin-rich), and some white space
- **No artifacts**: pen marks, folds, bubbles, or torn tissue

The pipeline automatically samples 50 tissue-rich 1024x1024 patches and uses the top 20 to fit the normalizer.

---

## 5. Step-by-Step Pipeline

### Step 1: Open slide and generate thumbnail

A low-resolution thumbnail (~2000px longest edge) is extracted from a pyramid level for fast tissue detection.

```python
import tifffile
import openslide

# For QPTIFF (fine-tuning reference)
tif = tifffile.TiffFile('path/to/reference.qptiff')
# Use a mid-level pyramid for thumbnail
thumb_data = tif.series[0].levels[4].asarray()[:, :, :3]

# For NDPI / SVS (target slides)
slide = openslide.OpenSlide('path/to/target.ndpi')
thumbnail = slide.get_thumbnail((1393, 2000))
```

### Step 2: Create tissue mask

Otsu thresholding on grayscale separates tissue (dark) from background (white):

```python
from skimage import color, filters, morphology

gray = color.rgb2gray(thumbnail)
threshold = filters.threshold_otsu(gray)
mask = gray < threshold
mask = morphology.remove_small_objects(mask, min_size=500)
mask = morphology.remove_small_holes(mask, area_threshold=500)
```

### Step 3: Extract tissue patches

Random sampling of 1024x1024 patches from tissue regions with >= 70% tissue content:

```python
# Convert thumbnail coords to full-resolution coords
full_x = int(thumb_x * scale_x)
full_y = int(thumb_y * scale_y)

# Read patch (method depends on format)
# QPTIFF: load pyramid level into memory, then slice
data = tif.series[0].levels[level].asarray()
patch = data[full_y:full_y+1024, full_x:full_x+1024, :3].copy()

# NDPI: use openslide read_region
region = slide.read_region((x_level0, y_level0), level, (1024, 1024))
patch = np.array(region)[:, :, :3]
```

### Step 4: Fit the Macenko normalizer on reference patches

```python
normalizer = MacenkoNormalizer()

# Stack ~20 reference patches into a composite image
composite = np.concatenate(reference_patches[:20], axis=0)

# Fit: estimates stain vectors and max concentrations
normalizer.fit(composite)
```

Output from fitting the current reference:
```
Hematoxylin vector: [0.0644, 0.9529, 0.2963]   (strong green-channel OD = blue-purple stain)
Eosin vector:       [0.6450, 0.7323, 0.2182]   (strong red-channel OD = pink stain)
Max concentrations: H=0.6328, E=1.2055
```

### Step 5: Normalize target patches

```python
normalized_patch = normalizer.transform(target_patch)
```

This:
1. Estimates the target patch's own stain vectors
2. Decomposes into H and E concentrations
3. Rescales concentrations to match the reference
4. Reconstructs using the reference stain vectors

### Step 6: Validate with W1 distance

Following the ROSIE paper's quality control methodology, compute the Wasserstein-1 distance between 256-bin intensity histograms:

```python
from scipy.stats import wasserstein_distance

ref_hist, _ = np.histogram(ref_pixels[:, ch], bins=256, range=(0, 255), density=True)
target_hist, _ = np.histogram(target_pixels[:, ch], bins=256, range=(0, 255), density=True)
w1 = wasserstein_distance(np.arange(256), np.arange(256), ref_hist, target_hist)
```

---

## 6. Reference Parameters (Locked)

These parameters were fit on the fine-tuning reference slide and **must remain constant** across all slides you normalize. They are saved in `output/macenko_reference_params.json`:

```json
{
  "stain_matrix_H": [0.0644, 0.9529, 0.2963],
  "stain_matrix_E": [0.6450, 0.7323, 0.2182],
  "max_concentration_H": 0.6328,
  "max_concentration_E": 1.2055,
  "reference_slide": "SU17-19620 2J_HE_downsampled.qptiff",
  "od_threshold": 0.15,
  "alpha_percentile": 1,
  "beta_percentile": 99
}
```

### What these mean

| Parameter | Value | Meaning |
|---|---|---|
| `stain_matrix_H` | [0.064, 0.953, 0.296] | Hematoxylin OD vector (R, G, B channels). Dominated by green-channel absorption = blue-purple stain |
| `stain_matrix_E` | [0.645, 0.732, 0.218] | Eosin OD vector. Higher red-channel absorption = pink stain |
| `max_concentration_H` | 0.6328 | 99th percentile hematoxylin concentration in the reference |
| `max_concentration_E` | 1.2055 | 99th percentile eosin concentration in the reference |
| `od_threshold` | 0.15 | Minimum optical density magnitude to be considered tissue (filters out background) |

---

## 7. How to Normalize a New Slide

### Quick usage (single patch or small image)

```python
import numpy as np
import json

# 1. Load the locked reference parameters
with open('output/macenko_reference_params.json') as f:
    params = json.load(f)

# 2. Create normalizer and load reference params
normalizer = MacenkoNormalizer()
normalizer.stain_matrix_ref = np.array([params['stain_matrix_H'], params['stain_matrix_E']])
normalizer.max_concentrations_ref = np.array([params['max_concentration_H'], params['max_concentration_E']])

# 3. Normalize any RGB uint8 image
normalized = normalizer.transform(my_image)
```

### Batch usage (many slides)

To normalize multiple slides, only change the `VALIDATION_SLIDE` path and rerun:

```python
# In macenko_normalize.py, change:
VALIDATION_SLIDE = '/path/to/new/slide.ndpi'  # <-- change this
OUTPUT_DIR = '/path/to/output/new_slide'       # <-- change this

# REFERENCE_SLIDE stays the same (it's your fine-tuning slide)
```

Or, for a batch loop:

```python
import glob

slides = glob.glob('/path/to/slides/*.ndpi')

# Fit once on reference
normalizer = MacenkoNormalizer()
normalizer.stain_matrix_ref = np.array([params['stain_matrix_H'], params['stain_matrix_E']])
normalizer.max_concentrations_ref = np.array([params['max_concentration_H'], params['max_concentration_E']])

# Normalize each slide's patches
for slide_path in slides:
    slide = openslide.OpenSlide(slide_path)
    for (x, y) in patch_coordinates:
        patch = read_patch(slide, x, y, 1024)
        normalized = normalizer.transform(patch)
        save_patch(normalized, output_path)
```

### Resolution matching

The reference slide is at **0.50 um/px (~20x)**. When normalizing slides at different resolutions:

| Source resolution | Action |
|---|---|
| ~0.50 um/px (20x) | Use directly (same as reference) |
| ~0.25 um/px (40x) | Read from pyramid level 1 (2x downsample), or resize patches after reading |
| ~1.00 um/px (10x) | Read from level 0 and resize up, or accept lower resolution |

The Macenko normalization itself is resolution-agnostic (it operates on color distributions, not spatial features), but matching resolution is important for consistency with the ROSIE model's 128x128 patch input.

---

## 8. Validation & QC

### W1 distance check

After normalizing each slide, compute the W1 distance to verify it moved closer to the reference:

| Metric | Before normalization | After normalization | Reduction |
|--------|---------------------|--------------------|----|
| **R channel** | 3.32 | 3.59 | -8.0% |
| **G channel** | 10.02 | 4.09 | **59.2%** |
| **B channel** | 20.91 | 7.64 | **63.5%** |
| **Grayscale** | 9.27 | 4.45 | **52.0%** |
| **Mean RGB** | **11.42** | **5.10** | **55.3%** |

These numbers are from the initial validation (`16-223-110_3LP.ndpi` normalized to `SU17-19620`). Use them as a benchmark. New slides should show similar or better W1 reductions.

### Acceptable ranges

- **Mean RGB W1 after normalization < 10**: Good. Slide is reasonably close to reference.
- **Mean RGB W1 after normalization 10-20**: Marginal. Inspect patches visually.
- **Mean RGB W1 after normalization > 20**: Poor. Slide may have severe artifacts, different stain type, or tissue-free regions contaminating the estimate.

### Visual QC checklist

For each normalized slide, inspect the comparison images:

- [ ] Nuclei should appear blue-purple (not brown, not black)
- [ ] Stroma/cytoplasm should appear pink (not orange, not yellow)
- [ ] White space should remain white (not tinted)
- [ ] Overall appearance should resemble the reference patches
- [ ] No color inversion or extreme saturation artifacts

---

## 9. Troubleshooting

### "Too few tissue pixels" error

**Cause**: The patch contains mostly background (white glass).
**Fix**: Increase `max_attempts` in `extract_patch_coords` or lower `MIN_TISSUE_FRACTION` from 0.7 to 0.5.

### Stain vectors are nearly identical (H ~ E)

**Cause**: The reference composite has too little color variance (e.g., all stroma, no nuclei).
**Fix**: Ensure reference patches include a mix of tissue types. Increase `NUM_REFERENCE_PATCHES`.

### W1 distance increases after normalization

**Cause**: Usually means the stain vectors were poorly estimated for the target slide.
**Fix**:
1. Inspect the target patches — do they have actual H&E tissue?
2. Try lowering `OPTICAL_DENSITY_THRESHOLD` from 0.15 to 0.10
3. Check if the slide has unusual staining (IHC instead of H&E, special stain, etc.)

### NDPI file can't be opened

**Cause**: Missing OpenSlide system library.
**Fix**: `brew install openslide` (macOS) or `apt-get install openslide-tools` (Linux).

### QPTIFF reads only 1 pyramid level via openslide

**Cause**: OpenSlide's generic-tiff reader doesn't traverse SubIFDs.
**Fix**: Use `tifffile` for QPTIFF/OME-TIFF files (already handled in the pipeline).

---

## 10. Appendix: Full Pipeline Script

The complete script is at:
```
/Users/sashurameshbabu/ROSIE/stain_normalization/macenko_normalize.py
```

### Output directory structure

```
output/
  macenko_reference_params.json    # Locked reference stain vectors (reuse for all slides)
  normalization_report.json        # W1 distance before/after + full report
  patches_reference/               # Sampled reference tissue patches
    ref_patch_000.png ... 004.png
  patches_validation/              # Original target tissue patches
    val_patch_000.png ... 004.png
  patches_normalized/              # Side-by-side comparisons (original | normalized)
    comparison_000.png ... 004.png
    normalized_000.png ... 004.png
  diagnostics/
    ref_thumbnail.png              # Reference slide thumbnail
    val_thumbnail.png              # Target slide thumbnail
    ref_tissue_mask.png            # Binary tissue mask (reference)
    val_tissue_mask.png            # Binary tissue mask (target)
    reference_patches_mosaic.png   # Grid of reference patches
    before_after_comparison.png    # 3-column: original | normalized | reference
```

### Key configuration parameters

```python
PATCH_SIZE = 1024              # Patch size in pixels at extraction resolution
NUM_REFERENCE_PATCHES = 50     # Patches sampled for fitting the normalizer
NUM_SAMPLE_PATCHES = 20        # Patches sampled for W1 distance
MIN_TISSUE_FRACTION = 0.7      # Minimum tissue content in valid patches
OPTICAL_DENSITY_THRESHOLD = 0.15  # Background filter in OD space
ALPHA = 1                      # Lower percentile for stain vector angles
BETA = 99                      # Upper percentile for stain vector angles
```
