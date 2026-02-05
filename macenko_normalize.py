"""
Macenko Stain Normalization Pipeline for ROSIE Fine-Tuning
==========================================================

Reference: Macenko et al., "A method for normalizing histology slides for
quantitative analysis," ISBI 2009.

This script:
1. Extracts tissue patches from the fine-tuning H&E slide (reference)
2. Extracts tissue patches from the validation H&E slide (target)
3. Computes W1 (Wasserstein) distance BEFORE normalization
4. Fits a Macenko stain normalizer on the reference slide
5. Normalizes the validation slide to match the reference
6. Computes W1 distance AFTER normalization
7. Saves normalized output and diagnostic visualizations

Usage:
    python macenko_normalize.py

Author: Generated for ROSIE fine-tuning pipeline
"""

import os
import sys
import time
import warnings
import numpy as np
from pathlib import Path
from scipy import linalg
from scipy.stats import wasserstein_distance
from skimage import io, color, filters, morphology
from PIL import Image
import json

# Increase PIL max image size for large patches
Image.MAX_IMAGE_PIXELS = None

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths
REFERENCE_SLIDE = '/Users/sashurameshbabu/Pair 6 scaled/SU17-19620 2J_HE_downsampled.qptiff'
VALIDATION_SLIDE = '/Users/sashurameshbabu/Documents/validation/16-223-110_3LP.ndpi'
OUTPUT_DIR = '/Users/sashurameshbabu/ROSIE/stain_normalization/output'

# Patch extraction settings
PATCH_SIZE = 1024          # Patch size in pixels (at the extraction resolution)
NUM_REFERENCE_PATCHES = 50 # Number of tissue patches for fitting the normalizer
NUM_SAMPLE_PATCHES = 20    # Number of patches for W1 distance computation
TISSUE_THRESHOLD = 0.15    # Otsu-based tissue detection threshold (fraction of tissue)
MIN_TISSUE_FRACTION = 0.7  # Minimum tissue content in a valid patch

# Macenko parameters
OPTICAL_DENSITY_THRESHOLD = 0.15  # Minimum OD for valid pixels
ALPHA = 1                         # Percentile for robust stain vector estimation
BETA = 99                         # Percentile for robust stain vector estimation


# ============================================================================
# MACENKO STAIN NORMALIZATION
# ============================================================================

class MacenkoNormalizer:
    """
    Macenko stain normalization.

    Decomposes H&E images into optical density (OD) space, estimates
    hematoxylin and eosin stain vectors via SVD, and normalizes images
    by mapping their stain vectors to a reference.
    """

    def __init__(self):
        self.stain_matrix_ref = None
        self.max_concentrations_ref = None

    def _rgb_to_od(self, img):
        """Convert RGB image to optical density (OD) space."""
        img = img.astype(np.float64)
        # Avoid log(0)
        img = np.clip(img, 1, 255)
        od = -np.log(img / 255.0)
        return od

    def _od_to_rgb(self, od):
        """Convert optical density back to RGB."""
        rgb = 255.0 * np.exp(-od)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return rgb

    def _get_stain_matrix(self, img, od_threshold=OPTICAL_DENSITY_THRESHOLD):
        """
        Estimate the stain matrix (H&E stain vectors) using Macenko's SVD method.

        Steps:
        1. Convert to OD space
        2. Filter out background pixels (low OD)
        3. Compute covariance-based SVD of the OD matrix
        4. Project onto the plane of the two largest eigenvectors
        5. Find extreme angles to identify H and E vectors
        6. Order so Hematoxylin (more blue) is first, Eosin (more red) is second
        """
        od = self._rgb_to_od(img)

        # Reshape to (N, 3)
        od_flat = od.reshape(-1, 3)

        # Filter out background (low optical density)
        od_magnitude = np.sqrt(np.sum(od_flat ** 2, axis=1))
        mask = od_magnitude > od_threshold
        od_filtered = od_flat[mask]

        if od_filtered.shape[0] < 100:
            raise ValueError(
                f"Too few tissue pixels ({od_filtered.shape[0]}) found. "
                "Try lowering the OD threshold or using a patch with more tissue."
            )

        # Compute covariance and eigenvectors (more stable than SVD on raw data)
        cov = np.cov(od_filtered.T)  # (3, 3)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # eigh returns in ascending order; take the two largest
        # eigenvectors[:, -1] is the largest, eigenvectors[:, -2] is second largest
        V = eigenvectors[:, -2:].T  # (2, 3) - the two principal directions

        # Ensure eigenvectors point into the positive OD quadrant
        for i in range(2):
            if V[i].sum() < 0:
                V[i] = -V[i]

        # Project tissue pixels onto this plane
        projections = od_filtered @ V.T  # (N, 2)

        # Find the angle of each projected point
        angles = np.arctan2(projections[:, 1], projections[:, 0])

        # Find the extreme angles (robust percentiles)
        min_angle = np.percentile(angles, ALPHA)
        max_angle = np.percentile(angles, BETA)

        # The two stain vectors are at these extreme angles
        vec1 = np.array([np.cos(min_angle), np.sin(min_angle)]) @ V
        vec2 = np.array([np.cos(max_angle), np.sin(max_angle)]) @ V

        # Normalize to unit length
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)

        # Order: Hematoxylin first, Eosin second
        # In OD space, Hematoxylin has relatively more blue absorption (OD channel 2)
        # relative to red (OD channel 0). Eosin has more red absorption.
        # A robust check: H has a higher blue-to-red OD ratio
        ratio1 = vec1[2] / (vec1[0] + 1e-10)  # blue/red ratio
        ratio2 = vec2[2] / (vec2[0] + 1e-10)

        if ratio1 > ratio2:
            stain_matrix = np.array([vec1, vec2])
        else:
            stain_matrix = np.array([vec2, vec1])

        return stain_matrix

    def _get_concentrations(self, img, stain_matrix):
        """Get stain concentrations by decomposing OD into stain basis.

        OD (N, 3) = C (N, 2) @ S (2, 3)
        C = OD @ pinv(S)
        pinv(S) has shape (3, 2), so: C (N, 2) = OD (N, 3) @ pinv(S) (3, 2)
        """
        od = self._rgb_to_od(img)
        od_flat = od.reshape(-1, 3)

        # stain_matrix is (2, 3). pinv gives (3, 2)
        concentrations = od_flat @ np.linalg.pinv(stain_matrix)
        return concentrations

    def fit(self, img):
        """
        Fit the normalizer on a reference image.

        Parameters
        ----------
        img : np.ndarray
            Reference H&E image (RGB, uint8), shape (H, W, 3)
        """
        print("  Fitting Macenko normalizer on reference image...")
        self.stain_matrix_ref = self._get_stain_matrix(img)

        concentrations = self._get_concentrations(img, self.stain_matrix_ref)
        self.max_concentrations_ref = np.percentile(concentrations, 99, axis=0)

        print(f"  Reference stain matrix (H, E):")
        print(f"    Hematoxylin: [{self.stain_matrix_ref[0][0]:.4f}, {self.stain_matrix_ref[0][1]:.4f}, {self.stain_matrix_ref[0][2]:.4f}]")
        print(f"    Eosin:       [{self.stain_matrix_ref[1][0]:.4f}, {self.stain_matrix_ref[1][1]:.4f}, {self.stain_matrix_ref[1][2]:.4f}]")
        print(f"  Reference max concentrations: H={self.max_concentrations_ref[0]:.4f}, E={self.max_concentrations_ref[1]:.4f}")

    def transform(self, img):
        """
        Normalize a target image to match the reference stain.

        Parameters
        ----------
        img : np.ndarray
            Target H&E image (RGB, uint8), shape (H, W, 3)

        Returns
        -------
        np.ndarray
            Normalized image (RGB, uint8), shape (H, W, 3)
        """
        h, w, c = img.shape

        # Get the target's stain matrix
        stain_matrix_target = self._get_stain_matrix(img)

        # Get concentrations in the target's stain space
        concentrations = self._get_concentrations(img, stain_matrix_target)

        # Get max concentrations of target
        max_concentrations_target = np.percentile(concentrations, 99, axis=0)

        # Normalize concentrations to reference scale
        concentrations *= (self.max_concentrations_ref / (max_concentrations_target + 1e-10))

        # Reconstruct using reference stain matrix
        od_normalized = concentrations @ self.stain_matrix_ref

        # Convert back to RGB
        normalized = self._od_to_rgb(od_normalized.reshape(h, w, 3))

        return normalized


# ============================================================================
# SLIDE I/O UTILITIES
# ============================================================================

def open_reference_slide(path):
    """Open the QPTIFF reference slide using tifffile."""
    import tifffile
    print(f"Opening reference slide with tifffile: {os.path.basename(path)}")
    tif = tifffile.TiffFile(path)

    # Get base level dimensions
    base = tif.series[0].levels[0]
    print(f"  Base dimensions: {base.shape}")
    print(f"  Pyramid levels: {len(tif.series[0].levels)}")

    return tif


def open_validation_slide(path):
    """Open the NDPI validation slide using openslide."""
    import openslide
    print(f"Opening validation slide with OpenSlide: {os.path.basename(path)}")
    slide = openslide.OpenSlide(path)

    print(f"  Base dimensions: {slide.dimensions}")
    print(f"  Pyramid levels: {slide.level_count}")
    mpp = float(slide.properties.get('openslide.mpp-x', 0))
    print(f"  MPP: {mpp:.4f} µm/px")

    return slide


def get_thumbnail_from_tifffile(tif, target_size=2000):
    """Get a thumbnail from a tifffile object using lower pyramid levels."""
    # Find a suitable pyramid level
    levels = tif.series[0].levels
    for i, level in enumerate(levels):
        h, w = level.shape[0], level.shape[1]
        if max(h, w) <= target_size * 2:
            print(f"  Using pyramid level {i} ({w}x{h}) for thumbnail")
            data = level.asarray()
            if data.ndim == 3 and data.shape[2] >= 3:
                data = data[:, :, :3]  # Keep only RGB
            # Resize to target
            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(data)
            ratio = target_size / max(h, w)
            new_size = (int(w * ratio), int(h * ratio))
            pil_img = pil_img.resize(new_size, PILImage.LANCZOS)
            return np.array(pil_img)

    # Fallback: read lowest level
    data = levels[-1].asarray()
    if data.ndim == 3 and data.shape[2] >= 3:
        data = data[:, :, :3]
    return data


def get_thumbnail_from_openslide(slide, target_size=2000):
    """Get a thumbnail from an OpenSlide object."""
    w, h = slide.dimensions
    ratio = target_size / max(h, w)
    new_size = (int(w * ratio), int(h * ratio))
    print(f"  Getting thumbnail at {new_size[0]}x{new_size[1]}")
    thumb = slide.get_thumbnail(new_size)
    return np.array(thumb)[:, :, :3]  # Drop alpha if present


def create_tissue_mask(thumbnail):
    """
    Create a binary tissue mask from an H&E thumbnail.
    Uses Otsu thresholding in grayscale.
    """
    gray = color.rgb2gray(thumbnail)
    threshold = filters.threshold_otsu(gray)
    # Tissue is darker than background
    mask = gray < threshold
    # Clean up small holes and objects
    mask = morphology.remove_small_objects(mask, min_size=500)
    mask = morphology.remove_small_holes(mask, area_threshold=500)
    return mask


def extract_patch_coords(thumbnail_shape, tissue_mask, num_patches, patch_size_in_thumbnail,
                         min_tissue_fraction=MIN_TISSUE_FRACTION, seed=42):
    """
    Extract coordinates of tissue-rich patches from the tissue mask.
    Returns coordinates in thumbnail space.
    """
    rng = np.random.RandomState(seed)
    h, w = tissue_mask.shape
    ps = patch_size_in_thumbnail

    coords = []
    attempts = 0
    max_attempts = num_patches * 50

    while len(coords) < num_patches and attempts < max_attempts:
        y = rng.randint(0, max(1, h - ps))
        x = rng.randint(0, max(1, w - ps))

        patch_mask = tissue_mask[y:y+ps, x:x+ps]
        tissue_fraction = patch_mask.mean()

        if tissue_fraction >= min_tissue_fraction:
            coords.append((x, y))

        attempts += 1

    print(f"  Found {len(coords)} tissue patches (from {attempts} attempts)")
    return coords


def load_tifffile_level(tif, level=0):
    """Load an entire pyramid level from a tifffile into a numpy array (cached)."""
    if not hasattr(tif, '_level_cache'):
        tif._level_cache = {}
    if level not in tif._level_cache:
        print(f"  Loading pyramid level {level} into memory...")
        data = tif.series[0].levels[level].asarray()
        if data.ndim == 3 and data.shape[2] > 3:
            data = data[:, :, :3]
        tif._level_cache[level] = data
        print(f"  Loaded level {level}: shape={data.shape}, dtype={data.dtype}")
    return tif._level_cache[level]


def read_patch_tifffile(tif, x, y, patch_size, level=0):
    """Read a patch from a tifffile at a given pyramid level."""
    data = load_tifffile_level(tif, level)

    # tifffile stores as (H, W, C)
    full_h, full_w = data.shape[0], data.shape[1]

    # Clamp coordinates
    x = min(x, full_w - patch_size)
    y = min(y, full_h - patch_size)
    x = max(0, x)
    y = max(0, y)

    patch = data[y:y+patch_size, x:x+patch_size].copy()

    if patch.ndim == 3 and patch.shape[2] > 3:
        patch = patch[:, :, :3]

    return patch


def read_patch_openslide(slide, x, y, patch_size, level=0):
    """Read a patch from an OpenSlide object."""
    # OpenSlide read_region takes (x, y) at level 0 coordinates
    downsample = slide.level_downsamples[level]
    x_l0 = int(x * downsample)
    y_l0 = int(y * downsample)

    region = slide.read_region((x_l0, y_l0), level, (patch_size, patch_size))
    patch = np.array(region)[:, :, :3]  # Drop alpha
    return patch


# ============================================================================
# W1 DISTANCE (WASSERSTEIN) - Following ROSIE paper methodology
# ============================================================================

def compute_histogram(img, bins=256):
    """Compute a normalized 256-bin intensity histogram (grayscale)."""
    gray = color.rgb2gray(img)
    hist, bin_edges = np.histogram(gray, bins=bins, range=(0, 1), density=True)
    return hist, bin_edges


def compute_rgb_histograms(img, bins=256):
    """Compute per-channel RGB histograms."""
    hists = {}
    for i, ch_name in enumerate(['R', 'G', 'B']):
        hist, _ = np.histogram(img[:, :, i], bins=bins, range=(0, 255), density=True)
        hists[ch_name] = hist
    return hists


def compute_w1_distance(hist1, hist2):
    """Compute Wasserstein-1 distance between two histograms."""
    # Create bin centers
    bin_centers = np.arange(len(hist1))
    return wasserstein_distance(bin_centers, bin_centers, hist1, hist2)


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    start_time = time.time()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'patches_reference'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'patches_validation'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'patches_normalized'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'diagnostics'), exist_ok=True)

    print("=" * 70)
    print("MACENKO STAIN NORMALIZATION PIPELINE")
    print("Reference: Fine-tuning slide (QPTIFF)")
    print("Target:    Validation slide (NDPI)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # STEP 1: Open slides and get thumbnails
    # ------------------------------------------------------------------
    print("\n[STEP 1] Opening slides and generating thumbnails...")

    ref_tif = open_reference_slide(REFERENCE_SLIDE)
    ref_thumbnail = get_thumbnail_from_tifffile(ref_tif, target_size=2000)
    print(f"  Reference thumbnail shape: {ref_thumbnail.shape}")
    io.imsave(os.path.join(OUTPUT_DIR, 'diagnostics', 'ref_thumbnail.png'), ref_thumbnail)

    val_slide = open_validation_slide(VALIDATION_SLIDE)
    val_thumbnail = get_thumbnail_from_openslide(val_slide, target_size=2000)
    print(f"  Validation thumbnail shape: {val_thumbnail.shape}")
    io.imsave(os.path.join(OUTPUT_DIR, 'diagnostics', 'val_thumbnail.png'), val_thumbnail)

    # ------------------------------------------------------------------
    # STEP 2: Create tissue masks
    # ------------------------------------------------------------------
    print("\n[STEP 2] Creating tissue masks...")

    ref_mask = create_tissue_mask(ref_thumbnail)
    val_mask = create_tissue_mask(val_thumbnail)

    ref_tissue_pct = ref_mask.mean() * 100
    val_tissue_pct = val_mask.mean() * 100
    print(f"  Reference tissue coverage: {ref_tissue_pct:.1f}%")
    print(f"  Validation tissue coverage: {val_tissue_pct:.1f}%")

    # Save tissue masks
    io.imsave(os.path.join(OUTPUT_DIR, 'diagnostics', 'ref_tissue_mask.png'),
              (ref_mask * 255).astype(np.uint8))
    io.imsave(os.path.join(OUTPUT_DIR, 'diagnostics', 'val_tissue_mask.png'),
              (val_mask * 255).astype(np.uint8))

    # ------------------------------------------------------------------
    # STEP 3: Extract reference patches for fitting normalizer
    # ------------------------------------------------------------------
    print("\n[STEP 3] Extracting reference patches for Macenko fitting...")

    # Determine scale factor between thumbnail and full resolution
    ref_base_shape = ref_tif.series[0].levels[0].shape  # (H, W, C)
    ref_scale_y = ref_base_shape[0] / ref_thumbnail.shape[0]
    ref_scale_x = ref_base_shape[1] / ref_thumbnail.shape[1]

    # We'll read from a mid-level pyramid for efficiency
    # Find a level close to ~0.5 µm/px (the reference is already at ~0.5 µm/px)
    ref_read_level = 0
    ref_levels = ref_tif.series[0].levels
    for i, lvl in enumerate(ref_levels):
        if max(lvl.shape[0], lvl.shape[1]) < 20000:
            ref_read_level = i
            break

    ref_read_shape = ref_levels[ref_read_level].shape
    ref_read_scale_y = ref_read_shape[0] / ref_thumbnail.shape[0]
    ref_read_scale_x = ref_read_shape[1] / ref_thumbnail.shape[1]

    print(f"  Reading reference patches from pyramid level {ref_read_level} "
          f"(shape: {ref_read_shape[1]}x{ref_read_shape[0]})")

    # Patch size in thumbnail coordinates
    thumb_patch_size = int(PATCH_SIZE / ref_read_scale_x)
    thumb_patch_size = max(thumb_patch_size, 10)

    ref_coords_thumb = extract_patch_coords(
        ref_thumbnail.shape[:2], ref_mask, NUM_REFERENCE_PATCHES,
        thumb_patch_size, min_tissue_fraction=MIN_TISSUE_FRACTION
    )

    # Read patches at full resolution
    ref_patches = []
    for i, (tx, ty) in enumerate(ref_coords_thumb):
        fx = int(tx * ref_read_scale_x)
        fy = int(ty * ref_read_scale_y)

        try:
            patch = read_patch_tifffile(ref_tif, fx, fy, PATCH_SIZE, level=ref_read_level)
            if patch.shape[0] >= PATCH_SIZE // 2 and patch.shape[1] >= PATCH_SIZE // 2:
                ref_patches.append(patch)
                if i < 5:  # Save first 5 for inspection
                    io.imsave(os.path.join(OUTPUT_DIR, 'patches_reference', f'ref_patch_{i:03d}.png'), patch)
        except Exception as e:
            print(f"  Warning: Could not read reference patch {i}: {e}")

    print(f"  Successfully extracted {len(ref_patches)} reference patches")

    # ------------------------------------------------------------------
    # STEP 4: Extract validation patches
    # ------------------------------------------------------------------
    print("\n[STEP 4] Extracting validation patches...")

    val_dims = val_slide.dimensions  # (W, H)
    val_scale_x = val_dims[0] / val_thumbnail.shape[1]
    val_scale_y = val_dims[1] / val_thumbnail.shape[0]

    # The validation slide is 40x (~0.22 µm/px), reference is ~20x (~0.50 µm/px)
    # Use level 1 of the NDPI (which is 2x downsampled ≈ 0.44 µm/px, close to reference)
    val_read_level = 1 if val_slide.level_count > 1 else 0
    val_downsample = val_slide.level_downsamples[val_read_level]
    val_level_dims = val_slide.level_dimensions[val_read_level]

    print(f"  Reading validation patches from level {val_read_level} "
          f"(dims: {val_level_dims[0]}x{val_level_dims[1]}, "
          f"downsample: {val_downsample:.1f}x)")

    val_read_scale_x = val_level_dims[0] / val_thumbnail.shape[1]
    val_read_scale_y = val_level_dims[1] / val_thumbnail.shape[0]

    thumb_patch_size_val = int(PATCH_SIZE / val_read_scale_x)
    thumb_patch_size_val = max(thumb_patch_size_val, 10)

    val_coords_thumb = extract_patch_coords(
        val_thumbnail.shape[:2], val_mask, NUM_SAMPLE_PATCHES,
        thumb_patch_size_val, min_tissue_fraction=MIN_TISSUE_FRACTION
    )

    val_patches = []
    for i, (tx, ty) in enumerate(val_coords_thumb):
        # Convert to level 0 coordinates for openslide
        fx_l0 = int(tx * val_scale_x)
        fy_l0 = int(ty * val_scale_y)

        try:
            region = val_slide.read_region((fx_l0, fy_l0), val_read_level, (PATCH_SIZE, PATCH_SIZE))
            patch = np.array(region)[:, :, :3]
            if patch.shape[0] >= PATCH_SIZE // 2 and patch.shape[1] >= PATCH_SIZE // 2:
                val_patches.append(patch)
                if i < 5:
                    io.imsave(os.path.join(OUTPUT_DIR, 'patches_validation', f'val_patch_{i:03d}.png'), patch)
        except Exception as e:
            print(f"  Warning: Could not read validation patch {i}: {e}")

    print(f"  Successfully extracted {len(val_patches)} validation patches")

    # ------------------------------------------------------------------
    # STEP 5: Compute W1 distance BEFORE normalization
    # ------------------------------------------------------------------
    print("\n[STEP 5] Computing W1 distance BEFORE normalization...")

    # Compute per-channel histograms for all reference patches (aggregate)
    ref_all_pixels = np.concatenate([p.reshape(-1, 3) for p in ref_patches[:NUM_SAMPLE_PATCHES]], axis=0)
    val_all_pixels = np.concatenate([p.reshape(-1, 3) for p in val_patches], axis=0)

    w1_before = {}
    for ch_idx, ch_name in enumerate(['R', 'G', 'B']):
        ref_hist, _ = np.histogram(ref_all_pixels[:, ch_idx], bins=256, range=(0, 255), density=True)
        val_hist, _ = np.histogram(val_all_pixels[:, ch_idx], bins=256, range=(0, 255), density=True)
        w1 = compute_w1_distance(ref_hist, val_hist)
        w1_before[ch_name] = w1

    # Also compute grayscale W1
    ref_gray = np.mean(ref_all_pixels, axis=1)
    val_gray = np.mean(val_all_pixels, axis=1)
    ref_gray_hist, _ = np.histogram(ref_gray, bins=256, range=(0, 255), density=True)
    val_gray_hist, _ = np.histogram(val_gray, bins=256, range=(0, 255), density=True)
    w1_before['Gray'] = compute_w1_distance(ref_gray_hist, val_gray_hist)

    print(f"  W1 distance BEFORE normalization:")
    print(f"    R: {w1_before['R']:.4f}")
    print(f"    G: {w1_before['G']:.4f}")
    print(f"    B: {w1_before['B']:.4f}")
    print(f"    Gray: {w1_before['Gray']:.4f}")
    print(f"    Mean RGB: {np.mean([w1_before['R'], w1_before['G'], w1_before['B']]):.4f}")

    # ------------------------------------------------------------------
    # STEP 6: Fit Macenko normalizer on reference
    # ------------------------------------------------------------------
    print("\n[STEP 6] Fitting Macenko normalizer on reference patches...")

    # Concatenate reference patches into a large composite image for fitting
    # Use a subset to keep memory manageable
    n_fit = min(20, len(ref_patches))
    composite_patches = ref_patches[:n_fit]

    # Stack patches vertically for fitting
    composite = np.concatenate(composite_patches, axis=0)
    print(f"  Composite reference image for fitting: {composite.shape}")

    normalizer = MacenkoNormalizer()
    normalizer.fit(composite)

    # Save normalizer parameters
    params = {
        'stain_matrix_H': normalizer.stain_matrix_ref[0].tolist(),
        'stain_matrix_E': normalizer.stain_matrix_ref[1].tolist(),
        'max_concentration_H': float(normalizer.max_concentrations_ref[0]),
        'max_concentration_E': float(normalizer.max_concentrations_ref[1]),
        'reference_slide': os.path.basename(REFERENCE_SLIDE),
        'od_threshold': OPTICAL_DENSITY_THRESHOLD,
        'alpha_percentile': ALPHA,
        'beta_percentile': BETA,
    }
    with open(os.path.join(OUTPUT_DIR, 'macenko_reference_params.json'), 'w') as f:
        json.dump(params, f, indent=2)
    print(f"  Saved normalizer parameters to macenko_reference_params.json")

    # ------------------------------------------------------------------
    # STEP 7: Normalize validation patches
    # ------------------------------------------------------------------
    print("\n[STEP 7] Normalizing validation patches...")

    normalized_patches = []
    for i, patch in enumerate(val_patches):
        try:
            normalized = normalizer.transform(patch)
            normalized_patches.append(normalized)
            if i < 5:
                # Save side-by-side comparison
                comparison = np.concatenate([patch, normalized], axis=1)
                io.imsave(
                    os.path.join(OUTPUT_DIR, 'patches_normalized', f'comparison_{i:03d}.png'),
                    comparison
                )
                io.imsave(
                    os.path.join(OUTPUT_DIR, 'patches_normalized', f'normalized_{i:03d}.png'),
                    normalized
                )
        except Exception as e:
            print(f"  Warning: Could not normalize patch {i}: {e}")

    print(f"  Successfully normalized {len(normalized_patches)} / {len(val_patches)} patches")

    # ------------------------------------------------------------------
    # STEP 8: Compute W1 distance AFTER normalization
    # ------------------------------------------------------------------
    print("\n[STEP 8] Computing W1 distance AFTER normalization...")

    if normalized_patches:
        norm_all_pixels = np.concatenate([p.reshape(-1, 3) for p in normalized_patches], axis=0)

        w1_after = {}
        for ch_idx, ch_name in enumerate(['R', 'G', 'B']):
            ref_hist, _ = np.histogram(ref_all_pixels[:, ch_idx], bins=256, range=(0, 255), density=True)
            norm_hist, _ = np.histogram(norm_all_pixels[:, ch_idx], bins=256, range=(0, 255), density=True)
            w1 = compute_w1_distance(ref_hist, norm_hist)
            w1_after[ch_name] = w1

        norm_gray = np.mean(norm_all_pixels, axis=1)
        norm_gray_hist, _ = np.histogram(norm_gray, bins=256, range=(0, 255), density=True)
        w1_after['Gray'] = compute_w1_distance(ref_gray_hist, norm_gray_hist)

        print(f"  W1 distance AFTER normalization:")
        print(f"    R: {w1_after['R']:.4f}")
        print(f"    G: {w1_after['G']:.4f}")
        print(f"    B: {w1_after['B']:.4f}")
        print(f"    Gray: {w1_after['Gray']:.4f}")
        print(f"    Mean RGB: {np.mean([w1_after['R'], w1_after['G'], w1_after['B']]):.4f}")

        # ------------------------------------------------------------------
        # STEP 9: Summary report
        # ------------------------------------------------------------------
        print("\n" + "=" * 70)
        print("NORMALIZATION SUMMARY REPORT")
        print("=" * 70)

        report = {
            'reference_slide': os.path.basename(REFERENCE_SLIDE),
            'validation_slide': os.path.basename(VALIDATION_SLIDE),
            'reference_resolution_umpp': 0.4988,
            'validation_resolution_umpp': 0.2212,
            'num_reference_patches': len(ref_patches),
            'num_validation_patches': len(val_patches),
            'num_normalized_patches': len(normalized_patches),
            'w1_before': w1_before,
            'w1_after': w1_after,
            'w1_reduction': {},
            'stain_matrix_reference': params,
        }

        print(f"\n  {'Channel':<10} {'W1 Before':>12} {'W1 After':>12} {'Reduction':>12}")
        print(f"  {'-'*46}")
        for ch in ['R', 'G', 'B', 'Gray']:
            before = w1_before[ch]
            after = w1_after[ch]
            reduction = ((before - after) / before) * 100 if before > 0 else 0
            report['w1_reduction'][ch] = reduction
            print(f"  {ch:<10} {before:>12.4f} {after:>12.4f} {reduction:>11.1f}%")

        mean_before = np.mean([w1_before[c] for c in ['R', 'G', 'B']])
        mean_after = np.mean([w1_after[c] for c in ['R', 'G', 'B']])
        mean_reduction = ((mean_before - mean_after) / mean_before) * 100 if mean_before > 0 else 0
        print(f"  {'Mean RGB':<10} {mean_before:>12.4f} {mean_after:>12.4f} {mean_reduction:>11.1f}%")

        report['w1_reduction']['Mean_RGB'] = mean_reduction

        with open(os.path.join(OUTPUT_DIR, 'normalization_report.json'), 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n  Report saved to: {os.path.join(OUTPUT_DIR, 'normalization_report.json')}")

    # ------------------------------------------------------------------
    # STEP 10: Save reference patch mosaic for visual QC
    # ------------------------------------------------------------------
    print("\n[STEP 10] Saving diagnostic visualizations...")

    # Create a mosaic of reference patches (5 columns)
    n_show = min(10, len(ref_patches))
    if n_show > 0:
        cols = 5
        rows = (n_show + cols - 1) // cols
        mosaic_h = rows * PATCH_SIZE
        mosaic_w = cols * PATCH_SIZE
        mosaic = np.ones((mosaic_h, mosaic_w, 3), dtype=np.uint8) * 255

        for i in range(n_show):
            r, c = i // cols, i % cols
            p = ref_patches[i]
            ph, pw = p.shape[:2]
            mosaic[r*PATCH_SIZE:r*PATCH_SIZE+ph, c*PATCH_SIZE:c*PATCH_SIZE+pw] = p

        io.imsave(os.path.join(OUTPUT_DIR, 'diagnostics', 'reference_patches_mosaic.png'), mosaic)

    # Create before/after comparison mosaic for validation
    n_show = min(5, len(val_patches), len(normalized_patches))
    if n_show > 0:
        comparison_h = n_show * PATCH_SIZE
        comparison_w = 3 * PATCH_SIZE  # original | normalized | reference
        comparison = np.ones((comparison_h, comparison_w, 3), dtype=np.uint8) * 255

        for i in range(n_show):
            vp = val_patches[i]
            np_ = normalized_patches[i]
            rp = ref_patches[i] if i < len(ref_patches) else ref_patches[0]

            ph, pw = min(vp.shape[0], PATCH_SIZE), min(vp.shape[1], PATCH_SIZE)
            comparison[i*PATCH_SIZE:i*PATCH_SIZE+ph, 0:pw] = vp[:ph, :pw]
            comparison[i*PATCH_SIZE:i*PATCH_SIZE+ph, PATCH_SIZE:PATCH_SIZE+pw] = np_[:ph, :pw]

            rph, rpw = min(rp.shape[0], PATCH_SIZE), min(rp.shape[1], PATCH_SIZE)
            comparison[i*PATCH_SIZE:i*PATCH_SIZE+rph, 2*PATCH_SIZE:2*PATCH_SIZE+rpw] = rp[:rph, :rpw]

        io.imsave(os.path.join(OUTPUT_DIR, 'diagnostics', 'before_after_comparison.png'), comparison)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"Pipeline completed in {elapsed:.1f} seconds")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'=' * 70}")

    # Cleanup
    ref_tif.close()
    val_slide.close()


if __name__ == '__main__':
    main()
