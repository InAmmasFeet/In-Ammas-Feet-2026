"""
Patchwork IoU: Alignment Quality Metric for H&E/mIF Tile Pairs
==============================================================
Scores and ranks tile pairs based on nuclei overlap between H&E and DAPI.

Features:
- LAB color space H&E nuclei extraction (hematoxylin-specific)
- Otsu-based DAPI nuclei extraction
- Local patchwork matching with small search window
- Ranking and statistics output

Usage:
    python patchwork_iou.py --images_dir /path/to/images --masks_dir /path/to/masks

Expected Input:
    - images_dir: Folder containing H&E tiles as PNG (BGR format)
    - masks_dir: Folder containing mIF tiles as TIF (DAPI in channel 0)

Output:
    - Console statistics (mean, median, percentiles)
    - CSV file with per-tile scores
    - Optional visualizations
"""

import os
import csv
import cv2
import numpy as np
import tifffile
import argparse
from tqdm import tqdm


# =============================================================================
# NUCLEI EXTRACTION FUNCTIONS
# =============================================================================

def get_dapi_nuclei_mask(dapi_image):
    """
    Extract nuclei binary mask from DAPI fluorescence image.
    
    Args:
        dapi_image: 2D numpy array (any dtype, typically uint8 or uint16)
    
    Returns:
        Binary mask (uint8, 0 or 255)
    """
    img_float = dapi_image.astype(np.float32)
    p_high = np.percentile(img_float, 99.5)
    
    # Handle low-signal images (sensor noise)
    if p_high < 10 or img_float.max() < 15:
        return np.zeros(dapi_image.shape, dtype=np.uint8)
    
    # Normalize to 0-255
    img_norm = (np.clip(img_float / (p_high + 1e-5), 0, 1) * 255).astype(np.uint8)
    
    # Blur and threshold
    blurred = cv2.GaussianBlur(img_norm, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def get_he_nuclei_mask(he_bgr):
    """
    Extract nuclei binary mask from H&E stained image using LAB color space.
    
    Hematoxylin (nuclei stain) appears blue-purple, which corresponds to:
    - Low b* channel (blue end of blue-yellow axis)
    - Lower L channel (darker than background)
    
    Args:
        he_bgr: 3-channel BGR image (as loaded by cv2.imread)
    
    Returns:
        Binary mask (uint8, 0 or 255)
    """
    # Convert BGR to LAB
    lab = cv2.cvtColor(he_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    
    # Invert b channel so blue becomes high values
    b_inv = 255 - b
    
    # Combine: nuclei are dark (low L) AND blue (high b_inv)
    # Weight: prioritize blue-ness (0.6) over darkness (0.4)
    nuclei_score = b_inv.astype(np.float32) * 0.6 + (255 - L).astype(np.float32) * 0.4
    
    # Normalize to 0-255
    score_min, score_max = nuclei_score.min(), nuclei_score.max()
    nuclei_score = ((nuclei_score - score_min) / (score_max - score_min + 1e-5) * 255).astype(np.uint8)
    
    # Blur and threshold
    blurred = cv2.GaussianBlur(nuclei_score, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask


# =============================================================================
# PATCHWORK IoU CALCULATION
# =============================================================================

def calculate_patchwork_iou(he_bgr, dapi, window_size=32, search_range=6):
    """
    Calculate Patchwork IoU between H&E and DAPI nuclei masks.
    
    The image is divided into windows. For each window in H&E, we search for
    the best-matching window in DAPI within a small search range. This allows
    for minor local misalignments while still measuring overall overlap quality.
    
    Args:
        he_bgr: H&E image as BGR numpy array
        dapi: DAPI image as 2D numpy array (any dtype)
        window_size: Size of each patchwork window (default 32)
        search_range: Pixels to search around each window (default 6)
    
    Returns:
        score: Mean IoU across all windows (0.0 to 1.0)
        reconstructed: DAPI mask with best-matching windows aligned to H&E
    """
    # Get binary masks
    he_mask = get_he_nuclei_mask(he_bgr)
    dapi_mask = get_dapi_nuclei_mask(dapi)
    
    h, w = he_mask.shape
    window_ious = []
    reconstructed = np.zeros_like(dapi_mask)
    
    for y in range(0, h, window_size):
        for x in range(0, w, window_size):
            # Extract H&E window
            he_win = he_mask[y:y+window_size, x:x+window_size]
            
            # Define search area in DAPI
            y_start = max(0, y - search_range)
            y_end = min(h, y + window_size + search_range)
            x_start = max(0, x - search_range)
            x_end = min(w, x + window_size + search_range)
            
            dapi_search = dapi_mask[y_start:y_end, x_start:x_end]
            
            # Skip empty windows
            if np.sum(he_win) == 0:
                continue
            
            # Find best match using template matching
            result = cv2.matchTemplate(
                dapi_search.astype(np.float32),
                he_win.astype(np.float32),
                cv2.TM_CCORR
            )
            _, _, _, max_loc = cv2.minMaxLoc(result)
            
            # Extract best-matching DAPI window
            best_y = y_start + max_loc[1]
            best_x = x_start + max_loc[0]
            dapi_win = dapi_mask[best_y:best_y+window_size, best_x:best_x+window_size]
            
            # Handle edge cases
            if dapi_win.shape != he_win.shape:
                dapi_win = np.zeros_like(he_win)
            
            # Calculate IoU for this window
            intersection = np.logical_and(he_win > 0, dapi_win > 0)
            union = np.logical_or(he_win > 0, dapi_win > 0)
            iou = np.sum(intersection) / (np.sum(union) + 1e-5)
            window_ious.append(iou)
            
            # Store in reconstructed image
            reconstructed[y:y+window_size, x:x+window_size] = dapi_win
    
    if not window_ious:
        return 0.0, reconstructed
    
    return float(np.mean(window_ious)), reconstructed


# =============================================================================
# MAIN EVALUATION PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Patchwork IoU: Score and rank H&E/mIF tile pairs by alignment quality"
    )
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory containing H&E tiles (PNG, BGR)")
    parser.add_argument("--masks_dir", type=str, required=True,
                        help="Directory containing mIF tiles (TIF, DAPI in channel 0)")
    parser.add_argument("--output_csv", type=str, default="patchwork_iou_scores.csv",
                        help="Output CSV file for scores")
    parser.add_argument("--viz_dir", type=str, default=None,
                        help="Optional: Directory to save visualizations")
    parser.add_argument("--num_viz", type=int, default=50,
                        help="Number of visualizations to generate")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit evaluation to N random samples")
    args = parser.parse_args()
    
    # Collect tile pairs
    image_files = sorted([f for f in os.listdir(args.images_dir) if f.endswith('.png')])
    
    pairs = []
    for img_file in image_files:
        mask_file = img_file.replace('.png', '.tif')
        mask_path = os.path.join(args.masks_dir, mask_file)
        if os.path.exists(mask_path):
            pairs.append((
                os.path.join(args.images_dir, img_file),
                mask_path
            ))
    
    print(f"Found {len(pairs)} tile pairs")
    
    # Subsample if requested
    if args.num_samples and len(pairs) > args.num_samples:
        import random
        random.seed(42)
        pairs = random.sample(pairs, args.num_samples)
        print(f"Subsampled to {len(pairs)} pairs")
    
    # Evaluate all pairs
    results = []
    print("Calculating Patchwork IoU scores...")
    
    for img_path, mask_path in tqdm(pairs):
        he = cv2.imread(img_path)
        mask = tifffile.imread(mask_path)
        
        # Handle different mask formats
        if mask.ndim == 3:
            dapi = mask[0]  # Assume DAPI is channel 0
        else:
            dapi = mask
        
        score, _ = calculate_patchwork_iou(he, dapi)
        results.append({
            'filename': os.path.basename(img_path),
            'img_path': img_path,
            'mask_path': mask_path,
            'score': score
        })
    
    # Sort by score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Calculate statistics
    scores = [r['score'] for r in results]
    stats = {
        'count': len(scores),
        'mean': np.mean(scores),
        'median': np.median(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores),
        'pct_above_30': sum(1 for s in scores if s > 0.30) / len(scores) * 100,
        'pct_above_40': sum(1 for s in scores if s > 0.40) / len(scores) * 100,
        'pct_above_50': sum(1 for s in scores if s > 0.50) / len(scores) * 100,
    }
    
    # Print statistics
    print("\n" + "=" * 50)
    print("PATCHWORK IoU STATISTICS")
    print("=" * 50)
    print(f"Total tiles evaluated:    {stats['count']}")
    print(f"Mean Score:               {stats['mean']:.4f}")
    print(f"Median Score:             {stats['median']:.4f}")
    print(f"Std Dev:                  {stats['std']:.4f}")
    print(f"Range:                    [{stats['min']:.4f}, {stats['max']:.4f}]")
    print("-" * 50)
    print(f"Usable (>0.30):           {stats['pct_above_30']:.1f}%")
    print(f"High Quality (>0.40):     {stats['pct_above_40']:.1f}%")
    print(f"Excellent (>0.50):        {stats['pct_above_50']:.1f}%")
    print("=" * 50)
    
    # Save CSV
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'filename', 'score'])
        for rank, r in enumerate(results):
            writer.writerow([rank, r['filename'], f"{r['score']:.4f}"])
    
    print(f"\nScores saved to: {args.output_csv}")
    
    # Generate visualizations if requested
    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)
        
        # Sample evenly across ranking
        step = max(1, len(results) // args.num_viz)
        viz_indices = list(range(0, len(results), step))[:args.num_viz]
        
        print(f"Generating {len(viz_indices)} visualizations...")
        
        for rank in tqdm(viz_indices):
            r = results[rank]
            he = cv2.imread(r['img_path'])
            mask = tifffile.imread(r['mask_path'])
            dapi = mask[0] if mask.ndim == 3 else mask
            
            score, dapi_aligned = calculate_patchwork_iou(he, dapi)
            he_mask = get_he_nuclei_mask(he)
            
            # Create visualization
            h, w = he.shape[:2]
            
            # Normalize DAPI for display
            dapi_viz = (np.clip(dapi.astype(float) / (np.percentile(dapi, 99.5) + 1), 0, 1) * 255).astype(np.uint8)
            dapi_bgr = cv2.cvtColor(dapi_viz, cv2.COLOR_GRAY2BGR)
            
            # Overlay: H&E nuclei (green) + DAPI aligned (red)
            overlay = np.zeros((h, w, 3), dtype=np.uint8)
            overlay[:, :, 1] = he_mask
            overlay[:, :, 2] = dapi_aligned
            
            # Grid lines
            grid_step = 64 if w >= 1024 else 32
            for i in range(0, w, grid_step):
                cv2.line(overlay, (i, 0), (i, h), (50, 50, 50), 1)
                cv2.line(overlay, (0, i), (w, i), (50, 50, 50), 1)
            
            # Combine
            combined = np.hstack([he, dapi_bgr, overlay])
            label = f"Rank:{rank} IoU:{r['score']:.3f}"
            cv2.putText(combined, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            out_path = os.path.join(args.viz_dir, f"rank_{rank:04d}_iou_{r['score']:.3f}.png")
            cv2.imwrite(out_path, combined)
        
        print(f"Visualizations saved to: {args.viz_dir}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
