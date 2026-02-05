import pandas as pd
import os
import numpy as np
from PIL import Image
import multiprocessing
import boto3
import tifffile
import io
import cv2
import argparse
import imageio
import json
import sys
from scipy import ndimage
import zarr
from skimage.transform import resize

# Configuration Constants
# Replace these values with your actual paths when using the code
SEGMENTATION_BASE_PATH = "/path/to/segmentation/data"  # Base path for segmentation data
DATA_BASE_PATH = "/path/to/data"  # Base path for general data
RUNS_BASE_PATH = "/path/to/runs"  # Base path for storing run outputs
METADATA_BASE_PATH = "/path/to/metadata"  # Base path for metadata files

# Add HandE package to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))

from dataset import *

def get_segmask_from_acq_id(acq_id, version=1):
    """
    Retrieve segmentation mask for a given acquisition ID.
    
    Args:
        acq_id (str): Acquisition ID
        version (int): Version number of the segmentation
        
    Returns:
        np.array: Segmentation mask
    """
    seg_path = os.path.join(SEGMENTATION_BASE_PATH, f'{acq_id}.ome.zarr')
    assert os.path.exists(seg_path), f'Seg path does not exist {seg_path}'
    store = zarr.open(seg_path, mode='r')
    seg_image = np.array(store[0])
    return seg_image

def normalize_image(image, min_value, max_value):
    """
    Normalize image values to 0-255 range.
    
    Args:
        image (np.array): Input image
        min_value (float): Minimum value for normalization
        max_value (float): Maximum value for normalization
        
    Returns:
        np.array: Normalized image
    """
    return ((image - min_value)*255./(max_value - min_value)).astype(np.uint8)

# Padding size for cell patches
PAD_SIZE = 50

def generate_codex_from_acq_id(codex_acq_id, full_size):
    """
    Generate CODEX images from acquisition ID.
    
    Args:
        codex_acq_id (str): CODEX acquisition ID
        full_size (tuple): Full size of the output image (height, width)
        
    Returns:
        tuple: (true_image, pred_image) - Ground truth and predicted images
    """
    df_ = pred_df[pred_df['CODEX_ACQUISITION_ID'] == codex_acq_id].copy()
    true_df = pd.read_parquet(os.path.join(DATA_BASE_PATH, f'acq_id_exps_v7/{codex_acq_id}.pqt'))

    # Create cell IDs and set as index
    df_['CELL_ID'] = df_.apply(lambda row: f"{row['X']}_{row['Y']}", axis=1)
    df_.set_index('CELL_ID', inplace=True)
    true_df['CELL_ID'] = true_df.apply(lambda row: f"{row['X']}_{row['Y']}", axis=1)
    true_df.set_index('CELL_ID', inplace=True)

    # Initialize prediction columns
    pred_biomarkers = [f'pred_{x}' for x in all_biomarkers]
    true_df[pred_biomarkers] = 0.0
    true_df.loc[df_.index, pred_biomarkers] = df_[pred_biomarkers]

    # Sort and calculate dimensions
    true_df = true_df.sort_values(by=['Y', 'X'])
    width, height = (true_df['X'].max()+8)//8, (true_df['Y'].max()+8)//8
    full_height, full_width = full_size

    # Get background cells
    background_df = true_df[((true_df['X'] < PAD_SIZE) | (true_df['X'] > width - PAD_SIZE)) | 
                           ((true_df['Y'] < PAD_SIZE) | (true_df['Y'] > height - PAD_SIZE))]
    
    # Initialize output images
    true_image = np.zeros((len(all_biomarkers), height, width), dtype=np.uint8)
    pred_image = np.zeros((len(all_biomarkers), height, width), dtype=np.uint8)

    # Process each biomarker
    for idx, bm in enumerate(all_biomarkers):
        pred_bm = f'pred_{bm}'
        img_true = true_df[bm].values.reshape((height, width))
        img_pred = np.clip(true_df[f'pred_{bm}'].values.reshape((height, width)), 0, 1)
        
        # Calculate thresholds
        bg_threshold_true = background_df[bm].quantile(0.9)
        bg_threshold_pred = background_df[pred_bm].quantile(0.9)
        vmax_true = np.percentile(img_true, 99.9)
        vmax_pred = np.percentile(img_pred, 99.9)
        
        # Adjust thresholds if necessary
        if bg_threshold_true >= vmax_true:
            bg_threshold_true = 0
        if bg_threshold_pred >= vmax_pred:
            bg_threshold_pred = 0
            
        # Clip and normalize images
        img_true = np.clip(img_true, bg_threshold_true, vmax_true)
        img_pred = np.clip(img_pred, bg_threshold_pred, vmax_pred)
        true_image[idx] = normalize_image(img_true, bg_threshold_true, vmax_true)
        pred_image[idx] = normalize_image(img_pred, bg_threshold_pred, vmax_pred)

    return true_image, pred_image

def generate_exp_df(codex_acq_id):
    """
    Generate expression dataframe for a given CODEX acquisition ID.
    
    Args:
        codex_acq_id (str): CODEX acquisition ID
        
    Returns:
        list: List of expression vectors for each cell
    """
    seg_img = get_segmask_from_acq_id(codex_acq_id)
    
    # Generate or load images
    img_true, img_pred = generate_codex_from_acq_id(codex_acq_id, seg_img.shape)
    
    # Save generated images
    output_dir = os.path.join(RUNS_BASE_PATH, 'images_s4065_c001')
    os.makedirs(output_dir, exist_ok=True)
    tifffile.imwrite(os.path.join(output_dir, f'{codex_acq_id}_true.tif'), img_true)
    tifffile.imwrite(os.path.join(output_dir, f'{codex_acq_id}_pred.tif'), img_pred)

    # Initialize full-size images
    img_true_full = np.zeros((img_true.shape[0], seg_img.shape[0], seg_img.shape[1]), dtype=np.float32)
    img_pred_full = np.zeros((img_pred.shape[0], seg_img.shape[0], seg_img.shape[1]), dtype=np.float32)

    # Resize if necessary
    if img_true.shape[1:] != seg_img.shape:
        for i in range(img_true.shape[0]):
            img_true_full[i] = cv2.resize(img_true[i], seg_img.shape, interpolation=cv2.INTER_NEAREST)/255.0
            img_pred_full[i] = cv2.resize(img_pred[i], seg_img.shape, interpolation=cv2.INTER_NEAREST)/255.0
    
    # Process each cell
    df = []
    unique_values = np.unique(seg_img[seg_img > 0])

    for cell_id in unique_values:
        y, x = np.unravel_index((seg_img == cell_id).argmax(), seg_img.shape)
        y_min, y_max = y.min(), y.max()
        x_min, x_max = x.min(), x.max()

        # Calculate patch boundaries
        b = max(0, y_min - PAD_SIZE)
        t = min(seg_img.shape[0], y_max + PAD_SIZE)
        l = max(0, x_min - PAD_SIZE)
        r = min(seg_img.shape[1], x_max + PAD_SIZE)

        # Extract patches
        seg_patch = seg_img[b:t, l:r]
        pred_patch = img_pred_full[:, b:t, l:r]
        true_patch = img_true_full[:, b:t, l:r]

        cell_mask = (seg_patch == cell_id)

        if cell_mask.sum() == 0:
            print(f'Cell {cell_id} has no pixels')
            continue

        # Calculate expression vectors
        pred_exp_vec = np.mean(pred_patch, axis=(1, 2), where=cell_mask)
        true_exp_vec = np.mean(true_patch, axis=(1, 2), where=cell_mask)
        ref_exp_vec = true_exp_vec

        df.append([cell_id, codex_acq_id] + list(pred_exp_vec) + list(true_exp_vec) + list(ref_exp_vec))
    
    return df

if __name__ == '__main__':
    # Load metadata
    pair_df = pd.read_parquet(os.path.join(METADATA_BASE_PATH, 'he_codex_pairs_v9_s4065_c001.pqt'))
    uuid_df = pd.read_parquet(os.path.join(METADATA_BASE_PATH, 'study_uuid_id_map_v3.pqt'))
    uuid_dict = uuid_df.set_index('STUDY_ID').to_dict(orient='index')
    version_df = pd.read_parquet(os.path.join(METADATA_BASE_PATH, 'version_df.pqt'))
    
    metadata_dict = pair_df[['CODEX_ACQUISITION_ID', 'HE_ACQUISITION_ID', 'HE_REGION_UUID']].groupby('CODEX_ACQUISITION_ID').first().to_dict(orient='index')
    
    study_id = 4065
    study_uuid = uuid_dict[study_id]['STUDY_UUID']
    
    # Load reference and prediction data
    ref_df = pd.read_parquet(os.path.join(DATA_BASE_PATH, f'studies/{study_id}_exp_df.pqt'))
    run_name = '0219_all_patch_v6_top50_whitespace_df_weighted_ssim'
    run_path = os.path.join(RUNS_BASE_PATH, run_name)
    pred_df = pd.read_parquet(os.path.join(run_path, 'predictions_s4065_c001_fixed_masking_all.pqt'))
    
    # Process test data
    test_df = pair_df[(pair_df['STUDY_ID'] == study_id)]
    all_biomarkers = [x.replace('pred_', '') for x in pred_df.columns if 'pred' in x]
    pred_cols = [f'pred_{x}' for x in all_biomarkers]
    ref_cols = [f'ref_{x}' for x in all_biomarkers]
    
    # Process all acquisitions in parallel
    from pqdm.processes import pqdm
    results = pqdm(test_df['CODEX_ACQUISITION_ID'].unique(), generate_exp_df, n_jobs=16)
    
    # Create final dataframe
    results = [result for result in results if isinstance(result, list)]
    all_results = np.concatenate(results)
    df_ = pd.DataFrame(all_results, columns=['CELL_ID', 'ACQUISITION_ID']+pred_cols+all_biomarkers+ref_cols)
    
    # Convert and clean data types
    df_[pred_cols] = df_[pred_cols].astype(np.float32).fillna(0)
    df_[all_biomarkers] = df_[all_biomarkers].astype(np.float32).fillna(0)
    df_[ref_cols] = df_[ref_cols].astype(np.float32).fillna(0)