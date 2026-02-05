"""
Reconstructs CODEX images from parquet files containing patch-level expression data.

This script processes H&E and CODEX image pairs, scanning through H&E images in 8px increments
and computing biomarker values for corresponding CODEX patches.

Required environment variables:
- AWS credentials for accessing S3 bucket containing CODEX region data
"""

from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import os
import numpy as np
import pandas as pd
import pdb

# Configuration constants
PATCH_SIZE = 128  # Size of image patches to process
PATCH_HALF = PATCH_SIZE // 2
CENTER_SIZE = 8   # Size of center region to analyze
CENTER_HALF = CENTER_SIZE // 2

# File path constants - Replace these with your actual paths
DATA_VERSION_PATH = "/path/to/data/version"  # Directory containing normalized parquet files
BIOMARKER_MAP_PATH = "/path/to/biomarker/channel/map.pqt"  # Biomarker to channel mapping file
CODEX_HE_PAIRS_PATH = "/path/to/codex_he_pairs.csv"  # CODEX-H&E pair mapping file
S3_BUCKET_PATH = "s3://your-bucket-name"  # S3 bucket containing CODEX region data

def reconstruct_codex_img_from_df(parquet_file_path):
    """
    Reconstructs a CODEX image from a parquet file containing patch-level expression data.
    
    Args:
        parquet_file_path (str): Path to the parquet file containing expression data
        
    Returns:
        None: Saves reconstructed image as .npy file
    """
    df = pd.read_parquet(parquet_file_path)
    codex_acq_id = df['CODEX_ACQUISITION_ID'].values[0]
    acq_df = bm_df[bm_df['ACQUISITION_ID'] == codex_acq_id]
    codex_region_uuid = metadata_dict[codex_acq_id]['CODEX_REGION_UUID']

    # Create channel mapping, excluding Empty and Blank channels
    channel_map = {
        int(acq_df[acq_df['BIOMARKER_NAME'] == x]['IMAGE_CHANNEL'].min()): x 
        for x in acq_df['BIOMARKER_NAME'].unique() 
        if x not in ['Empty', 'Blank', 'DAPI']
    }
    channel_map[4] = 'DAPI'
    channel_idxs = sorted(channel_map.keys())
    channel_bms = [channel_map[i] for i in channel_idxs]

    # Read CODEX image shape from S3
    try:
        zarr_path = f'{S3_BUCKET_PATH}/{codex_region_uuid}/image.ome.zarr/0/0/'
        codex_shape = list(Reader(parse_url(zarr_path, mode="r"))())[0].data[0]
    except Exception as e:
        print(f'Failed to read {codex_region_uuid}: {str(e)}')
        return None

    # Initialize empty array for reconstructed image
    pred_codex_img = np.zeros(
        (len(channel_bms), codex_shape.shape[0], codex_shape.shape[1]), 
        dtype=np.float32
    )

    # Fill in predicted values
    for i, row in df.iterrows():
        X, Y = row['X'], row['Y']
        codex_exp = row[channel_bms]
        pred_codex_img[:, Y:Y+CENTER_SIZE, X:X+CENTER_SIZE] = codex_exp.values.reshape(-1, 1, 1)

    # Save reconstructed image
    output_path = os.path.join(DATA_VERSION_PATH, f'{codex_acq_id}_reconstruct.npy')
    np.save(output_path, pred_codex_img)

if __name__ == '__main__':
    # Create output directory if it doesn't exist
    if not os.path.exists(DATA_VERSION_PATH):
        os.makedirs(DATA_VERSION_PATH)

    # Load required metadata
    bm_df = pd.read_parquet(BIOMARKER_MAP_PATH)
    pair_df = pd.read_csv(CODEX_HE_PAIRS_PATH)
    
    # Create metadata lookup dictionary
    metadata_dict = pair_df[
        ['CODEX_ACQUISITION_ID', 'HE_ACQUISITION_ID', 'CODEX_REGION_UUID', 'HE_REGION_UUID']
    ].set_index('CODEX_ACQUISITION_ID').to_dict(orient='index')

    # Example usage with a single file
    example_file = os.path.join(DATA_VERSION_PATH, "example_norm.pqt")
    reconstruct_codex_img_from_df(example_file)