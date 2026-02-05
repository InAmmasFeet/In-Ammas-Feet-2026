"""
Process H&E images to generate expression predictions and cell-level measurements.
This script handles the conversion of whole-slide predictions to cell-level expression values.

Required environment variables:
- AWS_ACCESS_KEY: AWS access key for data storage
- AWS_SECRET_ACCESS_KEY: AWS secret access key for data storage
"""

import os
import pandas as pd
import numpy as np
import sys
from typing import List, Tuple

# Add path to utility functions
sys.path.append('path/to/utils')  # Replace with your utils path
from utils import *
from dataset import *

# Configuration Constants
STUDY_UUID = 'your-study-uuid'  # UUID for the study being processed
BUCKET_NAME = 'your-bucket-name'  # AWS S3 bucket name
RUN_NAME = 'experiment-name'  # Name of the experimental run
PATCH_SIZE = 8  # Size of prediction patches
CELL_PADDING = 40  # Padding around cells for segmentation

# File paths - replace with your paths
DATA_PATHS = {
    'RUN_FOLDER': '/path/to/runs/folder',
    'EXPRESSION_DATA': '/path/to/expression/data.pqt',
    'XY_DATA': '/path/to/xy/data.pqt',
    'PREDICTIONS': '/path/to/predictions/final.npy'
}

# Biomarker labels
ALL_BIOMARKER_LABELS = [
    'DAPI', 'TIGIT', 'CD31', 'CD4', 'CD44', 'HLA-A', 'ECad', 'CD20', 'CD68', 
    'CD45RO', 'aSMA', 'CD66', 'IFNg', 'CD45', 'Podoplanin', 'Vimentin', 'CD11c', 
    'CD34', 'Keratin8/18', 'IDO1', 'CD8', 'Caveolin1', 'CD21', 'CD79a', 'HLA-DR', 
    'HLA-E', 'EpCAM', 'PCNA', 'Gal3', 'CD40', 'CollagenIV', 'CD14', 'FoxP3', 
    'PDL1', 'LAG3', 'VISTA', 'CD3e', 'GranzymeB', 'PD1', 'Ki67', 'ICOS', 'GATA3', 
    'CD163', 'BCL2', 'ATM', 'TP63', 'ERa', 'CD141', 'CD38', 'MPO', 'CD39', 'PanCK'
]

def construct_codex_from_hande(pair_id: int) -> np.ndarray:
    """
    Construct CODEX predictions from H&E image data.
    
    Args:
        pair_id: Identifier for the H&E-CODEX image pair
        
    Returns:
        np.ndarray: Predicted CODEX image
    """
    fn = f'{DATA_PATHS["RUN_FOLDER"]}/generated_imgs/{pair_id}.npy'
    if os.path.exists(fn):
        return np.load(fn, allow_pickle=True)
    
    rows = test_df[test_df['pair_id'] == pair_id]
    hande_acq_id = rows['hande_acq_id'].values[0]
    hande_img = get_img_from_acq_id(STUDY_UUID, hande_acq_id, bucket_name=BUCKET_NAME)
    
    pred_img = np.zeros((len(pred_cols), hande_img.shape[1], hande_img.shape[2]), 
                       dtype=np.float32)
    
    for idx, row in rows.iterrows():
        x = int(row['x_center'])
        y = int(row['y_center'])
        pred_img[:, y-PATCH_SIZE//2:y+PATCH_SIZE//2, 
                x-PATCH_SIZE//2:x+PATCH_SIZE//2] = row[pred_cols].values.reshape((len(pred_cols),1,1))
    
    np.save(fn, pred_img)
    return pred_img

def generate_exp_df(pair_id: int) -> List[List]:
    """
    Generate expression dataframe for a given image pair.
    
    Args:
        pair_id: Identifier for the H&E-CODEX image pair
        
    Returns:
        List[List]: Cell-level expression measurements
    """
    hande_acq_id = test_df[test_df['pair_id']==pair_id]['hande_acq_id'].values[0]
    codex_acq_id = test_df[test_df['pair_id']==pair_id]['codex_acq_id'].values[0]

    pred_img = construct_codex_from_hande(pair_id)
    codex_img = get_img_from_acq_id(STUDY_UUID, codex_acq_id)
    ref_img = codex_img

    df = []
    seg_img = get_segmask_from_acq_id(STUDY_UUID, codex_acq_id, 2, bucket_name=BUCKET_NAME)
    xy_rows = xy_df[xy_df['ACQUISITION_ID']==codex_acq_id]

    for _, rows in xy_rows.iterrows():
        cell_id = rows['CELL_ID']
        x, y = rows['X'], rows['Y']
        seg_patch = seg_img[y-CELL_PADDING:y+CELL_PADDING, x-CELL_PADDING:x+CELL_PADDING]
        pred_patch = pred_img[:, y-CELL_PADDING:y+CELL_PADDING, x-CELL_PADDING:x+CELL_PADDING]
        ref_patch = ref_img[:, y-CELL_PADDING:y+CELL_PADDING, x-CELL_PADDING:x+CELL_PADDING]
        cell_mask = (seg_patch == cell_id)
        
        pred_exp_vec = np.mean(pred_patch, axis=(1,2), where=cell_mask==1)
        ref_exp_vec = np.mean(ref_patch, axis=(1,2), where=cell_mask==1)
        df.append([cell_id, codex_acq_id] + list(pred_exp_vec) + list(ref_exp_vec))
    
    return df

def main():
    """Main execution function."""
    os.makedirs(f'{DATA_PATHS["RUN_FOLDER"]}/processed_imgs', exist_ok=True)

    # Load data
    data_df = pd.read_parquet(DATA_PATHS['EXPRESSION_DATA'])
    test_df = data_df[data_df['exp_name']=='TA-257'].reset_index()
    
    # Load predictions
    model_preds = np.load(DATA_PATHS['PREDICTIONS'])
    pred_cols = [f'pred_{bm}' for bm in ALL_BIOMARKER_LABELS]
    test_df[pred_cols] = model_preds[0]
    
    # Load cell position data
    xy_df = pd.read_parquet(DATA_PATHS['XY_DATA'])
    test_df_ = test_df[['pair_id', 'hande_acq_id', 'codex_acq_id']].drop_duplicates()

    # Process all pairs
    from pqdm.processes import pqdm
    results = pqdm(test_df_['pair_id'].unique(), generate_exp_df, n_jobs=16)
    
    # Combine and save results
    all_results = np.concatenate(results)
    df_ = pd.DataFrame(all_results, 
                      columns=['CELL_ID', 'ACQUISITION_ID'] + pred_cols + ALL_BIOMARKER_LABELS)
    df_[pred_cols] = df_[pred_cols].astype(np.float32).fillna(0)
    df_[ALL_BIOMARKER_LABELS] = df_[ALL_BIOMARKER_LABELS].astype(np.float32).fillna(0)
    df_.to_parquet(f'{DATA_PATHS["RUN_FOLDER"]}/pred_exp_df.pqt')

if __name__ == '__main__':
    main()