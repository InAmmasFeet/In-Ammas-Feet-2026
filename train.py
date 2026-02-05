"""
Training script for H&E to multiplex protein prediction model.

This script trains a deep learning model to predict protein expression levels
from H&E images. It supports both training and evaluation modes.

Required directory structure:
ROOT_DIR/
    ├── data/                     # Contains training data
    │   └── cell_measurements.pqt # Parquet file with cell measurements
    ├── images/                   # H&E image data
    │   └── {uuid}/image.ome.zarr # Zarr formatted image files  
    ├── metadata/                 # Metadata files
    │   └── metadata_dict.pkl     # Dictionary with experiment metadata
    └── runs/                     # Training run outputs
"""

import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import wandb
from typing import Tuple, List, Dict, Optional
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

# Configure torch multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Configuration constants
ROOT_DIR = "/path/to/project/root"  # Base directory for project
DATA_FILE = os.path.join(ROOT_DIR, "data/cell_measurements.pqt")
METADATA_FILE = os.path.join(ROOT_DIR, "metadata/metadata_dict.pkl")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# Model training constants
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
EVAL_INTERVAL = 3000
PATIENCE = 75000
NUM_WORKERS = 16
PATCH_SIZE = 128

# Dataset splits for train/val/test
DATASET_SPLITS = {
    'train': [
        # TODO: Add training dataset splits
    ],
    'val': [
        # TODO: Add validation dataset splits
    ],
    'test': [
        # TODO: Add test dataset splits
    ]
}

def pad_patch(patch: np.ndarray, 
             original_size: Tuple[int, int], 
             x_center: int, 
             y_center: int, 
             patch_size: int = PATCH_SIZE) -> np.ndarray:
    """
    Pads the given patch if its size is less than patch_size x patch_size pixels.

    Args:
        patch: NumPy array representing the patch image
        original_size: Tuple of (width, height) of the original image
        x_center: X coordinate of the center of the patch in the original image
        y_center: Y coordinate of the center of the patch in the original image
        patch_size: The target size of the patch

    Returns:
        Padded patch as a NumPy array
    """
    original_height, original_width = original_size
    current_height, current_width = patch.shape[:2]
    
    if current_height == patch_size and current_width == patch_size:
        return patch
        
    # Calculate padding needed
    pad_left = max(patch_size // 2 - x_center, 0)
    pad_right = max(x_center + patch_size // 2 - original_width, 0)
    pad_top = max(patch_size // 2 - y_center, 0)
    pad_bottom = max(y_center + patch_size // 2 - original_height, 0)

    # Apply padding
    pad_shape = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)) if patch.ndim == 3 else ((pad_top, pad_bottom), (pad_left, pad_right))
    padded_patch = np.pad(patch, pad_shape, mode='constant', constant_values=0)

    # Ensure the patch is exactly patch_size x patch_size
    padded_patch = padded_patch[:patch_size, :patch_size]

    return padded_patch

def masked_mse_loss(pred: torch.Tensor, 
                   target: torch.Tensor, 
                   mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean squared error loss with a mask.

    Args:
        pred: Predicted tensor
        target: Target tensor
        mask: Mask tensor with 1s for elements to include and 0s to exclude

    Returns:
        Loss value
    """
    mask = mask.bool()
    masked_pred = torch.masked_select(pred, mask)
    masked_target = torch.masked_select(target, mask)
    return F.mse_loss(masked_pred, masked_target, reduction='mean')

def get_model(num_outputs: Optional[int] = None, 
             use_context: bool = False, 
             use_mask: bool = False) -> nn.Module:
    """
    Creates and returns the model architecture.

    Args:
        num_outputs: Number of output features to predict
        use_context: Whether to use contextual features
        use_mask: Whether to use masking in the model

    Returns:
        PyTorch model instance
    """
    model = models.convnext_small(weights='IMAGENET1K_V1')
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_outputs)
    return model

class ImageDataset(Dataset):
    """
    Dataset class for loading H&E image patches and their corresponding protein expression values.
    
    Args:
        data_df: DataFrame containing cell measurements
        root_dir: Root directory containing image data
        is_test: Whether this is a test dataset
        use_mask: Whether to use cell segmentation masks
        transform: Transforms to apply to images
        metadata_dict: Dictionary containing experiment metadata
        test_acq_ids: List of acquisition IDs to use for testing
        subset: Subset of coverslip IDs to use
        pred_only: Whether to only generate predictions (no ground truth)
    """
    def __init__(self,
                data_df: pd.DataFrame,
                root_dir: str,
                is_test: bool = False,
                use_mask: bool = False,
                transform: Optional[Dict] = None,
                metadata_dict: Optional[Dict] = None,
                test_acq_ids: Optional[List[str]] = None,
                subset: Optional[List[str]] = None,
                pred_only: bool = False):
        
        self.df = data_df
        self.root_dir = root_dir
        self.transform = transform
        self.patch_size = PATCH_SIZE
        self.ps = self.patch_size//2
        self.metadata_dict = metadata_dict
        self.use_mask = use_mask
        self.invalid_acq_ids = set()
        self.zarr_cache = {}
        self.is_test = is_test
        self.pred_only = pred_only

        self.all_biomarkers = self.metadata_dict['all_biomarkers']
        assert len(self.all_biomarkers) != 0, "No biomarker labels found"
        
        if subset is not None:
            self.df = self.df[self.df['HE_COVERSLIP_ID'].isin(subset)]
        if test_acq_ids is not None:
            self.df = self.df[self.df['CODEX_ACQUISITION_ID'].isin(test_acq_ids)]
            
        self.df.reset_index(inplace=True)
        self.acq_map = {i: x for i, x in enumerate(self.df['CODEX_ACQUISITION_ID'].unique())}
        self.acq_map.update({x: i for i, x in enumerate(self.df['CODEX_ACQUISITION_ID'].unique())})

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Optional[Tuple]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to get
            
        Returns:
            Tuple containing (image_patch, expression_values, mask, metadata)
            Returns None if the item is invalid
        """
        row = self.df.iloc[idx]
        seg_acq_id = row['CODEX_ACQUISITION_ID']
        he_region_uuid = self.metadata_dict[seg_acq_id]['HE_REGION_UUID']
        he_region_path = os.path.join(IMAGE_DIR, he_region_uuid, 'image.ome.zarr')
        
        if he_region_path in self.invalid_acq_ids or not os.path.exists(he_region_path):
            self.invalid_acq_ids.add(he_region_uuid)
            return None

        if self.use_mask and seg_acq_id in self.invalid_acq_ids:
            return None

        # Handle expression values
        if self.pred_only:
            exp_row = np.zeros(len(self.all_biomarkers))
            nan_mask = np.zeros(len(self.all_biomarkers))
            valid_mask = np.ones(len(self.all_biomarkers))
            exp_vec = exp_row.astype(np.float32)
        else:
            exp_row = row[self.all_biomarkers]
            nan_mask = exp_row.isnull().values
            exp_row[nan_mask] = 0
            valid_mask = ~nan_mask
            exp_vec = exp_row.values.astype(np.float32)

        # Get image patch coordinates
        X, Y = row['X'], row['Y']
        
        # Handle segmentation masks if needed
        if self.use_mask:
            seg_path = os.path.join(self.root_dir, 'codex_segs', f'{seg_acq_id}.ome.zarr')
            if not os.path.exists(seg_path):
                self.invalid_acq_ids.add(seg_acq_id)
                return None

        # Load H&E image
        if he_region_path in self.zarr_cache:
            he_zarr = self.zarr_cache[he_region_path]
        else:
            he_zarr = [list(Reader(parse_url(he_region_path+f'/{i}', mode="r"))())[0].data[0].compute() 
                      for i in range(3)]
            self.zarr_cache[he_region_path] = he_zarr

        # Extract patch
        b = np.clip(Y-self.ps, 0, he_zarr[0].shape[0])
        t = np.clip(Y+self.ps, 0, he_zarr[0].shape[0])
        l = np.clip(X-self.ps, 0, he_zarr[0].shape[1])
        r = np.clip(X+self.ps, 0, he_zarr[0].shape[1])
        
        he_patch = np.array([channel[b:t, l:r] for channel in he_zarr]).transpose(1, 2, 0)
        he_patch = pad_patch(he_patch, he_zarr[0].shape, X, Y)
        assert he_patch.shape == (128, 128, 3), f'H&E patch shape is {he_patch.shape}'

        # Apply transforms
        seed = np.random.randint(2**32)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if isinstance(self.transform, dict):
            he_patch_pt = self.transform['all_channels'](he_patch)
            patch = self.transform['image_only'](he_patch_pt)
        else:
            patch = self.transform(he_patch)
        
        assert patch.shape == (3, 224, 224), f'Patch shape is {patch.shape}'

def evaluate(model: nn.Module,
            data_loader: DataLoader,
            device: torch.device,
            run_dir: str,
            step: int,
            bm_labels: List[str],
            acq_dict: Optional[Dict] = None,
            save_predictions: bool = False,
            pred_biomarkers: Optional[List[str]] = None) -> Optional[Tuple[float, float]]:
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The model to evaluate
        data_loader: DataLoader for the evaluation dataset
        device: Device to run evaluation on
        run_dir: Directory to save results
        step: Current training step
        bm_labels: List of biomarker labels
        acq_dict: Dictionary mapping acquisition IDs to metadata
        save_predictions: Whether to save predictions to disk
        pred_biomarkers: List of biomarkers to predict (if different from bm_labels)
        
    Returns:
        None
    """
    model.eval()
    if pred_biomarkers is None:
        pred_biomarkers = bm_labels
        
    try:
        acq_map = data_loader.dataset.acq_map
    except:
        acq_map = data_loader.dataset.dataset.acq_map
    
    eval_dataset = []
    save_interval = 2000
    
    with torch.no_grad():
        for idx, (inputs, exp_vec, nan_mask, X, Y, indices) in tqdm(enumerate(data_loader)):
            inputs = inputs.to(device)
            outputs = model(inputs).detach().cpu().numpy()
            indices = indices.numpy()
            exp_vec = exp_vec.numpy()
            
            acq_ids = [acq_map[x] for x in indices]
            rows = [[a,b,c]+list(d)+list(e) for a,b,c,d,e in zip(X.numpy(), Y.numpy(), acq_ids, outputs, exp_vec)]
            eval_dataset.extend(rows)
            
            if save_predictions and idx % save_interval == 0:
                temp_df = pd.DataFrame(eval_dataset, 
                                     columns=['X', 'Y', 'CODEX_ACQUISITION_ID'] + 
                                             [f'pred_{x}' for x in pred_biomarkers] + 
                                             [f'gt_{x}' for x in bm_labels])
                
                if os.path.exists(f'{run_dir}/predictions_{step}_{idx}.pqt'):
                    temp_df.to_parquet(f'{run_dir}/predictions_{step}_{idx}.pqt', 
                                     engine='fastparquet', append=True)
                else:
                    temp_df.to_parquet(f'{run_dir}/predictions_{step}_{idx}.pqt')
                eval_dataset = []

    eval_dataset = pd.DataFrame(eval_dataset, 
                              columns=['X', 'Y', 'CODEX_ACQUISITION_ID'] + 
                                      [f'pred_{x}' for x in pred_biomarkers] + 
                                      [f'gt_{x}' for x in bm_labels])

    if save_predictions:
        eval_dataset.to_parquet(f'{run_dir}/predictions_{step}.pqt')
    
    return None

def main():
    """Main training and evaluation function."""
    
    # Initialize wandb for experiment tracking
    wandb.init(project='hande_to_codex', name='model_training')

    # Set up data transforms
    transform_train = {
        'all_channels': transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Resize(224, antialias=True),
            transforms.RandomRotation(degrees=(-10, 10)),
        ]),
        'image_only': transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }

    transform_eval = {
        'all_channels': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, antialias=True),
        ]),
        'image_only': transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }

    # Load metadata and data
    metadata_dict = pd.read_pickle(METADATA_FILE)
    data_df = pd.read_parquet(DATA_FILE)
    
    # Create datasets
    train_dataset = ImageDataset(
        data_df=data_df,
        root_dir=ROOT_DIR,
        transform=transform_train,
        metadata_dict=metadata_dict,
        subset=DATASET_SPLITS['train']
    )

    val_dataset = ImageDataset(
        data_df=data_df,
        root_dir=ROOT_DIR,
        transform=transform_eval,
        metadata_dict=metadata_dict,
        subset=DATASET_SPLITS['val'],
        is_test=True
    )

    # Create data loaders
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.default_collate(batch)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 4,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn
    )

    # Set up model and training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_outputs=len(metadata_dict['all_biomarkers']))
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    criterion = masked_mse_loss

    # Training loop
    step = 0
    best_val_score = 0
    steps_since_best_val_score = 0

    while True:
        model.train()
        for inputs, labels, mask, _ in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels, mask)
            
            if torch.isnan(loss):
                print("Warning: Loss is NaN, skipping batch")
                continue
                
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                wandb.log({'train_loss': loss.item()})

            # Validation
            if step % EVAL_INTERVAL == 0:
                val_r2, val_ssim = evaluate(
                    model, val_loader, device, 
                    os.path.join(ROOT_DIR, 'runs'), 
                    step, metadata_dict['all_biomarkers']
                )
                
                val_score = val_r2 + val_ssim
                wandb.log({
                    'val_r2': val_r2,
                    'val_ssim': val_ssim,
                    'val_score': val_score
                })

                # Save best model
                if val_score > best_val_score:
                    best_val_score = val_score
                    torch.save(model.state_dict(), os.path.join(ROOT_DIR, 'runs', 'best_model.pth'))
                    steps_since_best_val_score = 0
                else:
                    steps_since_best_val_score += EVAL_INTERVAL

                scheduler.step(val_score)

                # Early stopping check
                if steps_since_best_val_score >= PATIENCE:
                    print(f'Early stopping after {step} steps')
                    return

            step += 1

if __name__ == '__main__':
    main()