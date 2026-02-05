"""
Evaluation script for H&E to multiplex protein prediction model.

This script runs inference on H&E images to predict protein expression levels
and outputs TIFF files containing the predictions.

Usage:
    python evaluate.py --input_dir /path/to/he/images --output_dir /path/to/output --model_path /path/to/model.pth
"""
from scipy.ndimage import gaussian_filter
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader
import tifffile
import argparse
from typing import Tuple, Optional
from pathlib import Path
import pdb
from PIL import Image
import cv2
from scipy.signal import convolve2d
from skimage.morphology import dilation, disk

# Configuration constants
BATCH_SIZE = 32
NUM_WORKERS = 8
PATCH_SIZE = 128
WHITE_THRESHOLD = 220

def pad_patch(patch: np.ndarray, 
             original_size: Tuple[int, int], 
             x_center: int, 
             y_center: int, 
             patch_size: int = PATCH_SIZE) -> np.ndarray:
    """
    Pads the given patch if its size is less than patch_size x patch_size pixels.
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

def box_blur(image_array, window_size=8):
    """
    Apply box blur to an image array using convolution.
    
    Args:
        image_array: 2D numpy array representing the image
        window_size: Size of the blur kernel
        
    Returns:
        2D numpy array with blur applied
    """
    # Create a kernel for box blur
    kernel = np.ones((window_size, window_size)) / (window_size ** 2)
    
    # Apply convolution using the kernel
    blurred_array = convolve2d(image_array, kernel, mode='same')
    
    return blurred_array

def normalize_image(image, min_value, max_value):
    """
    Normalize image values to 0-255 range.
    
    Args:
        image: Input image array
        min_value: Minimum value for normalization
        max_value: Maximum value for normalization
        
    Returns:
        Normalized image as uint8
    """
    return ((image - min_value)*255./(max_value - min_value)).astype(np.uint8)

def get_model(num_outputs: int) -> nn.Module:
    """Creates and returns the model architecture."""
    model = models.convnext_small(weights='IMAGENET1K_V1')
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_outputs)
    return model

class ImageDataset(Dataset):
    """Dataset class for loading H&E image patches from either ZARR or PNG."""
    def __init__(self, image_path: str, transform: Optional[dict] = None, stride_size: int = 8, exclude_background: bool = True):
        self.image_path = image_path
        self.transform = transform
        self.patch_size = PATCH_SIZE
        self.ps = self.patch_size//2
        self.stride_size = stride_size
        self.center_half = max(1, stride_size // 2)
        
        # Load image based on file type
        if image_path.endswith('.zarr'):
            self.he_zarr = [list(Reader(parse_url(image_path+f'/{i}', mode="r"))())[0].data[0].compute() 
                           for i in range(3)]
            height, width = self.he_zarr[0].shape
            
            # Crop to center 100x100 pixels for ZARR images
            center_y, center_x = height // 2, width // 2
            crop_size = 100
            half_crop = crop_size // 2
            
            # Calculate crop boundaries
            y_start = max(0, center_y - half_crop)
            y_end = min(height, center_y + half_crop)
            x_start = max(0, center_x - half_crop)
            x_end = min(width, center_x + half_crop)
            
            # Apply cropping to all channels
            self.he_zarr = [channel[y_start:y_end, x_start:x_end] for channel in self.he_zarr]
            
        else:  # Handle PNG/JPG
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Split channels and convert to same format as ZARR
            self.he_zarr = [img[:,:,i] for i in range(3)]
        
        # Create grid of patch centers
        height, width = self.he_zarr[0].shape
        self.coords = []
        for y in range(0, height, stride_size):
            for x in range(0, width, stride_size):
                # x and y are the center of the patch
                if exclude_background:
                    # Calculate boundaries for center region
                    x_start = max(0, x - self.center_half)
                    x_end = min(width, x + self.center_half)
                    y_start = max(0, y - self.center_half)
                    y_end = min(height, y + self.center_half)
                    
                    # Calculate average pixel value across all channels in center region
                    center_region = np.mean([channel[y_start:y_end, x_start:x_end] 
                                          for channel in self.he_zarr], axis=0)
                    avg_value = np.mean(center_region)
                    
                    # Only append coordinates if average value is below threshold
                    if avg_value < WHITE_THRESHOLD:
                        self.coords.append((x, y))
                else:
                    # Include all coordinates without filtering
                    self.coords.append((x, y))

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """Get a single patch from the image."""
        X, Y = self.coords[idx]
        
        # Extract patch
        b = np.clip(Y-self.ps, 0, self.he_zarr[0].shape[0])
        t = np.clip(Y+self.ps, 0, self.he_zarr[0].shape[0])
        l = np.clip(X-self.ps, 0, self.he_zarr[0].shape[1])
        r = np.clip(X+self.ps, 0, self.he_zarr[0].shape[1])
        
        he_patch = np.array([channel[b:t, l:r] for channel in self.he_zarr]).transpose(1, 2, 0)
        he_patch = pad_patch(he_patch, self.he_zarr[0].shape, X, Y)
        
        # Apply transforms
        if isinstance(self.transform, dict):
            he_patch_pt = self.transform['all_channels'](he_patch)
            patch = self.transform['image_only'](he_patch_pt)
        else:
            patch = self.transform(he_patch)
            
        return patch, X, Y

def create_tissue_mask(he_zarr, dilation_radius=9):
    """
    Create a tissue mask by identifying non-white areas and dilating.
    
    Args:
        he_zarr: List of H&E image channels
        dilation_radius: Radius for dilation operation
        
    Returns:
        Binary mask of tissue regions
    """
    # Convert to numpy array if not already
    he_array = np.array(he_zarr)
    
    # Create mask where any channel is below threshold (not white)
    tissue_mask = np.any((he_array < 220), axis=0)
    
    # Dilate the mask to include surrounding areas
    dilated_mask = dilation(tissue_mask, disk(dilation_radius))
    
    return dilated_mask

def postprocess_predictions(predictions, tissue_mask, apply_border_threshold=False, bg_percentile=90, max_percentile=99.9):
    """
    Postprocess model predictions with background thresholding, normalization and masking.
    
    Args:
        predictions: Raw model predictions (num_channels, height, width)
        tissue_mask: Binary mask of tissue regions
        apply_border_threshold: Whether to apply background thresholding from border regions
        bg_percentile: Percentile to use for background threshold
        max_percentile: Percentile to use for maximum value
        
    Returns:
        Processed predictions as uint8 array
    """
    processed = np.zeros_like(predictions, dtype=np.uint8)
    
    # Create a border region for background estimation
    height, width = tissue_mask.shape
    pad = 50
    border_mask = np.zeros_like(tissue_mask, dtype=bool)
    border_mask[:pad, :] = True
    border_mask[-pad:, :] = True
    border_mask[:, :pad] = True
    border_mask[:, -pad:] = True
    
    for i in range(predictions.shape[0]):
        channel = predictions[i]
        
        # Get background threshold based on apply_border_threshold flag
        if apply_border_threshold:
            # Use border regions for background threshold
            bg_values = channel[border_mask]
            if len(bg_values) > 0:
                bg_threshold = np.percentile(bg_values, bg_percentile)
            else:
                bg_threshold = 0
        else:
            # Use whole image for background threshold
            bg_threshold = np.percentile(channel, bg_percentile)
            
        # Get maximum value for normalization
        vmax = np.percentile(channel, max_percentile)
        
        # If max value is less than background threshold, reset background
        if vmax <= bg_threshold:
            bg_threshold = 0
            
        # Clip values between background and max
        clipped = np.clip(channel, bg_threshold, vmax)
        
        # Normalize to 0-255 range
        normalized = normalize_image(clipped, bg_threshold, vmax)
        
        blurred = box_blur(normalized)
        
        # Apply tissue mask
        masked = blurred# * tissue_mask
        
        processed[i] = masked
        
    return processed

def process_image(model: nn.Module,
                 image_path: str,
                 output_path: str,
                 device: torch.device,
                 num_channels: int,
                 stride_size: int,
                 exclude_background: bool = True,
                 apply_border_threshold: bool = False,
                 smooth_sigma: float = 1.0,
                 postprocess_image: bool = True):
    """Process a single H&E image (ZARR or PNG) and save predictions as TIFF."""
    
    transform = {
        'all_channels': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, antialias=True),
        ]),
        'image_only': transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    }
    
    # Use smaller stride for overlapping patches
    overlap_stride = max(1, stride_size // 2)  # More overlap to reduce artifacts
    
    dataset = ImageDataset(image_path, transform=transform, stride_size=overlap_stride, 
                          exclude_background=exclude_background)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    # Get image dimensions from dataset
    height, width = dataset.he_zarr[0].shape
    
    # Initialize output array and weight array for weighted blending
    raw_output = np.zeros((num_channels, height, width), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)
    
    # Create tissue mask for postprocessing
    tissue_mask = create_tissue_mask(dataset.he_zarr)
    
    # Create a Gaussian weight kernel for smoother blending
    kernel_size = stride_size * 2
    y, x = np.mgrid[0:kernel_size, 0:kernel_size]
    center = kernel_size // 2
    weight_kernel = np.exp(-((x - center)**2 + (y - center)**2) / (2 * (kernel_size/4)**2))
    
    model.eval()
    with torch.no_grad():
        for patches, X, Y in tqdm(dataloader):
            patches = patches.to(device)
            predictions = model(patches).cpu().numpy()
            
            # Fill in predictions with weighted blending
            for pred, x, y in zip(predictions, X, Y):
                # Define patch region with kernel_size
                half_size = kernel_size // 2
                t = np.clip(y - half_size, 0, height)
                b = np.clip(y + half_size, 0, height)
                l = np.clip(x - half_size, 0, width)
                r = np.clip(x + half_size, 0, width)
                
                # Get the portion of the weight kernel that fits
                kernel_h, kernel_w = b-t, r-l
                weight = weight_kernel[:kernel_h, :kernel_w]
                
                # Add weighted prediction values to the output array
                for c in range(num_channels):
                    raw_output[c, t:b, l:r] += pred[c] * weight
                
                # Add weights to the weight map
                weight_map[t:b, l:r] += weight
    
    # Normalize by weights
    # Avoid division by zero
    weight_map = np.maximum(weight_map, 1e-8)
    for c in range(num_channels):
        raw_output[c] /= weight_map
    
    # Apply postprocessing if enabled
    if postprocess_image:
        processed_output = postprocess_predictions(raw_output, tissue_mask, apply_border_threshold=apply_border_threshold)
        
        # Apply Gaussian smoothing
        if smooth_sigma > 0:
            processed_output = gaussian_filter(processed_output, sigma=smooth_sigma)
        
        # Save processed output as TIFF
        tifffile.imwrite(output_path, processed_output)
    else:
        # Save raw output as TIFF
        tifffile.imwrite(output_path, raw_output)

def main():
    parser = argparse.ArgumentParser(description='Run inference on H&E images')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing H&E zarr files/PNG images, or path to a single PNG image')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save TIFF outputs')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--stride_size', type=int, choices=[1, 2, 4, 8], default=8, 
                      help='Stride size for prediction (1=full resolution, 8=most downsampled)')
    parser.add_argument('--exclude_background', action='store_true', default=False,
                      help='Whether to exclude white background patches (default: False)')
    parser.add_argument('--apply_border_threshold', action='store_true', default=False,
                      help='Whether to use the border regions for background thresholding in postprocessing (default: False)')
    parser.add_argument('--smooth_sigma', type=float, default=1.0,
                      help='Sigma for Gaussian smoothing (0 to disable)')
    parser.add_argument('--postprocess_image', action='store_true', default=False,
                      help='Whether to apply postprocessing to the predictions (default: False)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    num_channels = 50
    model = get_model(num_outputs=num_channels)
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs")
    model = nn.DataParallel(model)
    # pdb.set_trace()
    model.load_state_dict(torch.load(args.model_path)['model_state_dict'])
    model = model.to(device)
    
    # Check if input_dir is a file or directory
    if os.path.isfile(args.input_dir):
        # Process single image
        if args.input_dir.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = args.input_dir
            output_name = os.path.splitext(os.path.basename(args.input_dir))[0]
            output_path = os.path.join(args.output_dir, f'{output_name}_ROSIE.tiff')
            process_image(model, image_path, output_path, device, num_channels, args.stride_size, 
                         args.exclude_background, args.apply_border_threshold, args.smooth_sigma, args.postprocess_image)
        else:
            print(f"Skipping {args.input_dir} - unsupported file type")
    else:
        # Process directory
        for img_name in tqdm(os.listdir(args.input_dir)):
            if img_name.endswith('.ome.zarr'):
                image_path = os.path.join(args.input_dir, img_name)
                output_name = os.path.dirname(img_name)
            elif img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(args.input_dir, img_name)
                output_name = os.path.splitext(img_name)[0]
            else:
                print(f"Skipping {img_name} - unsupported file type")
                continue
                
            if not os.path.exists(image_path):
                print(f"Skipping {img_name} because it does not exist")
                continue
                
            output_path = os.path.join(args.output_dir, f'{output_name}_ROSIE.tiff')
            process_image(model, image_path, output_path, device, num_channels, args.stride_size, 
                         args.exclude_background, args.apply_border_threshold, args.smooth_sigma, args.postprocess_image)
            print(f"Processed {img_name} and saved to {output_path}")

if __name__ == '__main__':
    main()
