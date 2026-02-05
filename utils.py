"""
Utility functions for image analysis and machine learning tasks.
Contains functions for:
- Feature computation
- Metric calculations 
- Data loading and processing
- Visualization
- Model training helpers
"""

import os
import boto3
import io
import tifffile
import numpy as np
import scipy
import pandas as pd
from sklearn.decomposition import PCA
from collections import defaultdict, OrderedDict
from sklearn.svm import SVC
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, r2_score, balanced_accuracy_score, 
                           f1_score, confusion_matrix, accuracy_score, 
                           classification_report, average_precision_score,
                           precision_recall_fscore_support)
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from sklearn.dummy import DummyClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import *
import matplotlib
import imageio
import json
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.stats import spearmanr
import multiprocessing
import scipy.stats as stats
import sklearn
from sklearn import multioutput
import pickle5 as pickle
import pdb
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn as nn

# Configuration Constants
# Replace these with your actual values when using the code
DATA_BUCKET = "your-s3-bucket-name"
DATA_PATH = "/path/to/data/directory"
CACHE_PATH = "/path/to/cache/directory" 

# Environment variables needed:
# AWS_ACCESS_KEY_ID
# AWS_SECRET_ACCESS_KEY

def kl_divergence(true_dist, predicted_dist, epsilon=1e-10):
    """
    Calculate the Kullback-Leibler Divergence between two probability distributions,
    avoiding the 'invalid value encountered in log' error.

    Parameters:
    - true_dist: numpy array, the true probability distribution.
    - predicted_dist: numpy array, the predicted probability distribution.
    - epsilon: small value to prevent log(0).

    Returns:
    - KL Divergence as a float.
    """
    # Normalize distributions to ensure they sum to 1
    true_dist = true_dist / np.sum(true_dist)
    predicted_dist = predicted_dist / np.sum(predicted_dist)
    
    # Ensure predicted distribution has no zeros where true distribution is non-zero
    predicted_dist_safe = np.where(true_dist != 0, np.maximum(predicted_dist, epsilon), 0)
    
    # Calculate KL Divergence only for elements where true distribution is non-zero
    kl_div = np.sum(true_dist[true_dist != 0] * np.log(true_dist[true_dist != 0] / predicted_dist_safe[true_dist != 0]))
    
    return kl_div

def compute_morphometry_features(im_label, rprops=None):
    """
    Calculate morphometry features for each object in a labeled image.

    Parameters
    ----------
    im_label : array_like
        A labeled mask image wherein intensity of a pixel is the ID of the
        object it belongs to. Non-zero values are considered foreground objects.

    rprops : output of skimage.measure.regionprops, optional
        rprops = skimage.measure.regionprops(im_label). If not provided,
        will be computed internally.

    Returns
    -------
    fdata: pandas.DataFrame
        DataFrame containing morphometry features for each object:
        - Orientation.Orientation: Angle between horizontal axis and major ellipse axis
        - Size.Area: Number of pixels in object
        - Size.ConvexHullArea: Number of pixels in convex hull
        - Size.MajorAxisLength: Length of major axis of fitted ellipse
        - Size.MinorAxisLength: Length of minor axis of fitted ellipse
        - Size.Perimeter: Object perimeter length
        - Shape.Circularity: Measure of circular similarity
        - Shape.Eccentricity: Aspect ratio measure
        - Shape.EquivalentDiameter: Diameter of circle with same area
        - Shape.Extent: Ratio of area to bounding box
        - Shape.MinorMajorAxisRatio: Ratio of minor to major axis
        - Shape.Solidity: Ratio of pixels to convex hull
        - Shape.HuMoments1-7: Hu moment invariants
        - Shape.WeightedHuMoments1-7: Intensity-weighted Hu moments
    """
    # Define mapping between feature names and regionprops attributes
    featname_map = OrderedDict({
        'Orientation.Orientation': 'orientation',
        'Size.Area': 'area', 
        'Size.ConvexHullArea': 'convex_area',
        'Size.MajorAxisLength': 'major_axis_length',
        'Size.MinorAxisLength': 'minor_axis_length',
        'Size.Perimeter': 'perimeter',
        'Shape.Circularity': None,
        'Shape.Eccentricity': 'eccentricity',
        'Shape.EquivalentDiameter': 'equivalent_diameter',
        'Shape.Extent': 'extent',
        'Shape.MinorMajorAxisRatio': None,
        'Shape.Solidity': 'solidity',
    })

    # Add Hu moments feature names
    hu_cols = [f'Shape.HuMoments{k}' for k in range(1, 8)]
    featname_map.update({col: None for col in hu_cols})
    
    # Add weighted Hu moments if intensity image available
    intensity_wtd = rprops[0]._intensity_image is not None
    if intensity_wtd:
        wtd_hu_cols = [col.replace('.Hu', '.WeightedHu') for col in hu_cols]
        featname_map.update({col: None for col in wtd_hu_cols})
    
    feature_list = featname_map.keys()
    mapped_feats = [k for k, v in featname_map.items() if v is not None]

    # Create output dataframe
    numFeatures = len(feature_list)
    numLabels = len(rprops)
    fdata = pd.DataFrame(np.zeros((numLabels, numFeatures)),
                        columns=feature_list)

    # Calculate features for each object
    for i, nprop in enumerate(rprops):
        # Copy directly mapped features
        for name in mapped_feats:
            fdata.at[i, name] = nprop[featname_map[name]]

        # Compute circularity
        numerator = 4 * np.pi * nprop.area
        denominator = nprop.perimeter ** 2
        if denominator > 0:
            fdata.at[i, 'Shape.Circularity'] = numerator / denominator

        # Compute minor to major axis ratio
        if nprop.major_axis_length > 0:
            fdata.at[i, 'Shape.MinorMajorAxisRatio'] = \
                nprop.minor_axis_length / nprop.major_axis_length
        else:
            fdata.at[i, 'Shape.MinorMajorAxisRatio'] = 1
            
        fdata.at[i, 'CELL_ID'] = nprop.label

        # Store Hu moments
        fdata.loc[i, hu_cols] = nprop.moments_hu
        if intensity_wtd:
            fdata.loc[i, wtd_hu_cols] = nprop.weighted_moments_hu

    return fdata

def norm_max_min(img):
    """
    Normalize image intensities using adaptive thresholding.
    
    Parameters
    ----------
    img : ndarray
        Input image
        
    Returns
    -------
    ndarray
        Normalized image with values scaled to [0,255]
    """
    const_upper = 5000
    const_lower = 10
    
    # Calculate intensity histogram
    cnts, vals = np.histogram(img, bins=range(0, 2**16, 256))
    pixels = img.shape[0] * img.shape[1]
    
    # Find threshold values
    upper = pixels/const_lower
    lower = pixels/const_upper
    crit = np.where((lower < cnts) & (cnts <= upper))[0]
    
    try:
        th_min = min(crit)
        th_max = max(crit)
        min_val, max_val = vals[th_min], vals[th_max]
    except:
        min_val = 0
        max_val = 2**16

    # Clip and normalize values
    img[img < min_val] = min_val
    img[img >= max_val] = max_val
    img = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
    
    return img

def get_img_from_acq_id(study_id, acq_id, bucket_name=DATA_BUCKET):
    """
    Load image from cache or S3 storage.
    
    Parameters
    ----------
    study_id : str
        Study identifier
    acq_id : str
        Acquisition identifier
    bucket_name : str
        S3 bucket name
        
    Returns
    -------
    ndarray
        Loaded image data
    """
    # Try loading from cache first
    cache_path = os.path.join(CACHE_PATH, 'images', study_id, f'{acq_id}.npy')
    if os.path.exists(cache_path):
        try:
            return np.load(cache_path, allow_pickle=True)
        except:
            pass
            
    # Load from S3 if not in cache
    s3_client = boto3.resource('s3')
    s3_bucket = s3_client.Bucket(bucket_name)
    
    img_key = f'{study_id}/stitched_image_output/{acq_id}.ome.tif'
    img_stream = io.BytesIO()
    s3_bucket.Object(img_key).download_fileobj(img_stream)
    img_stream.seek(0)
    
    # Load and cache image
    img = tifffile.imread(img_stream)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, img)
    
    return img

def get_segmask_from_acq_id(study_id, acq_id, version=5, bucket_name=DATA_BUCKET):
    """
    Load segmentation mask from cache or S3 storage.
    
    Parameters
    ----------
    study_id : str
        Study identifier
    acq_id : str
        Acquisition identifier  
    version : int
        Segmentation version number
    bucket_name : str
        S3 bucket name
        
    Returns
    -------
    ndarray
        Segmentation mask
    """
    # Try loading from cache first
    cache_path = os.path.join(CACHE_PATH, 'segmask', study_id, f'{acq_id}.npy')
    if os.path.exists(cache_path):
        try:
            return np.load(cache_path, allow_pickle=True)
        except:
            pass
            
    # Load from S3 if not in cache
    s3_client = boto3.resource('s3')
    s3_bucket = s3_client.Bucket(bucket_name)
    
    mask_key = f'{study_id}/segmentation/{version}/masks/{acq_id}.ome.tif'
    mask_stream = io.BytesIO()
    s3_bucket.Object(mask_key).download_fileobj(mask_stream)
    mask_stream.seek(0)
    
    # Load and cache mask
    mask = tifffile.imread(mask_stream)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.save(cache_path, mask)
    
    return mask

def load_model(model_path):
    """
    Load a trained model from a file.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model file
        
    Returns
    -------
    object
        Loaded model object
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def save_model(model, model_path):
    """
    Save a trained model to a file.
    
    Parameters
    ----------
    model : object
        Model to save
    model_path : str
        Path where model should be saved
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

def evaluate_classifier(y_true, y_pred, y_prob=None):
    """
    Evaluate a classifier using multiple metrics.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_prob : array-like, optional
        Prediction probabilities for each class
        
    Returns
    -------
    dict
        Dictionary containing various evaluation metrics:
        - accuracy
        - balanced_accuracy
        - f1_score
        - precision
        - recall
        - confusion_matrix
        - roc_auc (if y_prob provided)
        - average_precision (if y_prob provided)
    """
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    results['precision'] = precision
    results['recall'] = recall
    
    if y_prob is not None:
        results['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        results['average_precision'] = average_precision_score(y_true, y_prob)
    
    return results

def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=True, title=None):
    """
    Plot confusion matrix for classification results.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    labels : list, optional
        List of label names
    normalize : bool, default=True
        Whether to normalize confusion matrix values
    title : str, optional
        Plot title
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure containing confusion matrix plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    if labels is not None:
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    if title:
        ax.set_title(title)
    fig.tight_layout()
    
    return fig

class ImageDataset(Dataset):
    """
    PyTorch Dataset for loading image data.
    
    Parameters
    ----------
    images : list
        List of image file paths or numpy arrays
    labels : array-like
        Labels corresponding to images
    transform : callable, optional
        Optional transform to be applied to images
    """
    
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        image = self.images[idx]
        
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image)
            
        if self.transform:
            image = self.transform(image)
            
        label = self.labels[idx]
        return image, label

def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32, num_workers=4):
    """
    Create PyTorch DataLoaders for training and validation data.
    
    Parameters
    ----------
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    X_val : array-like
        Validation features
    y_val : array-like
        Validation labels
    batch_size : int, default=32
        Batch size for data loaders
    num_workers : int, default=4
        Number of worker processes for data loading
        
    Returns
    -------
    tuple
        (train_loader, val_loader) - PyTorch DataLoader objects
    """
    train_dataset = TensorDataset(torch.FloatTensor(X_train), 
                                torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val),
                              torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                          shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader