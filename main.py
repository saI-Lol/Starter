# Standard library imports
import json
import torch.multiprocessing as mp
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import warnings
from collections import Counter
from functools import partial
from pathlib import Path
import tarfile
import shutil
import argparse
from torch.amp import autocast, GradScaler
import torch.optim.lr_scheduler as lr_scheduler
# Data manipulation and analysis
import numpy as np
import pandas as pd
from tqdm import tqdm

# Visualization
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# Geospatial libraries
import geopandas as gpd
import pyproj
import pyproj.datadir
import pystac_client
import rasterio
import rioxarray
import xarray as xr
from pyproj import CRS, Transformer
from rasterio.mask import mask
from rasterio.plot import reshape_as_raster
from rasterio.transform import from_bounds
from shapely import wkt
from shapely.geometry import Polygon, box
from shapely.wkt import loads

# AWS
import boto3
from botocore import UNSIGNED
from botocore.client import Config

# PyTorch and computer vision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.transforms import functional as F

# Settings
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore")
from helper import ddp_setup, get_labelled_dataset, get_unlabelled_dataset, get_maskrcnn_model, train_epoch, validate_epoch, evaluate, predict
from dataset import BuildingDataset, collate_fn, TestDataset, test_collate_fn


def main(rank, world_size, args):
    ddp_setup(rank, world_size)
    classes = args.classes
    tier1_df = get_labelled_dataset('tier1')
    tier3_df = get_labelled_dataset('tier3')
    # train_df = pd.concat([tier1_df, tier3_df])
    train_df = tier1_df
    hold_df = get_labelled_dataset('hold')
    val_df = get_labelled_dataset('test')
    test_df = get_unlabelled_dataset('predict')

    train_dataset = BuildingDataset(train_df, classes, resize_size=(args.imgsz, args.imgsz))
    val_dataset = BuildingDataset(val_df, classes, resize_size=(args.imgsz, args.imgsz))
    holdout_dataset = BuildingDataset(hold_df, classes, resize_size=(args.imgsz, args.imgsz))
    test_dataset = TestDataset(test_df, resize_size=(args.imgsz, args.imgsz))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        sampler = DistributedSampler(train_dataset)
    )

    valid_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        sampler = DistributedSampler(val_dataset)
    )

    holdout_loader = DataLoader(
        holdout_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        sampler = DistributedSampler(holdout_dataset)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=test_collate_fn,
        sampler = DistributedSampler(test_dataset)
    )

    model = get_maskrcnn_model(num_classes=len(classes) + 1)
    model = model.cuda()
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    num_epochs = args.epochs
    MIN_LOSS = None

    try:
        for epoch in range(num_epochs):
            train_epoch(epoch, num_epochs, model, optimizer, train_loader, rank)
            validate_epoch(epoch, num_epochs, model, valid_loader, rank, MIN_LOSS, args.score_threshold, classes)
            torch.cuda.empty_cache()
        evaluate(model, holdout_loader, rank, args.score_threshold, classes)
        predict(model, test_loader, rank, args.score_threshold, classes)
        destroy_process_group()
    except Exception as e:
        if rank == 0:
            print(f"Error: {str(e)}")
        destroy_process_group()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Starter Notebook model")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loader")
    parser.add_argument("--train-batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--val-batch-size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size for training")
    parser.add_argument("--score-threshold", type=float, required=True, help="Score threshold for predictions")
    parser.add_argument("--classes", type=str, nargs="+", default=["no_damage", "minor_damage", "major_damage", "destroyed"], help="Classes to predict")
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)


