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
from utils import *

def train_epoch(epoch, num_epochs, model, optimizer, scheduler, train_loader, rank):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = GradScaler()
    model.train()
    iterator = tqdm(train_loader)
    for batch_idx, batch in enumerate(iterator):
        imgs, targets = batch
        imgs = [img.cuda() for img in imgs]
        targets = [
            {k: v.cuda() if torch.is_tensor(v) else v for k, v in t.items()}
            for t in targets
        ]
        
        with autocast(device_type=device):
            losses_dict = model(imgs, targets)
            losses = sum(loss for loss in losses_dict.values())

        del imgs, targets, batch
        iterator.set_description(f"Epoch [{epoch+1}/{num_epochs}] Loss: {losses.item():.4f}")

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

    scheduler.step()

def val_epoch(epoch, num_epochs, model, valid_loader, rank):    
    model.eval()
    sum_abs_diff_per_class = {c: 0 for c in range(1, 5)}
    total_images = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(valid_loader)):
            imgs, targets = batch
            imgs = [img.cuda() for img in imgs]
            predictions = model(imgs)

            for pred, gt in zip(predictions, targets):
                pred_counts = {
                    c: (pred["labels"] == c).sum().item() for c in range(1, 5)
                }
                gt_counts = {c: (gt["labels"] == c).sum().item() for c in range(1, 5)}

                for c in range(1, 5):
                    diff = abs(pred_counts[c] - gt_counts[c])
                    sum_abs_diff_per_class[c] += diff

                total_images += 1

    mae_per_class = {c: sum_abs_diff_per_class[c] / total_images for c in range(1, 5)}
    overall_mae = sum(mae_per_class.values()) / len(mae_per_class)
    if rank == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation MAE per class: {mae_per_class}")
        print(
            f"Epoch [{epoch+1}/{num_epochs}] Validation Overall MAE: {overall_mae:.4f} \n"
        )

def evaluate(model, holdout_loader, rank):
    model.eval()
    sum_abs_diff_per_class = {c: 0 for c in range(1, 5)}
    total_images = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(holdout_loader)):
            imgs, targets = batch
            imgs = [img.cuda() for img in imgs]
            predictions = model(imgs)

            for pred, gt in zip(predictions, targets):
                pred_counts = {
                    c: (pred["labels"] == c).sum().item() for c in range(1, 5)
                }
                gt_counts = {c: (gt["labels"] == c).sum().item() for c in range(1, 5)}

                for c in range(1, 5):
                    diff = abs(pred_counts[c] - gt_counts[c])
                    sum_abs_diff_per_class[c] += diff

                total_images += 1

    mae_per_class = {c: sum_abs_diff_per_class[c] / total_images for c in range(1, 5)}
    overall_mae = sum(mae_per_class.values()) / len(mae_per_class)
    if rank == 0:
        print(f"Test MAE per class: {mae_per_class}")
        print(
            f"Test Overall MAE: {overall_mae:.4f} \n"
        )


def predict_and_count(model, data_loader, score_threshold):
    model.eval()
    image_counts = []

    damage_class_to_id = {
        "no-damage": 1,
        "minor-damage": 2,
        "major-damage": 3,
        "destroyed": 4,
    }
    id_to_damage_class = {v: k for k, v in damage_class_to_id.items()}

    with torch.no_grad():
        for imgs, img_ids in tqdm(
            data_loader, desc="Inference"
        ):
            imgs = [img.cuda() for img in imgs]
            outputs = model(imgs)

            for i, output in enumerate(outputs):
                img_id = img_ids[i]
                scores = output["scores"].cpu()
                keep = scores >= score_threshold
                labels = output["labels"][keep].cpu().numpy()

                counts = {class_name: 0 for class_name in damage_class_to_id.keys()}
                for label in labels:
                    class_name = id_to_damage_class.get(label, "unknown")
                    if class_name in counts:
                        counts[class_name] += 1

                image_counts.append({"img_id": img_id, "counts": counts})

    return image_counts



def create_prediction(model, test_loader, rank, submission_filename, score_threshold):
    damage_class_to_id = {
        "no-damage": 1,
        "minor-damage": 2,
        "major-damage": 3,
        "destroyed": 4,
    }
    id_to_damage_class = {v: k for k, v in damage_class_to_id.items()}
    predictions = predict_and_count(model, test_loader, score_threshold)

    rows = []
    for entry in predictions:
        img_id = entry["img_id"]
        counts = entry["counts"]

        for damage_class, count in counts.items():
            damage_class_formatted = damage_class.replace("-", "_")
            row_id = f"{img_id}_X_{damage_class_formatted}"
            rows.append({"id": row_id, "target": count})

    sub_df = pd.DataFrame(rows)
    sub_df = sub_df.sort_values("id").reset_index(drop=True)
    sub_df.to_csv(f"{submission_filename}_{rank}.csv", index=False)




def main(rank, world_size, args):
    ddp_setup(rank, world_size)
    tier1_df = get_labelled_dataset('tier1')
    tier3_df = get_labelled_dataset('tier3')
    # train_df = pd.concat([tier1_df, tier3_df])
    train_df = tier3_df.copy()
    hold_df = get_labelled_dataset('hold')
    val_df = get_labelled_dataset('test')
    test_df = get_unlabelled_dataset('predict')

    train_dataset = BuildingDataset(train_df, resize_size=(args.imgsz, args.imgsz))
    val_dataset = BuildingDataset(val_df, resize_size=(args.imgsz, args.imgsz))
    holdout_dataset = BuildingDataset(hold_df, resize_size=(args.imgsz, args.imgsz))
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

    model = get_maskrcnn_model(num_classes=5)
    model = model.cuda()
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15, 29, 43, 53, 65, 80, 90, 100, 110, 130, 150, 170, 180, 190], gamma=0.5)
    num_epochs = args.epochs

    try:
        for epoch in range(num_epochs):
            train_epoch(epoch, num_epochs, model, optimizer, scheduler, train_loader, rank)
            val_epoch(epoch, num_epochs, model, valid_loader, rank)
            torch.cuda.empty_cache()
        evaluate(model, holdout_loader, rank)
        create_prediction(model, test_loader, rank, args.submission_filename, args.score_threshold)
        destroy_process_group()
    except Exception as e:
        if rank == 0:
            print(f"Error: {str(e)}")
        destroy_process_group()
        raise e
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Starter Notebook model")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loader")
    parser.add_argument("--train-batch-size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--val-batch-size", type=int, default=4, help="Batch size for validation")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    parser.add_argument("--imgsz", type=int, default=512, help="Image size for training")
    parser.add_argument("--submission-filename", type=str, default="submission", help="Name of submission file")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Score threshold for predictions")
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)


