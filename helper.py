# Standard library imports
import json
import multiprocessing
import os
import warnings
from collections import Counter
from functools import partial
from pathlib import Path
import tarfile
import shutil

# Data manipulation and analysis
import numpy as np
import pandas as pd
from uuid import uuid4
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
from torch.amp import autocast, GradScaler
import torch.optim.lr_scheduler as lr_scheduler

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
from torch.distributed import init_process_group
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import mobilenet_backbone

# Settings
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore")

def predict_and_count(model, data_loader, device, score_threshold=0.5):
    model.eval()
    image_counts = []

    with torch.no_grad():
        for imgs, img_ids in tqdm(
            data_loader, desc="Inference"
        ):
            imgs = [img.to(device) for img in imgs]
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


def get_labelled_dataset(folder_name):
    data = []
    root = f"/kaggle/input/{folder_name}-xview-geotiffdata/{folder_name}"
    for filename in sorted(os.listdir(f"{root}/images")):        
        if 'post' in filename:
            ID = '_'.join(filename.split('_')[:-2])
            image_path = f"{root}/images/{filename.replace('post', 'pre')}"
            label_path = f"{root}/labels/{filename.replace('.tif', '.json')}"
            data.append({
                'id':ID,
                'pre_image_path':image_path,
                'post_label_path':label_path
            })
    return pd.DataFrame(data)

def get_unlabelled_dataset(folder_name):
    data = []
    root = f"/kaggle/input/{folder_name}-xview-geotiffdata"
    for filename in sorted(os.listdir(f"{root}/Images")):        
        if 'pre' in filename:
            ID = '_'.join(filename.split('_')[:-2])
            image_path = f"{root}/Images/{filename}"
            data.append({
                'id':ID,
                'pre_image_path':image_path
            })
    return pd.DataFrame(data)

def get_maskrcnn_model(num_classes=5):
    model = maskrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )
    return model


def ddp_setup(rank: int, world_size: int):
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)

def train_epoch(epoch, num_epochs, model, optimizer, train_loader, rank):
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
        iterator.set_description(f"GPU: {rank} Epoch [{epoch+1}/{num_epochs}] Loss: {losses.item():.4f}")

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

def save_model(model, epoch, mae):
    torch.save({
        'epoch': epoch,
        'mae': mae,
        'model_state_dict': model.state_dict(),
    }, "model_best.pth")
    print(f"Successfully saved model with MAE: {mae:.4f}")

def validate_epoch(epoch, num_epochs, model, valid_loader, rank, MIN_LOSS, score_threshold, classes):    
    model.eval()
    total_images = 0
    diff_sum = {class_name: 0 for class_name in classes}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(valid_loader)):
            imgs, targets = batch
            imgs = [img.cuda() for img in imgs]
            predictions = model(imgs)

            for pred, gt in zip(predictions, targets):
                # pred_counts = (pred["labels"] == 1).sum().item()
                # bool_mask = pred['scores'] >= score_threshold
                # pred_counts = (pred["labels"][bool_mask] == 1).sum().item()
                # gt_counts = (gt["labels"] == 1).sum().item()
                # diff = abs(pred_counts - gt_counts)
                # diff_sum += diff
                # total_images += 1
                bool_mask = pred['scores'] >= score_threshold
                for idx, class_name in enumerate(classes, start=1):
                    pred_counts = (pred["labels"][bool_mask] == idx).sum().item()
                    gt_counts = (gt["labels"] == idx).sum().item()
                    diff = abs(pred_counts - gt_counts)
                    diff_sum[class_name] += diff
                total_images += 1


    class_mae = {class_name: diff_sum[class_name] / total_images for class_name in classes}
    total_mae = sum(class_mae.values()) / len(class_mae)    
    if (rank == 0 ) and (MIN_LOSS is None or total_mae < MIN_LOSS):
        MIN_LOSS = total_mae
        save_model(model, epoch, total_mae)
        print(f"GPU: {rank} Epoch [{epoch+1}/{num_epochs}] Threshold: {score_threshold} Validation MAE: {total_mae:.4f} Best MAE: {MIN_LOSS:.4f} "+\
        " ".join([f"{class_name}: {mae:.4f}" for class_name, mae in class_mae.items()])+"\n")


def evaluate(model, test_loader, rank, score_threshold, classes):    
    model.eval()
    total_images = 0
    diff_sum = {class_name: 0 for class_name in classes}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            imgs, targets = batch
            imgs = [img.cuda() for img in imgs]
            predictions = model(imgs)

            for pred, gt in zip(predictions, targets):
                # pred_counts = (pred["labels"] == 1).sum().item()
                # bool_mask = pred['scores'] >= score_threshold
                # pred_counts = (pred["labels"][bool_mask] == 1).sum().item()
                # gt_counts = (gt["labels"] == 1).sum().item()
                # diff = abs(pred_counts - gt_counts)
                # diff_sum += diff
                # total_images += 1
                bool_mask = pred['scores'] >= score_threshold
                for idx, class_name in enumerate(classes, start=1):
                    pred_counts = (pred["labels"][bool_mask] == idx).sum().item()
                    gt_counts = (gt["labels"] == idx).sum().item()
                    diff = abs(pred_counts - gt_counts)
                    diff_sum[class_name] += diff
                total_images += 1  
    
    class_mae = {class_name: diff_sum[class_name] / total_images for class_name in classes}
    total_mae = sum(class_mae.values()) / len(class_mae)
    
    if rank == 0:   
        print(f"GPU: {rank} Threshold: {score_threshold} Test MAE: {total_mae:.4f} "+\
        " ".join([f"{class_name}: {mae:.4f}" for class_name, mae in class_mae.items()])+"\n") 
        torch.save(model.state_dict(), f"model_last.pth")


def predict(model, data_loader, rank, score_threshold, classes):
    model.eval()
    predictions = []

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
                boxes = output["boxes"][keep].cpu().numpy()
                masks = output["masks"][keep].cpu().numpy()
                scores = scores[keep].cpu().numpy()

                if len(labels) > 0:
                    for label, box, mask, score in zip(labels, boxes, masks, scores):
                        xmin, ymin, xmax, ymax = box
                        class_name = classes[label - 1]
                        predictions.append(
                            {
                                "Image_ID": img_id,
                                "class": class_name,
                                "confidence": score,
                                "xmin": xmin,
                                "ymin": ymin,
                                "xmax": xmax,
                                "ymax": ymax
                            }
                        )
                else:
                    predictions.append(
                        {
                            "Image_ID": img_id,
                            "class": "unknown",
                            "confidence": 0,
                            "xmin": 0,
                            "ymin": 0,
                            "xmax": 0,
                            "ymax": 0
                        }
                    )
    df = pd.DataFrame(predictions)
    sub_id = ''.join(str(uuid4()).split("-")[:3]) + f'_{rank}_' + '_'.join(classes) + '.csv'
    df.to_csv(sub_id, index=False)
