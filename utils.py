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
from torch.distributed import init_process_group

# Settings
pd.set_option("display.max_colwidth", None)
pd.set_option("display.max_columns", None)
warnings.filterwarnings("ignore")

class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        orig_width, orig_height = image.size
        image = TF.resize(image, self.size)

        ratio_width = self.size[0] / orig_width
        ratio_height = self.size[1] / orig_height

        if "boxes" in target:
            boxes = target["boxes"]
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * ratio_width
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * ratio_height
            target["boxes"] = boxes

        if "masks" in target:
            masks = target["masks"]
            masks = masks.unsqueeze(1)  # Add channel dimension
            masks = TF.resize(
                masks, self.size, interpolation=TF.InterpolationMode.NEAREST
            )
            masks = masks.squeeze(1)
            target["masks"] = masks

        return image, target


class BuildingDataset(Dataset):
    def __init__(self, df, resize_size=(512, 512)):
        self.df = df.reset_index(drop=True)
        self.resize_size = resize_size
        self.damage_class_to_id = {
            "no-damage": 1,
            "minor-damage": 2,
            "major-damage": 3,
            "destroyed": 4,
        }

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row.post_image_path
        label_path = row.post_label_path
        image_id = torch.tensor([idx])

        with rasterio.open(img_path) as src:
            img_array = src.read()
            img_array = np.transpose(img_array, (1, 2, 0)).astype(np.uint8)
            img = Image.fromarray(img_array)

        width, height = img.size

        with open(label_path, "r") as f:
            annotations = json.load(f)

        boxes = []
        labels = []
        masks = []
        annotations = annotations["features"]["xy"]

        for annotation in annotations:
            properties = annotation["properties"]
            subtype = properties["subtype"]
            damage_label = self.damage_class_to_id.get(subtype, 0)

            polygon_wkt = annotation["wkt"]
            polygon = loads(polygon_wkt)

            if not polygon.is_valid:
                continue

            polygon = self._clip_polygon_to_image(polygon, width, height)
            if polygon.is_empty:
                continue

            xmin, ymin, xmax, ymax = polygon.bounds
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(damage_label)

            mask = self._polygon_to_mask(polygon, width, height)
            masks.append(mask)

        if len(boxes) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "masks": torch.zeros((0, height, width), dtype=torch.uint8),
                "image_id": image_id,
            }
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.stack(masks, axis=0), dtype=torch.uint8)

            target = {
                "boxes": boxes,
                "labels": labels,
                "masks": masks,
                "image_id": image_id,
            }

        resize_transform = Resize(self.resize_size)
        img, target = resize_transform(img, target)
        img = TF.to_tensor(img)

        return {
            "img": img,
            "target": target
        }

    def __len__(self):
        return len(self.df)

    def _polygon_to_mask(self, polygon, width, height):
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        if polygon.is_empty:
            return np.array(mask, dtype=np.uint8)

        if polygon.geom_type == "Polygon":
            if polygon.exterior is not None:
                x, y = polygon.exterior.coords.xy
                coords = [(xi, yi) for xi, yi in zip(x, y)]
                draw.polygon(coords, outline=1, fill=1)
        elif polygon.geom_type == "MultiPolygon":
            for poly in polygon.geoms:
                if poly.exterior is not None:
                    x, y = poly.exterior.coords.xy
                    coords = [(xi, yi) for xi, yi in zip(x, y)]
                    draw.polygon(coords, outline=1, fill=1)

        return np.array(mask, dtype=np.uint8)

    def _clip_polygon_to_image(self, polygon, width, height):
        image_box = box(0, 0, width, height)
        return polygon.intersection(image_box)


class TestDataset(Dataset):
    def __init__(self, df, resize_size=(512, 512)):
        self.df = df.reset_index(drop=True)
        self.resize_size = resize_size

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row.post_image_path
        img_id = row.id

        with rasterio.open(img_path) as src:
            img_array = src.read()
            img_array = np.transpose(img_array, (1, 2, 0)).astype(np.uint8)
            img = Image.fromarray(img_array)

        img = F.resize(img, self.resize_size)
        img = F.to_tensor(img)

        return {
            "img": img,
            "img_id": img_id
        }

    def __len__(self):
        return len(self.df)


def test_collate_fn(batch):
    imgs = [item["img"] for item in batch]
    img_ids = [item["img_id"] for item in batch]

    return imgs, img_ids


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
            image_path = f"{root}/images/{filename}"
            label_path = f"{root}/labels/{filename.replace('.tif', '.json')}"
            data.append({
                'id':ID,
                'post_image_path':image_path,
                'post_label_path':label_path
            })
    return pd.DataFrame(data)

def get_unlabelled_dataset(folder_name):
    data = []
    root = f"/kaggle/input/{folder_name}-xview-geotiffdata"
    for filename in sorted(os.listdir(f"{root}/Images")):        
        if 'post' in filename:
            ID = '_'.join(filename.split('_')[:-2])
            image_path = f"{root}/Images/{filename}"
            data.append({
                'id':ID,
                'post_image_path':image_path
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


def collate_fn(batch):
    imgs = [item["img"] for item in batch]
    targets = [item["target"] for item in batch]
    return imgs, targets

def ddp_setup(rank: int, world_size: int):
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   torch.cuda.set_device(rank)
   init_process_group(backend="nccl", rank=rank, world_size=world_size)