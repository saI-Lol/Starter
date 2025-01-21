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
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import mobilenet_backbone

def collate_fn(batch):
    imgs = [item["img"] for item in batch]
    targets = [item["target"] for item in batch]
    return imgs, targets

def test_collate_fn(batch):
    imgs = [item["img"] for item in batch]
    img_ids = [item["img_id"] for item in batch]

    return imgs, img_ids
    
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

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row.pre_image_path
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

            polygon_wkt = annotation["wkt"]
            polygon = loads(polygon_wkt)

            if not polygon.is_valid:
                continue

            polygon = self._clip_polygon_to_image(polygon, width, height)
            if polygon.is_empty:
                continue

            xmin, ymin, xmax, ymax = polygon.bounds
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)

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
        img_path = row.pre_image_path
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