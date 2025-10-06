import os
import json
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class CephDataset(Dataset):
    def __init__(self, data_dir, mode='train', img_size=512):
        self.data_dir = data_dir
        self.mode = mode
        self.img_size = img_size

        # Paths
        self.img_dir = os.path.join(data_dir, mode, 'Cephalograms')
        self.anno_dir = os.path.join(
            data_dir, mode, 'Annotations',
            'Cephalometric Landmarks', 'Senior Orthodontists'
        )

        # Load mappings
        self.mappings = pd.read_csv(
            os.path.join(data_dir, 'cephalogram_machine_mappings.csv')
        )

        # Get images
        self.images = sorted([f for f in os.listdir(self.img_dir)
                             if f.endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))

        # Normalize
        image = image.astype(np.float32) / 255.0
        image = (image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        image = torch.FloatTensor(image).permute(2, 0, 1)

        # Load annotations
        anno_path = os.path.join(self.anno_dir, img_name.replace('.png', '.json'))
        with open(anno_path, 'r') as f:
            anno = json.load(f)

        # Extract landmarks và normalize về [0, 1]
        landmarks = []
        for lm in anno['landmarks']:
            x = lm['value']['x'] / w  # Normalize x
            y = lm['value']['y'] / h  # Normalize y
            landmarks.append([x, y])
        landmarks = torch.FloatTensor(landmarks)

        # Get pixel size
        ceph_id = anno['ceph_id']
        pixel_size = self.mappings[
            self.mappings['cephalogram_id'] == ceph_id
        ]['pixel_size'].values[0]

        return {
            'image': image,
            'landmarks': landmarks,
            'pixel_size': pixel_size,
            'original_size': (h, w)
        }
