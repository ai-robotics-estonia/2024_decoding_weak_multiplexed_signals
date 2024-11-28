import sys
from pathlib import Path

import cv2
import numpy as np

import torch
from torch.nn import L1Loss, MSELoss, HuberLoss
from torch.utils.data import ConcatDataset, RandomSampler, WeightedRandomSampler

import json
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
import os

import glob

from PIL import Image
import re
from tqdm import tqdm



        
class GScan_BB(Dataset):
    def __init__(self,data_dir):
        self.abs_folders = os.path.abspath(data_dir)


        self.image_paths = os.listdir(self.abs_folders)


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        data_path=self.abs_folders+'/'+self.image_paths[index]
        label_np=np.load(data_path,allow_pickle=True)

        bounding_boxes = np.zeros((6, 4))
     
        
        # List of your matrices (P1, P2, P3 for up and down)
        input=label_np['raw_input']
        matrices = [
                    label_np['hit_upMatP1'],
                    label_np['hit_upMatP2'],
                    label_np['hit_upMatP3'],
                    label_np['hit_downMatP1'],
                    label_np['hit_downMatP2'],
                    label_np['hit_downMatP3']
                ]
        
        padded_input = [np.pad(arr, (0, max(0, 180 - len(arr))), mode='constant') 
                        for arr in input]

        for i, mat in enumerate(matrices):
            if mat.size == 0:
                # Handle the empty array case - set a default bounding box
                # For example, set the center and size as [0, 0, 0, 0]
                bounding_boxes[i]=([-1, -1, -1, -1])
            else:
                x_min, y_min = np.min(mat, axis=0)
                x_max, y_max = np.max(mat, axis=0)
                
                # Calculate center (midpoint between min and max)
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                
                # Calculate width and height
                width = x_max - x_min
                height = y_max - y_min
                
                # Append the bounding box data as [center_x, center_y, width, height]
                bounding_boxes[i]=([center_x, center_y, width, height])
       


        label=torch.tensor(bounding_boxes, dtype=torch.float32)
        padded_input = torch.tensor(padded_input, dtype=torch.float32)

        return {'input':padded_input,
                'label':label}


# train_data=GScan_Images('./data_classification_loose/train/')
# train_data=GScan_Images_coords('./data/train')
# train_data=GScan_BB('./data_bb')
# max_len=0
# for i in tqdm(range(len(train_data))):
#     for mat in range(6):
#         if (train_data[i][0][mat].shape[0]) > max_len:
#             max_len=train_data[i][0][mat].shape[0]

# print(train_data[9])

