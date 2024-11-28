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

class GScan_Images(Dataset):
    def __init__(self,data_dir):
        self.abs_folders = os.path.abspath(data_dir)


        self.image_paths = os.listdir(self.abs_folders)
        self.transforms = transforms.Compose([#transforms.Resize((1024, 1024)),
                                             transforms.ToTensor(),
        ])


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        img_path=self.abs_folders+'/'+self.image_paths[index]




        pattern = r'.*_(yes|no)\b'
        match = re.search(pattern, img_path)
                # Convert "yes" to 1 and "no" to 0
        label = 1 if match and match.group(1) == 'yes' else 0 if match and match.group(1) == 'no' else None

        input_front=img_path + '/input_front.png'
        input_side=img_path + '/input_side.png'

        front_image= self.transforms(Image.open(input_front))
        side_image= self.transforms(Image.open(input_side))

        label=torch.tensor(label, dtype=torch.float32)



        return {'front': front_image,
                'side': side_image,
                'label': label
        }

        


# train_data=GScan_Images('./data_classification_loose/train/')
# train_data=GScan_Images_coords('./data/train')
# train_data=GScan_BB('./data_bb')
# max_len=0
# for i in tqdm(range(len(train_data))):
#     for mat in range(6):
#         if (train_data[i][0][mat].shape[0]) > max_len:
#             max_len=train_data[i][0][mat].shape[0]

# print(train_data[9])

