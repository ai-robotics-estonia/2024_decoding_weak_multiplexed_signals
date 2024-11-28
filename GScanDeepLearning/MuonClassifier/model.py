
import torch


import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F
import torchvision.models as models


import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models import resnet34
import cv2

class MuonClassifier(nn.Module):
    def __init__(self):
        super(MuonClassifier, self).__init__()
        # Feature extractor for both inputs (front and side)
        self.feature_extractor = nn.Sequential( 
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
        )

        # Additional convolutional layers after combining features
        self.additional_conv_layers = nn.Sequential(

            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # Input channels are doubled after concatenation
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)  # Another pooling operation after further conv layers
        )

        # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(293632,1)
        )  # Adjust according to the image size after pooling

    def forward(self, front, side):
        # Extract features from both inputs
        features_front = self.feature_extractor(front)  # Shape: (batch_size, 256, H, W)
        features_side = self.feature_extractor(side)    # Shape: (batch_size, 256, H, W)

        # Combine features (concatenation along the channel axis)
        combined_features = torch.cat((features_front, features_side), dim=1)  # Shape: (batch_size, 512, H, W)

        # Apply additional convolutional layers after combining
        processed_features = self.additional_conv_layers(combined_features)  # Shape: (batch_size, 128, H/2, W/2)


        # Pass through the fully connected layer
        label = self.fc(processed_features)

        # Optionally, apply a sigmoid activation for binary classification
        label = torch.sigmoid(label)

        return label

