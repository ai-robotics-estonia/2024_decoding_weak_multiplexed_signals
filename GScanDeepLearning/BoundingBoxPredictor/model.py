
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



class BB_model(nn.Module):
    def __init__(self, input_size=180, hidden_size=256, output_size=4):
        super(BB_model, self).__init__()
        
        # Define 12 processing blocks for each of the inputs
        self.process1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )
        self.process2 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )
        self.process3 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )
        self.process4 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )
        self.process5 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )
        self.process6 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )
        self.process7 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )
        self.process8 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )
        self.process9 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )
        self.process10 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )
        self.process11 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)
        )
        self.process12 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size)




            
        )
        
        # Define a fully connected layer to process merged output
        self.fc_merge =  nn.Sequential(
            nn.Linear(12 * hidden_size, 12 * hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.3),  # Added dropout
            nn.Linear(12 * hidden_size, 6 * hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.3),  # Added dropout
            nn.Linear(6 * hidden_size, 6 * hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.3),  # Added dropout
            nn.Linear(6 * hidden_size, 3 * hidden_size),
            nn.BatchNorm1d(3 * hidden_size)
        )
        
        # Define the output layer
        self.fc_output = nn.Linear(3*hidden_size, 6 * output_size)
        
    def forward(self, x):
        # Assume input x has shape (batch_size, 12, 180)
        
        # Split the input tensor along the second dimension to get each part (shape: (batch_size, 180) each)
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = torch.chunk(x, chunks=12, dim=1)
        
        # Remove the extra dimension for each tensor in the batch
        x1 = x1.squeeze(1)
        x2 = x2.squeeze(1)
        x3 = x3.squeeze(1)
        x4 = x4.squeeze(1)
        x5 = x5.squeeze(1)
        x6 = x6.squeeze(1)
        x7 = x7.squeeze(1)
        x8 = x8.squeeze(1)
        x9 = x9.squeeze(1)
        x10 = x10.squeeze(1)
        x11 = x11.squeeze(1)
        x12 = x12.squeeze(1)
        
        # Process each input through its respective processing block
        x1 = self.process1(x1)
        x2 = self.process2(x2)
        x3 = self.process3(x3)
        x4 = self.process4(x4)
        x5 = self.process5(x5)
        x6 = self.process6(x6)
        x7 = self.process7(x7)
        x8 = self.process8(x8)
        x9 = self.process9(x9)
        x10 = self.process10(x10)
        x11 = self.process11(x11)
        x12 = self.process12(x12)

        
        # Concatenate processed outputs along the last dimension
        x = torch.cat([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12], dim=-1)

        
        # Further process the concatenated result
        x = self.fc_merge(x)
        
        # Output layer
        x = self.fc_output(x)
        
        # Reshape to desired output size (batch_size, 12, 4)
        output = x.view(-1, 6, 4)
        
        return output
    

