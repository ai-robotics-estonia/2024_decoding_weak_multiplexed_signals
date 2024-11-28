from dataloader import GScan_BB
from model import BB_model
from torch.utils.data import Dataset, random_split, DataLoader,ConcatDataset
import torch
from torch.nn import BCELoss, L1Loss
from loss import CustomMaskedMSELoss,CustomIOU

from tqdm import tqdm

import wandb
import random
from datetime import datetime


torch.manual_seed(42)
random.seed(42)
best_val_loss=float('inf')
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

base_name = "linear1"
run_name = f"{base_name}_{current_time}"

n_epochs=5000
BATCH_SIZE=1024
num_workers=1
lr=0.01

config = {
    "learning_rate": lr,
    "batch": BATCH_SIZE,
}



full_data = GScan_BB('./data_square_hit')

# Split proportions
train_ratio = 0.8  # 80% for training
val_ratio = 0.2    # 20% for validation

# Calculate split sizes
total_size = len(full_data)
train_size = int(train_ratio * total_size)
val_size = total_size - train_size

# Randomly split the dataset
train_data, val_data = random_split(full_data, [train_size, val_size])

# DataLoader for training and validation sets
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BB_model()
# model.load_state_dict(torch.load('weights/linear1_2024-11-04_02-43-03.pth'))
model.to(device=device)



optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss=CustomMaskedMSELoss()
eval=CustomIOU()

run_name = f"{base_name}_{current_time}"

filename = f"weights/{base_name}_{current_time}.pth"
wandb.init(project="gscan", config=config, name=run_name) 
wandb.watch(model)

for epoch in range(n_epochs):
    model.train()
    total_training_loss = 0
    total_training_iou = 0

    for i, data in enumerate( tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')):
        input=data['input'].to(device=device) 


        label=data['label'].to(device=device)

        output=model(input)


        
        train_loss=loss(output,label)
        total_training_loss+=train_loss.item()


        train_loss.backward()
        optimizer.step() #Apply
        optimizer.zero_grad()

        train_iou=eval(output,label)
        total_training_iou+=train_iou.item()

        wandb.log({"Train Loss": train_loss})

    avgTrainLoss = total_training_loss / len(train_dataloader)
    avgTrainIOU = total_training_iou / len(train_dataloader)

    model.eval()
    total_validation_loss = 0
    total_validation_iou = 0

    with torch.no_grad():
        for i, data in enumerate( tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')):
            input=data['input'].to(device=device) 
    
    
            label=data['label'].to(device=device)


            output=model(input)

            val_loss=loss(output,label)
            val_iou=eval(output,label)

            
            total_validation_loss+=val_loss.item()
            total_validation_iou+=val_iou.item()

    avgValLoss = total_validation_loss / len(val_dataloader)
    avgValIOU = total_validation_iou / len(val_dataloader)
    
    wandb.log({
        "Average Train Loss": avgTrainLoss,
        "Average Train IOU": avgTrainIOU,
        "Average Validation Loss": avgValLoss,
        "Average Validation IOU": avgValIOU

    })

    if avgValLoss<best_val_loss:
        best_val_loss=avgValLoss
        best_val=model

        torch.save(best_val.state_dict(), filename)





















    

 

    


