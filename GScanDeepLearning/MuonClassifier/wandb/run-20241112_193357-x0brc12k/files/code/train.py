from dataloader import GScan_Images
from model import MuonClassifier
from torch.utils.data import Dataset, random_split, DataLoader,ConcatDataset
import torch
from torch.nn import BCELoss, L1Loss


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

n_epochs=100
BATCH_SIZE=4
num_workers=2
lr=0.01

config = {
    "learning_rate": lr,
    "batch": BATCH_SIZE,
}



train_data=GScan_Images('./data_classification_loose/train/')
val_data=GScan_Images('./data_classification_loose/test')

# DataLoader for training and validation sets
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model= MuonClassifier()

model.to(device=device)



optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss=BCELoss()

run_name = f"{base_name}_{current_time}"

filename = f"weights/{base_name}_{current_time}.pth"
wandb.init(project="gscan", config=config, name=run_name) 
wandb.watch(model)

for epoch in range(n_epochs):
    model.train()
    total_training_loss = 0


    for i, data in enumerate( tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')):
        front = data['front'].to(device=device) 
        side = data['side'].to(device=device)
        label = data['label'].to(device=device)

        output = model(front, side).squeeze(1)

        
        train_loss=loss(output,label)
        total_training_loss+=train_loss.item()


        train_loss.backward()
        optimizer.step() #Apply
        optimizer.zero_grad()

        train_iou=eval(output,label)


        wandb.log({"Train Loss": train_loss})

    avgTrainLoss = total_training_loss / len(train_dataloader)


    model.eval()
    total_validation_loss = 0


    with torch.no_grad():
        for i, data in enumerate( tqdm(val_dataloader, desc=f'Epoch {epoch + 1}/{n_epochs}')):
            front = data['front'].to(device=device) 
            side = data['side'].to(device=device)
            label = data['label'].to(device=device)
    
            output = model(front, side).squeeze(1)

            val_loss=loss(output,label)


            
            total_validation_loss+=val_loss.item()


    avgValLoss = total_validation_loss / len(val_dataloader)

    
    wandb.log({
        "Average Train Loss": avgTrainLoss,
        "Average Validation Loss": avgValLoss
    })

    if avgValLoss<best_val_loss:
        best_val_loss=avgValLoss
        best_val=model

        torch.save(best_val.state_dict(), filename)





















    

 

    


