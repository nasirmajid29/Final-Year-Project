import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius

from GPUtil import showUtilization as gpu_usage
import matplotlib.pyplot as plt
from point_dataset import PointDataset
import numpy as np

import wandb
import pyvista
import matplotlib.pyplot as plt
import os

data = "take_off_weighing_scales_100eps"
wandb.init(project="Architectures", entity="final-year-project", name=data+"_pointnet")

wandb.config.update({
  "learning_rate": 0.001,
  "optimiser": "Adam",
  "epochs": 30,
  "batch_size": 32,
  "ratio": [0.1, 0.05],
  "radius": [0.05, 0.1]
})


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.sa1_module = GlobalSAModule(MLP([6, 16, 32, 64, 128, 256, 512]))

        self.mlp = MLP([512, 256, 128, 64, 32, 16, 8], dropout=0.0, norm=None)

    def forward(self, data):
        # print(data.dtype)
        sa0_out = (data.x, data.pos, data.batch)
        # print(np.array(data.batch).shape)
        sa1_out = self.sa1_module(*sa0_out)
        x, pos, batch = sa1_out

        # print(x.dtype)

        return self.mlp(x)


def train(epoch):
    model.train()
    total_loss = 0
    
    for data in train_loader:


        data = data.to(device)
        optimizer.zero_grad()

        pred =  model(data)
        loss = F.mse_loss(pred, data.y) 
        
        total_loss += loss

        
        loss.backward()
        optimizer.step()
        # break


    avg_loss = total_loss / len(train_loader)
    print(f'Epoch: {epoch:03d}, Training Loss: {avg_loss:.4f}')

    wandb.log({"training loss": avg_loss})
    wandb.watch(model)
        

def test(loader):
    model.eval()

    total_loss = 0
    total_cm_distance = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            # print("Prediction: ", pred)
        #     predmax = model(data).max(1)[1]
        # correct += predmax.eq(data.y).sum().item()
            loss = F.mse_loss(pred, data.y) #, correct / len(loader.dataset),  
            total_loss += loss
            
            data.y = data.y.detach().cpu().numpy()
            pred = pred.detach().cpu().numpy()
            max_x = 0.01
            min_x = -0.01
            range_x = max_x - min_x
            
            max_y = 0.02
            min_y = -0.02
            range_y = max_y - min_y
            
            max_z = 0.025
            min_z = -0.005
            range_z = max_z - min_z
            
            for i in range(len(pred)):
                
                pred[i][0] = (range_x * (pred[i][0] + 1)/2) + min_x    
                pred[i][1] = (range_y * (pred[i][1] + 1)/2) + min_y    
                pred[i][2] = (range_z * (pred[i][2] + 1)/2) + min_z    
                
                data.y[i][0] = (range_x * (data.y[i][0] + 1)/2) + min_x    
                data.y[i][1] = (range_y * (data.y[i][1] + 1)/2) + min_y    
                data.y[i][2] = (range_z * (data.y[i][2] + 1)/2) + min_z    
                
                
                cm_dist = np.linalg.norm(pred[i][:3] - data.y[i][:3])
                total_cm_distance += cm_dist
                
    val_loss = total_loss / len(loader)
    cm_off = total_cm_distance / len(loader)
    return val_loss, cm_off   


if __name__ == '__main__':

    gpu_usage()
    torch.cuda.empty_cache()
    gpu_usage()
    
    data_loc = "/vol/bitbucket/nm219/data/"+data
    
    pre_transform = None #T.NormalizeScale()
    
    point_cloud_data_train = PointDataset(data_loc, "data.pt", True, pre_transform)    
    point_cloud_data_test = PointDataset(data_loc, "data.pt", False, pre_transform)
        

    train_loader = DataLoader(point_cloud_data_train, batch_size=64, shuffle=False, num_workers=6) #shuffle true #batch 32
    test_loader = DataLoader(point_cloud_data_test, batch_size=64, shuffle=False,num_workers=6)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print("Number of Parameters: ", total_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) #0.001

    torch.cuda.empty_cache()

    epoch_losses = []
    accuracies = []
    for epoch in range(50): #201
        train(epoch)
        loss, translation_error = test(test_loader)
        loss = loss.detach().cpu().numpy()
        epoch_losses.append(loss)
        # accuracies.append(test_acc)
        print(f'Epoch: {epoch:03d}, Validation Loss: {loss:.4f}, Translation Error {translation_error:.4f}')

        wandb.log({"validation loss": loss})
        wandb.log({"translation error": translation_error})
        wandb.watch(model)

        torch.save(model, data+"_pointnet.pt")