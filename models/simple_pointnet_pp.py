#Inspired by https://wandb.ai/geekyrakshit/pyg-point-cloud/reports/Point-Cloud-Classification-Using-PyTorch-Geometric--VmlldzozMTExMTE3
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointConv, fps, global_max_pool, radius
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch.optim.lr_scheduler import StepLR

from positional_encoding import PositionalEncoding

from GPUtil import showUtilization as gpu_usage
import matplotlib.pyplot as plt
from point_dataset import PointDataset
import numpy as np

import wandb
import pyvista
import matplotlib.pyplot as plt
import os
import torch
from positional_encodings.torch_encodings import PositionalEncoding3D, Summer


data = 'reach_target_100eps'
wandb.init(project="Pos Encoding", entity="final-year-project", name=data)

wandb.config.update({
  "learning_rate": 0.001,
  "optimiser": "Adam",
  "epochs": 30,
  "batch_size": 32,
  "ratio": [0.1, 0.05],
  "radius": [0.05, 0.1]
})


# os.environ['WANDB_MODE'] = 'offline'

def visualise_pc_radius(point_cloud, indices, radius):
    
    points, colours = np.hsplit(point_cloud, 2)
    highlighted = points[indices]
    black = np.full((len(highlighted), 3), 0)
    plotter = pyvista.Plotter()
    plotter.add_points(points, opacity=1, point_size=4, render_points_as_spheres=True, scalars=colours.astype(int), rgb=True)
    plotter.add_points(highlighted, opacity=1, point_size=5, render_points_as_spheres=True, scalars=black.astype(int), rgb=True)

    for point in highlighted:
        sphere = pyvista.Sphere(radius, point)
        plotter.add_sphere_widget(sphere)
    plotter.show()


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        # visualise_pc_radius(x.detach().numpy(), idx.detach().numpy(), self.r)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]

        # print("X dtype", x.dtype)
        # print("Pos dtype", pos.dtype)
        # print("Edge dtype", edge_index.dtype)


        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


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

        # Input channels account for both `pos` and node features.
        # self.sa1_module = SAModule(0.4, 0.2, MLP([6, 32, 64]))
        # self.sa2_module = SAModule(0.25, 0.5, MLP([64 + 3, 128, 256]))
        # self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512]))

        # self.mlp = MLP([512, 256, 32, 3], dropout=0.5, norm=None)
        
        # self.sa1_module = SAModule(0.1, 0.05, MLP([6, 32, 64]))
        # self.sa2_module = SAModule(0.05, 0.1, MLP([64 + 3, 128, 256]))
        # self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512]))

        # self.mlp = MLP([512, 256, 32, 3], dropout=0.5, norm=None)
        
        # self.sa1_module = SAModule(0.1, 0.05, MLP([6, 16, 32])) #SAModule(0.1, 0.05, MLP([9, 16, 32]))
        # # self.sa2_module = SAModule(0.05, 0.1, MLP([64 + 3, 128, 256]))
        # self.sa3_module = GlobalSAModule(MLP([32 + 3, 64]))

        # self.mlp = MLP([64, 32, 3], dropout=0.1, norm=None)
        
        
        # self.sa1_module = SAModule(0.1, 0.05, MLP([6, 16, 32]))
        # self.sa2_module = SAModule(0.05, 0.1, MLP([32 + 3, 64]))
        
        # self.sa1_module = SAModule(0.4, 0.05, MLP([6, 16, 24, 32]))
        # # self.sa2_module = SAModule(0.2, 0.05, MLP([32 + 3, 64]))
        # self.sa3_module = GlobalSAModule(MLP([32 + 3, 64]))

        # self.mlp = MLP([64, 32, 3], dropout=0.5, norm=None)
        
        # self.sa1_module = SAModule(0.5, 0.2, MLP([6, 64, 64, 128]))
        # self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        # self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        # self.mlp = MLP([1024, 512, 256, 3], dropout=0.5, norm=None)
        
        # self.sa1_module = SAModule(0.5, 0.2, MLP([6, 16]))
        # self.sa2_module = SAModule(0.25, 0.4, MLP([16 + 3, 32]))
        # self.sa3_module = GlobalSAModule(MLP([32 + 3, 64]))

        # self.mlp = MLP([64, 32, 3], dropout=0.5, norm=None)
        
        positional_encoding = PositionalEncoding() #, num_freq)
        pos_net = nn.Sequential(positional_encoding, MLP([9, 16, 32]))
        self.sa1_module = SAModule(0.5, 0.2, pos_net)
        
        # self.sa1_module = SAModule(0.5, 0.2, MLP([6, 16, 32], norm=None))
        self.sa2_module = SAModule(0.25, 0.4, MLP([32 + 3, 64, 128], norm=None))
        self.sa3_module = GlobalSAModule(MLP([128 + 3, 256, 512], norm=None))

        self.mlp = MLP([512, 128, 32, 3], dropout=0.0 , norm=None) #0.5
        
        # self.mlp = MLP([512, 256, 128, 32, 3], dropout=0.0 , norm=None) #0.5
        
        # self.sa1_module = SAModule(0.09, 0.191, MLP([6, 8, 16, 32]))
        # self.sa2_module = SAModule(0.365, 0.062, MLP([32 + 3, 64, 128]))
        # self.sa3_module = GlobalSAModule(MLP([128 + 3, 256, 512]))

        # self.mlp = MLP([512, 128, 64, 32, 3], dropout=0.636, norm=None)
        
        # self.sa1_module = SAModule(0.5, 0.2, MLP([6, 16]))
        # self.sa2_module = SAModule(0.25, 0.4, MLP([16 + 3, 32]))
        # self.sa3_module = SAModule(0.125, 0.8, MLP([32 + 3, 64]))
        # self.sa4_module = GlobalSAModule(MLP([64 + 3, 128]))

        # self.mlp = MLP([128, 64, 3], dropout=0.5, norm=None)
        
        
        # self.sa1_module = SAModule(0.5, 0.2, MLP([6, 32, 64, 64, 128]))
        # self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 256]))
        # self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        # self.mlp = MLP([1024, 512, 256, 128, 64, 3], dropout=0.5, norm=None)
        
        # self.sa1_module = SAModule(0.5, 0.2, MLP([6, 8, 16, 32, 64]))
        # self.sa3_module = GlobalSAModule(MLP([64+3, 128, 256, 512]))

        # self.mlp = MLP([512, 256, 128, 64, 32, 16, 3], dropout=0.5, norm=None)
    

    def forward(self, data):
        # print(data.dtype)
        sa0_out = (data.x, data.pos, data.batch)
        # # print(np.array(data.batch).shape)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out
        
        # sa1_out = self.sa1_module(*sa0_out)
        # sa2_out = self.sa2_module(*sa1_out)
        # sa3_out = self.sa3_module(*sa2_out)
        # sa4_out = self.sa4_module(*sa3_out)
        # x, pos, batch = sa4_out
        
        # sa1_out = self.sa1_module(*sa0_out)
        # sa3_out = self.sa3_module(*sa1_out)
        # x, pos, batch = sa3_out
        
        # print(x.dtype)

        return self.mlp(x)
      


def train(epoch):
    model.train()
    total_loss = 0
    
    for data in train_loader:


        data = data.to(device)
        optimizer.zero_grad()

        pred =  model(data)
        # print(pred, data.y)
        loss = F.mse_loss(pred, data.y) 
        # F.smooth_l1_loss(pred, data.y)

        total_loss += loss

        
        loss.backward()
        optimizer.step()
        # break


    avg_loss = total_loss / len(train_loader)
    print(f'Epoch: {epoch:03d}, Training Loss: {avg_loss:.4f}')

    wandb.log({"training loss": avg_loss})
    wandb.watch(model)
    return avg_loss    

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

    print(torch.cuda.is_available())
    gpu_usage()
    torch.cuda.empty_cache()
    gpu_usage()
    
    data_loc = "/vol/bitbucket/nm219/data/"+data
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..',
    #                 'data/ModelNet10')

    pre_transform = None # T.NormalizeScale()
    transform = None
    # transform = T.SamplePoints(10)
    
    point_cloud_data_train = PointDataset(data_loc, "data.pt", True, pre_transform, transform)
    
    point_cloud_data_test = PointDataset(data_loc, "data.pt", False, pre_transform, transform)
    
    print(data)
    print(len(point_cloud_data_train))
    print(len(point_cloud_data_test))
    
    
    # train_dataset = ModelNet(path, '10', True, transform, pre_transform)
    # test_dataset = ModelNet(path, '10', False, transform, pre_transform)

    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
    #                           num_workers=6)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
    #                          num_workers=6)


    train_loader = DataLoader(point_cloud_data_train, batch_size=64, shuffle=False, num_workers=6) #16
    test_loader = DataLoader(point_cloud_data_test, batch_size=64, shuffle=False, num_workers=6) #32
    
    # train_x = []
    # train_y = []
    # train_z = []
    # for data in train_loader:
    #     for point in data.pos:
    #         train_x.append(point[0])
    #         train_y.append(point[1])
    #         train_z.append(point[2])
            
    # # fixed bin size
    # bins = np.arange(-0.11, 0.11, 0.005) # fixed bin size

    # plt.xlim([-0.11, 0.11])
    # plt.hist(train_x, bins=bins, alpha=1, edgecolor='black', linewidth=1.2)
    # plt.title('Training data - X coordinate')
    # plt.xlabel('X values')
    # plt.ylabel('count')
    # plt.savefig('train_x.png')
    # plt.clf()
    
    # plt.xlim([-0.11, 0.11])
    # plt.hist(train_y, bins=bins, alpha=1, edgecolor='black', linewidth=1.2)
    # plt.title('Training data - Y coordinate')
    # plt.xlabel('Y values')
    # plt.ylabel('count')
    # plt.savefig('train_y.png')
    # plt.clf()
    
    # plt.xlim([-0.11, 0.11])
    # plt.hist(train_z, bins=bins, alpha=1, edgecolor='black', linewidth=1.2)
    # plt.title('Training data - Z coordinate')
    # plt.xlabel('Z values')
    # plt.ylabel('count')
    # plt.savefig('train_z.png')   
    # plt.clf()
    
    # test_x = []
    # test_y = []
    # test_z = []
    # for data in test_loader:
    #     for point in data.pos:
    #         test_x.append(point[0])
    #         test_y.append(point[1])
    #         test_z.append(point[2])
            
    # # fixed bin size
    # bins = np.arange(-0.11, 0.11, 0.005) # fixed bin size

    # plt.xlim([-0.11, 0.11])
    # plt.hist(test_x, bins=bins, alpha=1, edgecolor='black', linewidth=1.2)
    # plt.title('Testing data - X coordinate')
    # plt.xlabel('X values')
    # plt.ylabel('count')
    # plt.savefig('test_x.png')
    # plt.clf()
    
    # plt.xlim([-0.11, 0.11])
    # plt.hist(test_y, bins=bins, alpha=1, edgecolor='black', linewidth=1.2)
    # plt.title('Testing data - Y coordinate')
    # plt.xlabel('Y values')
    # plt.ylabel('count')
    # plt.savefig('test_y.png')
    # plt.clf()
    
    # plt.xlim([-0.11, 0.11])
    # plt.hist(test_z, bins=bins, alpha=1, edgecolor='black', linewidth=1.2)
    # plt.title('Testing data - Z coordinate')
    # plt.xlabel('Z values')
    # plt.ylabel('count')
    # plt.savefig('test_z.png')   
    # plt.clf()
    
    # action_x = []
    # action_y = []
    # action_z = []
    # for data in train_loader:
    #     for action in data.y:            
    #         action_x.append(action[0])
    #         action_y.append(action[1])
    #         action_z.append(action[2])
                
    # # fixed bin size
    # bins = np.arange(-0.025, 0.025, 0.001) # fixed bin size

    # plt.xlim([-0.025, 0.025])
    # plt.hist(action_x, bins=bins, alpha=1, edgecolor='black', linewidth=1.2)
    # plt.title('Training data - X action')
    # plt.xlabel('X values')
    # plt.ylabel('count')
    # plt.savefig('action_x.png')
    # plt.clf()
    # print("Maximum X value", max(action_x))
    # print("Minimum X value", min(action_x))
    
    # plt.xlim([-0.025, 0.025])
    # plt.hist(action_y, bins=bins, alpha=1, edgecolor='black', linewidth=1.2)
    # plt.title('Training data - Y action')
    # plt.xlabel('Y values')
    # plt.ylabel('count')
    # plt.savefig('action_y.png')
    # plt.clf()
    # print("Maximum Y value", max(action_y))
    # print("Minimum Y value", min(action_y))
    
    # plt.xlim([-0.025, 0.025])
    # plt.hist(action_z, bins=bins, alpha=1, edgecolor='black', linewidth=1.2)
    # plt.title('Training data - Z action')
    # plt.xlabel('Z values')
    # plt.ylabel('count')
    # plt.savefig('action_z.png')   
    # plt.clf()    
    # print("Maximum Z value", max(action_z))
    # print("Minimum Z value", min(action_z))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print("Number of Parameters: ", total_params)

    # test_acc, loss = test(test_loader)
    # print(f'Test: {test_acc:.4f}, Loss: {loss:.4f}')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) #0.001 #0.0001 # 0.0005
    
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    torch.cuda.empty_cache()

    epoch_losses = []
    accuracies = []
    
    # conv_loss = np.inf
    # conv = np.inf
    # epoch = 0
    for epoch in range(50): #50 #30 15 # 75
    # while conv_loss > 0.005 and epoch < 100:
    # while conv > 0 or epoch < 20:
        train_loss = train(epoch)
        loss, translation_error = test(test_loader)
        loss = loss.detach().cpu().numpy()
        epoch_losses.append(loss)
        # accuracies.append(test_acc)
        # scheduler.step()
        print(f'Epoch: {epoch:03d}, Validation Loss: {loss:.4f}, Translation Error: {translation_error:.4f}')

        wandb.log({"validation loss": loss})
        wandb.log({"translation error": translation_error})
        wandb.watch(model)
        
        # prev_loss = conv_loss
        # conv_loss = loss
        
        # conv = prev_loss - conv_loss
        # epoch += 1
        

        # print(loss)

    # plt.plot(accuracies, label = 'accuracy')
    # plt.plot(epoch_losses, label = 'loss')
    # plt.legend(loc='upper left')
    # plt.savefig("pointnet++.png")
    # plt.show()

        torch.save(model, data+"_pnpp_posenc.pt")