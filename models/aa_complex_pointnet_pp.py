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
from aa_complex_point_dataset import AngleAxisComplexPointDataset
import numpy as np

import wandb
import pyvista
import matplotlib.pyplot as plt
import os
import open3d as o3d
import math

data = 'reach_target_200eps'
wandb.init(project="Angle Axis", entity="final-year-project", name=data)

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
        self.sa1_module = SAModule(0.5, 0.2, MLP([6, 16, 32]))
        self.sa2_module = SAModule(0.05, 0.1, MLP([32 + 3, 64, 128]))
        self.sa3_module = GlobalSAModule(MLP([128 + 3, 256, 512]))

        self.mlp = MLP([512, 128, 32, 8], dropout=0.0, norm=None)

    def forward(self, data):
        # print(data.dtype)
        sa0_out = (data.x, data.pos, data.batch)
        # print(np.array(data.batch).shape)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

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
    print(f'Epoch: {epoch:03d}, Average Training Loss: {avg_loss:.4f}')

    wandb.log({"training loss": avg_loss})
    wandb.watch(model)
        

def test(loader):
    model.eval()

    total_loss = 0
    total_translation_error = 0
    total_rotational_error = 0
    gripper_correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            
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
                total_translation_error += cm_dist
                
                aa1 = pred[i][3:7]
                aa2 = data.y[i][3:7]
                
                q1 = o3d.geometry.Quaternion()
                q1.set_rotation_vector(aa1)
                
                q2 = o3d.geometry.Quaternion()
                q2.set_rotation_vector(aa2)
                
                # Calculate the angle between the quaternions
                dot_product = q1.dot(q2)
                angle_in_radians = 2 * math.acos(abs(dot_product))
                angle_in_degrees = math.degrees(angle_in_radians)
                
                total_rotational_error += angle_in_degrees
                
                pred[i][-1] = 0 if pred[i][-1] < 0.5 else 1
                if pred[i][-1] == data.y[i][-1]:
                    gripper_correct += 1
            
    val_loss = total_loss / len(loader)
    translation_error = total_translation_error / len(loader)
    rotational_error = total_rotational_error /len(loader)
    gripper_percentage = gripper_correct / len(loader)
    return val_loss, translation_error, rotational_error, gripper_percentage


if __name__ == '__main__':

    print(torch.cuda.is_available())
    gpu_usage()
    torch.cuda.empty_cache()
    gpu_usage()
    
    data_loc = "/vol/bitbucket/nm219/data/"+data
    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..',
    #                 'data/ModelNet10')

    pre_transform = None #T.NormalizeScale()
    # transform = T.SamplePoints(64)
    
    point_cloud_data_train = AngleAxisComplexPointDataset(data_loc, "data.pt", True, pre_transform)
    
    point_cloud_data_test = AngleAxisComplexPointDataset(data_loc, "data.pt", False, pre_transform)
    
    print(len(point_cloud_data_train))
    print(len(point_cloud_data_test))
    
    
    # train_dataset = ModelNet(path, '10', True, transform, pre_transform)
    # test_dataset = ModelNet(path, '10', False, transform, pre_transform)

    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
    #                           num_workers=6)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
    #                          num_workers=6)


    train_loader = DataLoader(point_cloud_data_train, batch_size=64, shuffle=False, num_workers=6) #shuffle true #batch 32
    test_loader = DataLoader(point_cloud_data_test, batch_size=64, shuffle=False,num_workers=6)
    
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
    


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print("Number of Parameters: ", total_params)
    
    # test_acc, loss = test(test_loader)
    # print(f'Test: {test_acc:.4f}, Loss: {loss:.4f}')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005) #0.001

    torch.cuda.empty_cache()

    epoch_losses = []
    accuracies = []
    for epoch in range(50): #201
        train(epoch)
        loss, translation_error, rotation_error, gripper_correct = test(test_loader)
        loss = loss.detach().cpu().numpy()
        epoch_losses.append(loss)
        # accuracies.append(test_acc)
        print(f'Epoch: {epoch:03d}, Validation Loss: {loss:.4f}, Gripper Correctness: {gripper_correct:.4f}')
        print(f'Translation Error: {translation_error:.4f}, Rotation Error: {rotation_error:.4f}')

        wandb.log({"validation loss": loss, "translation error": translation_error, 
                   "rotation_error": rotation_error, "gripper_correct": gripper_correct})
        wandb.watch(model)

        # print(loss)

    # plt.plot(accuracies, label = 'accuracy')
    # plt.plot(epoch_losses, label = 'loss')
    # plt.legend(loc='upper left')
    # plt.savefig("pointnet++.png")
    # plt.show()

        torch.save(model, data+"aa_pnpp.pt")