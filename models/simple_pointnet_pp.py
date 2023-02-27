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

wandb.init(project="test-project", entity="final-year-project")

wandb.config.update({
  "learning_rate": 0.001,
  "optimiser": "Adam",
  "epochs": 30,
  "batch_size": 32,
  "ratio": [0.25, 0.125],
  "radius": [0.1, 0.2]
})

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
        self.sa1_module = SAModule(0.25, 0.1, MLP([9, 32, 64]))
        self.sa2_module = SAModule(0.125, 0.2, MLP([64 + 3, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512]))

        self.mlp = MLP([512, 256, 32, 3], dropout=0.5, norm=None)

    def forward(self, data):
        # print(data.dtype)
        sa0_out = (data.x, data.pos, data.batch)
        # print(np.array(data.batch).shape)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        # print(x.dtype)

        return self.mlp(x)#.log_softmax(dim=-1)


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
        break


    avg_loss = total_loss / len(train_loader)
    print(f'Epoch: {epoch:03d}, Average Training Loss: {avg_loss:.4f}')

    wandb.log({"training loss": avg_loss})
    wandb.watch(model)
        

def test(loader):
    model.eval()

    # correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
        #     predmax = model(data).max(1)[1]
        # correct += predmax.eq(data.y).sum().item()
    return F.mse_loss(pred, data.y) #, correct / len(loader.dataset),  
        


if __name__ == '__main__':

    gpu_usage()
    torch.cuda.empty_cache()
    gpu_usage()
    

    # path = osp.join(osp.dirname(osp.realpath(__file__)), '..',
    #                 'data/ModelNet10')

    pre_transform = T.NormalizeScale()
    # transform = T.SamplePoints(64)
    
    point_cloud_data_train = PointDataset("data/reach_red_ball", "data.pt", True, pre_transform)
    
    point_cloud_data_test = PointDataset("data/reach_red_ball", "data.pt", False, pre_transform)
    
    
    # train_dataset = ModelNet(path, '10', True, transform, pre_transform)
    # test_dataset = ModelNet(path, '10', False, transform, pre_transform)

    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
    #                           num_workers=6)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
    #                          num_workers=6)


    train_loader = DataLoader(point_cloud_data_train, batch_size=32, shuffle=True, num_workers=6)
    test_loader = DataLoader(point_cloud_data_test, batch_size=32, shuffle=False,num_workers=6)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net().to(device)

    # test_acc, loss = test(test_loader)
    # print(f'Test: {test_acc:.4f}, Loss: {loss:.4f}')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #0.001

    torch.cuda.empty_cache()

    epoch_losses = []
    accuracies = []
    for epoch in range(30): #201
        train(epoch)
        loss = test(test_loader)
        loss = loss.detach().cpu().numpy()
        epoch_losses.append(loss)
        # accuracies.append(test_acc)
        print(f'Epoch: {epoch:03d}, Validation Loss: {loss:.4f}')

        wandb.log({"validation loss": loss})
        wandb.watch(model)

        # print(loss)

    # plt.plot(accuracies, label = 'accuracy')
    # plt.plot(epoch_losses, label = 'loss')
    # plt.legend(loc='upper left')
    # plt.savefig("pointnet++.png")
    # plt.show()

    torch.save(model, "pointnet++.pt")