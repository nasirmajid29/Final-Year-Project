#Inspired by https://github.com/fork123aniket/Graph-based-3D-Point-Cloud-Classification/blob/main/Graph_Point_Net.py
import torch
from torch.nn import Sequential, Linear, ReLU

from torch_cluster import fps
from torch_cluster import knn_graph

from torch_geometric.nn import PPFConv
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose, RandomRotate, SamplePoints
from torch_geometric.datasets import GeometricShapes
from torch_geometric.nn import global_max_pool
import matplotlib.pyplot as plt
import numpy as np
import wandb
from GPUtil import showUtilization as gpu_usage
from point_dataset import PointDataset
import pyvista
import open3d as o3d


degrees, axes, num_layers, downsample, lr = [180, 180, 180], [0, 1, 2], 3, True, 0.01
batch_size, num_samples, k_neighbors, sampling_ratio, n_epochs = 10, 128, 16, 0.5, 100
in_channels, hidden_channels, seed, num_shapes = 4, 32, 12345, 4

# list_of_rotations = [RandomRotate(degrees=i, axis=j) for i, j in zip(degrees, axes)]
# random_rotate = Compose(list_of_rotations)

# test_transform = Compose([
#     random_rotate,
#     SamplePoints(num=num_samples, include_normals=True),
# ])

data = 'reach_target_10eps'
wandb.init(project="Correct", entity="final-year-project", name=data)
data_loc = "/vol/bitbucket/nm219/data/"+data

pre_transform = None
transform = None

point_cloud_data_train = PointDataset(data_loc, "data.pt", True, pre_transform, transform)

point_cloud_data_test = PointDataset(data_loc, "data.pt", False, pre_transform, transform)

print(data)
print(len(point_cloud_data_train))
print(len(point_cloud_data_test))

train_loader = DataLoader(point_cloud_data_train, batch_size=64, shuffle=False, num_workers=6)
test_loader = DataLoader(point_cloud_data_test, batch_size=64, shuffle=False, num_workers=6)


class PPFNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        torch.manual_seed(seed)

        gnn_layers = []
        for layer in range(num_layers):
            if layer == 0:
                mlp = Sequential(Linear(in_channels, hidden_channels),
                                 ReLU(),
                                 Linear(hidden_channels, hidden_channels))
            else:
                mlp = Sequential(Linear(hidden_channels + in_channels, hidden_channels),
                                  ReLU(),
                                  Linear(hidden_channels, hidden_channels))
            gnn_layers.append(PPFConv(mlp))

        gnn_layers = torch.nn.ModuleList(gnn_layers)
        self.gnn_layers = gnn_layers

        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, pos, normal, batch):
        edge_index = knn_graph(pos, k=k_neighbors, batch=batch)

        for layer in range(len(self.gnn_layers)):
            if layer == 0:
                x = self.gnn_layers[layer](x=None, pos=pos, normal=normal, edge_index=edge_index)
                x = x.relu()
            else:
                if downsample:
                    index = fps(pos, batch, ratio=sampling_ratio)
                    pos, x, normal, batch = pos[index], x[index], normal[index], batch[index]
                    edge_index = knn_graph(pos, k=k_neighbors, batch=batch)

                x = self.gnn_layers[layer](x=x, pos=pos, normal=normal, edge_index=edge_index)
                x = x.relu()

        x = global_max_pool(x, batch)
        return self.classifier(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PPFNet(in_channels, hidden_channels, 3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()

total_params = sum(p.numel() for p in model.parameters())
print("Number of Parameters: ", total_params)



def train(model, optimizer, loader):
    model.train()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        pred = model(data.pos, data.normal, data.batch)
        loss = criterion(pred, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() #* data.num_graphs

    avg_loss = total_loss / len(loader.dataset)

    wandb.log({"training loss": avg_loss, "epoch": epoch})
    wandb.watch(model)
    return avg_loss

@torch.no_grad()
def test(model, loader):
    model.eval()

    total_loss = 0
    for data in loader:
        data = data.to(device)
        
        pred = model(data.pos, data.normal, data.batch)
        loss = criterion(pred, data.y)
        
        total_loss += loss 
    
    val_loss = total_loss / len(loader.dataset)    
    return val_loss


best_test_acc = 0
for epoch in range(1, n_epochs + 1):
    loss = train(model, optimizer, train_loader)
    val_loss = test(model, test_loader)
    print(f'Epoch: {epoch:02d}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')

    wandb.log({"training loss": loss, "validation loss": val_loss, "epoch": epoch})
    wandb.watch(model)

    torch.save(model, data+"_gnn.pt")


def visualize_points(pos, edge_index=None, index=None):
    fig = plt.figure(figsize=(4, 4))
    if edge_index is not None:
        for (src, dst) in edge_index.t().tolist():
             src = pos[src].tolist()
             dst = pos[dst].tolist()
             plt.plot([src[0], dst[0]], [src[1], dst[1]], linewidth=1, color='black')
    if index is None:
        plt.scatter(pos[:, 0], pos[:, 1], s=50, zorder=1000)
    else:
       mask = torch.zeros(pos.size(0), dtype=torch.bool)
       mask[index] = True
       plt.scatter(pos[~mask, 0], pos[~mask, 1], s=50, color='lightgray', zorder=1000)
       plt.scatter(pos[mask, 0], pos[mask, 1], s=50, zorder=1000)
    plt.axis('off')
    plt.show()


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, true = [], []
    for data in loader:
        logits = model(data.pos, data.normal, data.batch)
        pred = logits.argmax(dim=-1)
        values = pred == data.y
        preds.extend(pred[values].tolist())

    return preds


# preds = predict(model, test_loader)

# classes = ['2d_circle', '2d_ellipse', '2d_moon', '2d_pacman', '2d_plane', '2d_semicircle',
#            '2d_trapezoid', '2d_triangle', '3d_chimney_3', '3d_chimney_4', '3d_cone', '3d_cube',
#            '3d_cup', '3d_cylinder', '3d_dome', '3d_hexagon', '3d_icecream', '3d_ico', '3d_ico2',
#            '3d_L_cylinder',  '3d_monkey', '3d_moon', '3d_pacman', '3d_pentagon', '3d_pill',
#            '3d_pipe', '3d_pyramid_3_asy', '3d_pyramid_3_asym', '3d_pyramid_3_asym2',
#            '3d_pyramid_4_asym', '3d_pyramid_4_asym2', '3d_pyramid_4_sym', '3d_rotated_cube',
#            '3d_rotated_hexagon', '3d_sphere', '3d_torus', '3d_torus_fat', '3d_U_cylinder',
#            '3d_wedge', '3d_wedge_long']

# shape = np.random.choice(preds, num_shapes, replace=False)
# print(f'class(es) chosen: {[classes[fig] for fig in shape]}')

# for fig in shape:
#     point_data = test_dataset[fig]
#     if not downsample:
#         print(f'Position of points in the point cloud ({classes[fig]})')
#         visualize_points(point_data.pos)
#         point_data.edge_index = knn_graph(point_data.pos, k=k_neighbors)
#         print(f'Generated dynamic Graph of the point cloud ({classes[fig]})')
#         visualize_points(point_data.pos, edge_index=point_data.edge_index)
#     else:
#         print(f'Position of points in the point cloud ({classes[fig]})')
#         visualize_points(point_data.pos)
#         index = fps(point_data.pos, ratio=sampling_ratio)
#         print(f'Farthest points sampled in the point cloud ({classes[fig]})')
#         visualize_points(point_data.pos, index=index)
#         point_data.edge_index = knn_graph(point_data.pos[index], k=k_neighbors)
#         print(f'Generated dynamic Graph of the point cloud ({classes[fig]})')
#         visualize_points(point_data.pos[index], edge_index=point_data.edge_index)
#     print(f'pred_label: {classes[fig]} and true_label: {classes[fig]}\n')