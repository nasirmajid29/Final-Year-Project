# import torch
# from torch.nn import Sequential, Linear, ReLU
# from torch_geometric.nn import GCNConv, global_mean_pool
# from torch_geometric.datasets import ModelNet

# from point_dataset import PointDataset
# import torch_geometric.transforms as T
# from torch_geometric.loader import DataLoader

# # Define a graph convolutional neural network
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = GCNConv(3, 64)
#         self.conv2 = GCNConv(64, 64)
#         self.conv3 = GCNConv(64, 64)
#         self.fc1 = Linear(64, 32)
#         self.fc2 = Linear(32, 3)

#     def forward(self, x, edge_index, batch):
#         x = self.conv1(x, edge_index)
#         x = x.relu()
#         x = self.conv2(x, edge_index)
#         x = x.relu()
#         x = self.conv3(x, edge_index)
#         x = x.relu()
#         x = global_mean_pool(x, batch)
#         x = self.fc1(x)
#         x = x.relu()
#         x = self.fc2(x)
#         return x

# data_loc = "/vol/bitbucket/nm219/data/reach_target_10eps"

# pre_transform = None # T.NormalizeScale()

# point_cloud_data_train = PointDataset(data_loc, "data.pt", True, pre_transform)
# point_cloud_data_test = PointDataset(data_loc, "data.pt", False, pre_transform)
    
# train_loader = DataLoader(point_cloud_data_train, batch_size=64, shuffle=False, num_workers=6) #shuffle true #batch 32
# test_loader = DataLoader(point_cloud_data_test, batch_size=64, shuffle=False,num_workers=6)

# # Initialize the model, optimizer, and loss function
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net().to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = torch.nn.MSELoss()

# # Train the model
# for epoch in range(30):
#     model.train()
#     for batch in train_loader:
#         batch = batch.to(device)
#         optimizer.zero_grad()
#         pred = model(batch.x, batch.edge_index, batch.batch)
#         loss = criterion(pred, batch.y)
#         loss.backward()
#         optimizer.step()

#     model.eval()
#     with torch.no_grad():
#         val_loss = 0
#         correct = 0
#         for batch in test_loader:
#             batch = batch.to(device)
#             pred = model(batch.x, batch.edge_index, batch.batch)
#             val_loss += criterion(pred, batch.y).item() * batch.num_graphs
#             pred = pred.argmax(dim=1)
#             correct += pred.eq(batch.y).sum().item()
#         val_loss /= len(val_dataset)
#         acc = correct / len(val_dataset)
#         print(f'Epoch {epoch}, Val Loss: {val_loss:.4f}, Val Acc: {acc:.4f}')


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.optim as optim
from point_dataset import PointDataset
from torch_geometric.loader import DataLoader

class PointCloudGNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(PointCloudGNN, self).__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Perform graph convolution
        x = self.lin(x)
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        x = self.propagate(edge_index, x=x, norm=norm)

        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class PointCloudModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(PointCloudModel, self).__init__()
        self.gnn = PointCloudGNN(6, 64)
        self.lin = nn.Linear(64, 3)

    def forward(self, x, edge_index):
        x = self.gnn(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)
        return x

data_loc = "/vol/bitbucket/nm219/data/reach_target_10eps"

pre_transform = None # T.NormalizeScale()

point_cloud_data_train = PointDataset(data_loc, "data.pt", True, pre_transform)
point_cloud_data_test = PointDataset(data_loc, "data.pt", False, pre_transform)
    
train_loader = DataLoader(point_cloud_data_train, batch_size=64, shuffle=False, num_workers=6) #shuffle true #batch 32
test_loader = DataLoader(point_cloud_data_test, batch_size=64, shuffle=False,num_workers=6)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PointCloudModel(in_channels=6, hidden_channels=64, out_channels=3).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

def train(model, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch.num_graphs

    train_loss /= len(train_loader.dataset)
    return train_loss

def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            test_loss += loss.item() * batch.num_graphs

    test_loss /= len(test_loader.dataset)
    return test_loss

num_epochs = 10

for epoch in range(1, num_epochs + 1):
    train_loss = train(model, train_loader, optimizer, criterion)
    test_loss = test(model, test_loader, criterion)
    print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
