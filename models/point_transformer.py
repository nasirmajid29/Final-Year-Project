#https://github.com/pyg-team/pytorch_geometric/blob/master/examples/point_transformer_classification.py

import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    MLP,
    PointTransformerConv,
    fps,
    global_mean_pool,
    knn,
    knn_graph,
)
from torch_geometric.utils import scatter
from point_dataset import PointDataset
import wandb
import math
import open3d as o3d

data = 'reach_target_100eps'
wandb.init(project="Architectures", entity="final-year-project", name=data+"_transformer")
data_loc = "/vol/bitbucket/nm219/data/"+data
pre_transform = None 
transform = None
    
point_cloud_data_train = PointDataset(data_loc, "data.pt", True, pre_transform, transform)
point_cloud_data_test = PointDataset(data_loc, "data.pt", False, pre_transform, transform)

train_loader = DataLoader(point_cloud_data_train, batch_size=64, shuffle=False, num_workers=6)
test_loader = DataLoader(point_cloud_data_test, batch_size=64, shuffle=False, num_workers=6)


class TransformerBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin_in = Lin(in_channels, in_channels)
        self.lin_out = Lin(out_channels, out_channels)

        self.pos_nn = MLP([3, 64, out_channels], norm=None, plain_last=False)

        self.attn_nn = MLP([out_channels, 64, out_channels], norm=None,
                           plain_last=False)

        self.transformer = PointTransformerConv(in_channels, out_channels,
                                                pos_nn=self.pos_nn,
                                                attn_nn=self.attn_nn)

    def forward(self, x, pos, edge_index):
        x = self.lin_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.lin_out(x).relu()
        return x


class TransitionDown(torch.nn.Module):
    '''
        Samples the input point cloud by a ratio percentage to reduce
        cardinality and uses an mlp to augment features dimensionnality
    '''
    def __init__(self, in_channels, out_channels, ratio=0.25, k=16):
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)

        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch,
                            batch_y=sub_batch)

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out = scatter(x[id_k_neighbor[1]], id_k_neighbor[0], dim=0,
                        dim_size=id_clusters.size(0), reduce='max')

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim_model, k=16):
        super().__init__()
        self.k = k

        # dummy feature is created if there is none given
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)

        self.transformer_input = TransformerBlock(in_channels=dim_model[0],
                                                  out_channels=dim_model[0])
        # backbone layers
        self.transformers_down = torch.nn.ModuleList()
        self.transition_down = torch.nn.ModuleList()

        for i in range(len(dim_model) - 1):
            # Add Transition Down block followed by a Transformer block
            self.transition_down.append(
                TransitionDown(in_channels=dim_model[i],
                               out_channels=dim_model[i + 1], k=self.k))

            self.transformers_down.append(
                TransformerBlock(in_channels=dim_model[i + 1],
                                 out_channels=dim_model[i + 1]))

        # class score computation
        self.mlp_output = MLP([dim_model[-1], 64, out_channels], norm=None)

    def forward(self, x, pos, batch=None):

        # add dummy features in case there is none
        if x is None:
            x = torch.ones((pos.shape[0], 1), device=pos.get_device())

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # backbone
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)

            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

        # GlobalAveragePooling
        x = global_mean_pool(x, batch)

        # Class score
        out = self.mlp_output(x)

        return out


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.pos, data.batch)
        loss = F.mse_loss(out, data.y)
        loss.backward()
        total_loss += loss.item() # * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)



@torch.no_grad()
def test(loader):
    model.eval()

    total_loss = 0
    total_translation_error = 0
    total_rotational_error = 0
    gripper_correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.pos, data.batch)
        loss = F.mse_loss(pred, data.y)
        
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
        
            quat1 = pred[i][3:7]
            quat2 = data.y[i][3:7]
            
            q1 = np.array(quat1)
            q2 = np.array(quat2)
            
            q1 /= np.linalg.norm(q1)
            q2 /= np.linalg.norm(q2)
            
            dot_product = np.dot(q1, q2)
            angle_in_radians = 2 * np.arccos(abs(dot_product))
            angle_in_degrees = math.degrees(angle_in_radians)
            
            total_rotational_error += angle_in_degrees
            
            pred[i][-1] = 0 if pred[i][-1] < 0.5 else 1
            if pred[i][-1] == data.y[i][-1]:
                gripper_correct += 1
            
    val_loss = total_loss / len(loader.dataset)    
    translation_error = total_translation_error / len(loader)
    rotational_error = total_rotational_error /len(loader)
    gripper_percentage = gripper_correct / len(loader)
    return val_loss, translation_error, rotational_error, gripper_percentage
    
    val_loss = total_loss / len(loader.dataset)    
    return val_loss

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(3, 3, dim_model=[32, 64, 128, 256], k=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


    total_params = sum(p.numel() for p in model.parameters())
    print("Number of Parameters: ", total_params)

    for epoch in range(30):
        loss = train()
        val_loss = test(test_loader)
        print(f'Epoch {epoch:03d}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}')
        scheduler.step()

        wandb.log({"training loss": loss, "validation loss": val_loss, "epoch": epoch})
        wandb.watch(model)


        torch.save(model, data+"_transformer.pt")