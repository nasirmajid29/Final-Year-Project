import torch
from torch.utils.data import Dataset

class PointCloudDataset(Dataset):

    def __init__(self, torch_file, transform=None):
        
            self.point_cloud, self.action = torch.load(torch_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        point_cloud = self.point_cloud[idx]
        action = self.action[idx]
        sample = {"Point cloud": point_cloud, "Action": action}
        return sample
        