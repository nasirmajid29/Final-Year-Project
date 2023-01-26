import torch
from torch.utils.data import Dataset

class PointDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = torch.load(dataset_path)
        self.point_clouds = self.data[:, 0]
        self.actions = self.data[:, 1]

  def __len__(self):
        return len(self.data)

  def __getitem__(self, index):
    
        # Select sample
        pc = self.point_clouds[index]
        action = self.actions[index]

        return pc, action