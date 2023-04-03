import glob
import os
import torch
from torch.utils.data import random_split
import numpy as np


import glob
import os
import os.path as osp
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from torch_geometric.data import Data, Dataset
   


# class PointDataset(Dataset):
      # def __init__(self, dataset_path, pre_transform=None, transform=None):
      #       self.dataset_path = dataset_path
      #       self.pre_transform = pre_transform
      #       self.transform = transform

      #       self.data = torch.load(dataset_path)

      #       self.point_clouds = self.data[:, 0]
      #       self.actions = self.data[:, 1]

      #       print(np.array(self.point_clouds).shape)
      #       print(np.array(self.actions).shape)

      # def __len__(self):
      #       return len(self.data)

      # def __getitem__(self, index):
      #       point_cloud = Data(torch.tensor(self.point_clouds[index]))
      #       action = torch.tensor(self.actions[index])
      #   # Select sample

      #       if self.pre_transform is not None:
      #             point_cloud = self.pre_transform(point_cloud)
            
      #       if self.transform is not None:
      #             point_cloud = self.transform(point_cloud)
      
      #       return point_cloud, action




     
#       def __init__(
#         self,
#         dataset_path: str,
#       #   name: str = '10',
#         train: bool = True,
#         transform: Optional[Callable] = None,
#         pre_transform: Optional[Callable] = None,
#         pre_filter: Optional[Callable] = None,
#     ):
#             # assert name in ['10', '40']
#             # self.name = name
#             super().__init__(dataset_path, transform, pre_transform, pre_filter)
#             path = self.processed_paths[0] if train else self.processed_paths[1]
#             self.data = torch.tensor(torch.load(path))
#             self.size = len(self.data)
#             # self.point_clouds = self.data[:, 0]
#             # self.actions = self.data[:, 1]
            





#       def process_set(self) -> Tuple[Data, Dict[str, Tensor]]:
 
 
#             if self.pre_filter is not None:
#                   self.data = [d for d in self.data if self.pre_filter(d)]

#             if self.pre_transform is not None:
#                   self.data = [self.pre_transform(d) for d in self.data]

            
#             if self.transform is not None:
#                   self.data = [self.transform(d) for d in self.data]

#             train, test = random_split(self.data, [0.8, 0.2])
#             return self.collate(train), self.collate(test)

#       def __repr__(self) -> str:
#             return f'{self.__class__.__name__}{self.name}({len(self)})'




class PointDataset(Dataset):
        
      def __init__(self, root, filename, train=True, pre_transform=None, transform=None):
            """
            root = Where the dataset should be stored. This folder is split
            into raw_dir (downloaded dataset) and processed_dir (processed data). 
            """
            self.filename = filename
            self.train = train
            self.data = torch.load(f"{root}/raw/{filename}")
            super(PointDataset, self).__init__(root, transform, pre_transform)

            
      @property
      def raw_file_names(self):
            """ If this file exists in raw_dir, the download is not triggered.
                  (The download func. is not implemented here)  
            """
            return ['example.pt']

      @property
      def processed_file_names(self):
            """ If these files are found in raw_dir, processing is skipped"""

            return ['data_train.pt', 'data_test.pt']

      def process(self):
            
            data_list = []

            for state_action_pair in self.data:
                  point_cloud, action = torch.tensor(state_action_pair[0], dtype=torch.float32), torch.tensor(state_action_pair[1], dtype=torch.float32)

                  action = action[:3, 3].view(1,3)
                  # print(action.size())
                  
                  data = Data(x = point_cloud, y = action)
                  data.pos = point_cloud[:, :3]

                  # print(data.x.size())
                  # print(data.y.size())
                  # print(data.pos.size())

                  data_list.append(data)

            if self.pre_transform is not None:
                  data_list = [self.pre_transform(d) for d in data_list]

            if self.transform is not None:
                  data_list = [self.transform(d) for d in data_list] 

        
            train_data, test_data = random_split(data_list, [0.8, 0.2])

            # if self.train:
            torch.save(train_data, f'{self.root}/processed/data_train.pt')
                  # return train_data
            # else:
            torch.save(test_data, f'{self.root}/processed/data_test.pt')
                  # return test_data

            return train_data if self.train else test_data #data_list

      # def _get_labels(self, label):
      #       label = np.asarray([label])
      #       return torch.tensor(label, dtype=torch.int64)

      def __len__(self):
            # return self.data.shape[0]
            if self.train:
                  data = torch.load(f'{self.processed_dir}/data_train.pt')
            else:
                  data = torch.load(f'{self.processed_dir}/data_test.pt')
            return len(data)
      
      

      def __getitem__(self, idx):
            """ - Equivalent to __getitem__ in pytorch
                  - Is not needed for PyG's InMemoryDataset
            """
            if self.train:
                  data = torch.load(f'{self.root}/processed/data_train.pt')
            else:
                  data = torch.load(f'{self.root}/processed/data_test.pt')

            return data[idx]


# class PointDataset(Dataset):
#     '''
#     Structures class for datasets that do not fit into memory.
#     '''
#     def __init__(self, root='./data/example', pre_transform=None, transform=None):
#         super(PointDataset, self).__init__(root, transform, pre_transform)
#         self.has_nan = []
#       #   self.device = torch.device('c')

#     @property
#     def raw_file_names(self):
#         return ['example.pt']

#     @property
#     def processed_file_names(self):
#         return ['data_train.pt', 'data_test.pt'] 

#     def download(self):
#         pass

#     def process(self):
#       #   from utils import has_nan

#         i = 0
#         for raw_path in self.raw_paths:
#             data = torch.load(raw_path)

#             # if self.pre_filter is not None:
#             #     if max(torch.isnan(data.shape_index)):
#             #         self.has_nan.append(i)
#             #         continue

#             if self.pre_transform is not None:
#                 data = self.pre_transform(data)

#             torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))
#             i += 1

#     def len(self):
#         return len(self.processed_file_names)

#     def get(self, idx):
#         data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
#         return data