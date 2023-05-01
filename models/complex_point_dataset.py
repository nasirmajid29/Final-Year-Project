import glob
import os
import torch
from torch.utils.data import random_split
import numpy as np


import glob
import os
import torch
from torch_geometric.data import Data, Dataset

import numpy as np

def rotation_matrix_quaternion(rot_matrix):
    """
    Convert a 3x3 rotation matrix to a quaternion.
    Input
    :param rot_matrix: A 3x3 element matrix representing the full 3D rotation matrix. 
                       This rotation matrix converts a point in the local reference 
                       frame to a point in the global reference frame.
    Output
    :return: A 4 element array representing the quaternion (q0,q1,q2,q3)
    """
    # Extract the values from the rotation matrix
    r00, r01, r02 = rot_matrix[0]
    r10, r11, r12 = rot_matrix[1]
    r20, r21, r22 = rot_matrix[2]
    
    # Calculate the trace of the rotation matrix
    trace = r00 + r11 + r22
    
    # Calculate the quaternion components based on the trace of the rotation matrix
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2.0
        q0 = 0.25 * S
        q1 = (r21 - r12) / S
        q2 = (r02 - r20) / S
        q3 = (r10 - r01) / S
    elif (r00 > r11) and (r00 > r22):
        S = np.sqrt(1.0 + r00 - r11 - r22) * 2.0
        q0 = (r21 - r12) / S
        q1 = 0.25 * S
        q2 = (r01 + r10) / S
        q3 = (r02 + r20) / S
    elif r11 > r22:
        S = np.sqrt(1.0 + r11 - r00 - r22) * 2.0
        q0 = (r02 - r20) / S
        q1 = (r01 + r10) / S
        q2 = 0.25 * S
        q3 = (r12 + r21) / S
    else:
        S = np.sqrt(1.0 + r22 - r00 - r11) * 2.0
        q0 = (r10 - r01) / S
        q1 = (r02 + r20) / S
        q2 = (r12 + r21) / S
        q3 = 0.25 * S
    
    # Return the quaternion as a numpy array
    quaternion = torch.Tensor([q0, q1, q2, q3])
    return quaternion




class ComplexPointDataset(Dataset):
        
      def __init__(self, root, filename, train=True, pre_transform=None, transform=None):
            """
            root = Where the dataset should be stored. This folder is split
            into raw_dir (downloaded dataset) and processed_dir (processed data). 
            """
            self.filename = filename
            self.train = train
            self.data = torch.load(f"{root}/raw/{filename}")
            super(ComplexPointDataset, self).__init__(root, transform, pre_transform)

            
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

                  action_translate = action[:3, 3].view(-1) #(1,3)
                  matrix = action[:3,:3]
                  action_rotate = rotation_matrix_quaternion(matrix)
                  gripper_state = action[3,0].reshape(1)
                  
                  action = torch.cat((action_translate, action_rotate, gripper_state), dim=0)
                  # print(action.size())
                  action = action.view(1,8)
                  
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
