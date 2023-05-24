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
from scipy.spatial.transform import Rotation as R

def rotation_matrix_to_angle_axis(rot_matrix):
      
      rotation = R.from_matrix(rotation_matrix)

      angle = rotation.magnitude()
      axis = rotation.as_rotvec()
      
      
      angle_axis = np.concatenate((axis, [angle]))

      return angle_axis

class AngleAxisComplexPointDataset(Dataset):
        
      def __init__(self, root, filename, train=True, pre_transform=None, transform=None):
            """
            root = Where the dataset should be stored. This folder is split
            into raw_dir (downloaded dataset) and processed_dir (processed data). 
            """
            self.filename = filename
            self.train = train
            self.data = torch.load(f"{root}/raw/{filename}")
            super(AngleAxisComplexPointDataset, self).__init__(root, transform, pre_transform)

            
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

                  action_translate = action[:3, 3].view(1,3)
                  matrix = action[:3,:3].view(1,9)
                  action_rotate = rotation_matrix_to_angle_axis(matrix)
                  gripper_state = action[3,0].reshape(1)
                  
                  action = np.concatenate((action_translate, action_rotate, gripper_state))
                  # print(action.size())
                  action = action.view(1,8)
                  
                  normalise = True # True
                  unnormalise = False # True
                  
                  max_x = 0.01
                  min_x = -0.01
                  range_x = max_x - min_x
                  
                  max_y = 0.02
                  min_y = -0.02
                  range_y = max_y - min_y
                  
                  max_z = 0.025
                  min_z = -0.005
                  range_z = max_z - min_z
                  if normalise:
    

                        action[0][0] = 2*((action[0][0] - min_x) / range_x) - 1
                        action[0][1] = 2*((action[0][1] - min_y) / range_y) - 1
                        action[0][2] = 2*((action[0][2] - min_z) / range_z) - 1

                  if unnormalise:

                        action[0][0] = (range_x * (action[0][0] + 1)/2) + min_x    
                        action[0][1] = (range_y * (action[0][1] + 1)/2) + min_y    
                        action[0][2] = (range_z * (action[0][2] + 1)/2) + min_z    


                  downsample = True
                  if downsample:
                        
                        print("Point cloud before: ", point_cloud.shape)
                        o3d_pc = o3d.geometry.PointCloud()
                        o3d_pc.points = o3d.utility.Vector3dVector(point_cloud[:, :3].numpy())
                        o3d_pc.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:].numpy())
                        
                  
                        voxel_size = 0.05  # 0.1 0.05 0.025 0.01 0.005 
                        downsampled_o3d_pc = o3d_pc.voxel_down_sample(voxel_size)
                        
                        new_points = torch.tensor(downsampled_o3d_pc.points, dtype=torch.float32)
                        new_colours = torch.tensor(downsampled_o3d_pc.colors, dtype=torch.float32)
                        
                        point_cloud = torch.cat((new_points, new_colours), dim=1)
                        
                        print("Point cloud after: ", point_cloud.shape)
                  
                  fixed_sample = False
                  # fixed point sampling
                  if fixed_sample:
                        point_cloud = point_cloud[torch.randperm(point_cloud.size(0))[:2]]
                        
                  pc = o3d.geometry.PointCloud()
                  pc.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
                  pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                  normals = torch.Tensor(pc.normals)                
                  
                  data = Data(x = point_cloud[:, 3:], y = action, pos = point_cloud[:, :3], normal = normals)

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
