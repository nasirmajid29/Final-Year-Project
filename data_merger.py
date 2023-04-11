import torch
import numpy as np

# train = torch.load("/homes/nm219/Final-Year-Project/data/reach_red_ball/processed/data_train.pt")

# test = torch.load("/homes/nm219/Final-Year-Project/data/reach_red_ball/processed/data_test.pt")

data1 = torch.load("/vol/bitbucket/nm219/data/reach_target_256size_1.pt")
data2 = torch.load("/vol/bitbucket/nm219/data/reach_target_256size_2.pt")
data3 = torch.load("/vol/bitbucket/nm219/data/reach_target_256size_3.pt")
data4 = torch.load("/vol/bitbucket/nm219/data/reach_target_256size_4.pt")

print(len(data1))
print(len(data2))
print(len(data3))
print(len(data4))

full_data = np.concatenate((data1, data2, data3, data4))

print(len(full_data))

torch.save(full_data, "/vol/bitbucket/nm219/data/reach_target_256size.pt")
      