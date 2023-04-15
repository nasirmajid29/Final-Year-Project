import torch
import numpy as np

# train = torch.load("/homes/nm219/Final-Year-Project/data/reach_red_ball/processed/data_train.pt")

# test = torch.load("/homes/nm219/Final-Year-Project/data/reach_red_ball/processed/data_test.pt")

data1 = torch.load("/vol/bitbucket/nm219/data/reach_target_512size_1.pt")
data2 = torch.load("/vol/bitbucket/nm219/data/reach_target_512size_2.pt")
data3 = torch.load("/vol/bitbucket/nm219/data/reach_target_512size_3.pt")
data4 = torch.load("/vol/bitbucket/nm219/data/reach_target_512size_4.pt")
data5 = torch.load("/vol/bitbucket/nm219/data/reach_target_512size_5.pt")
data6 = torch.load("/vol/bitbucket/nm219/data/reach_target_512size_6.pt")
data7 = torch.load("/vol/bitbucket/nm219/data/reach_target_512size_7.pt")
data8 = torch.load("/vol/bitbucket/nm219/data/reach_target_512size_8.pt")
data9 = torch.load("/vol/bitbucket/nm219/data/reach_target_512size_9.pt")
data10 = torch.load("/vol/bitbucket/nm219/data/reach_target_512size_10.pt")

print(len(data1))
print(len(data2))
print(len(data3))
print(len(data4))
print(len(data5))
print(len(data6))
print(len(data7))
print(len(data8))
print(len(data9))
print(len(data10))

full_data = np.concatenate((data1, data2, data3, data4, data5, data6, data7, data8, data9, data10))

print(len(full_data))

torch.save(full_data, "/vol/bitbucket/nm219/data/reach_target_512size.pt")
      