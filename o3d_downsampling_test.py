import open3d as o3d
import numpy as np

print(o3d.__version__)

# 1024*1024 = 1048576
# 512*512 = 262144
# 256*256 = 65536
# 128*128 = 16384
# 64*64 = 4096


points = np.random.rand(1000000, 3)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
# o3d.visualization.draw_geometries([point_cloud])


print("Downsample the point cloud with a voxel of 0.05")

downpcd1 = point_cloud.voxel_down_sample(voxel_size=0.0155)
downpcd2 = point_cloud.voxel_down_sample(voxel_size=0.0255) #256*256
downpcd3 = point_cloud.voxel_down_sample(voxel_size=0.04) #128*128
downpcd4 = point_cloud.voxel_down_sample(voxel_size=0.065)
# o3d.visualization.draw_geometries([downpcd])

print(points.shape)
print(len(downpcd1.points))
print(len(downpcd2.points))
print(len(downpcd3.points))
print(len(downpcd4.points))