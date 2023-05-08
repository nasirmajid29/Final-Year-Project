import os
import pickle
import numpy as np

import torch

from PIL import Image

from rlbench.utils import get_stored_demos, ObservationConfig, _resize_if_needed
from rlbench.backend.utils import image_to_float_array, rgb_handles_to_mask
from pyrep.objects import VisionSensor

from visualise import visualise_pc, visualise_pc_rgb

# def rgb_depth_to_pc(colour_path, depth_path):
#     depth = o3d.io.read_image(depth_path)
#     colour = o3d.io.read_image(colour_path)


#     rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(colour, depth, convert_rgb_to_intensity = False)
#     camera = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
#     pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera)

#     # Flip it, otherwise the pointcloud will be upside down
#     pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
#     # o3d.visualization.draw_geometries([pcd])



#     points = np.asarray(pcd.points)
#     colours = np.asarray(pcd.colors)

#     return points, colours



# def generate_pc_for_episode(episode_size, episode_path, camera_type):

#     all_points = []
#     all_colours = []

#     for i in range(episode_size):
#         depth_path = episode_path+"/wrist_depth/" +i+".png"
#         rgb_path = episode_path+"/wrist_rgb/" +i+".png"
        
#         points, colours = rgb_depth_to_pc(rgb_path, depth_path)
#         all_points.append(points)
#         all_colours.append(colours)
#     return all_points, all_colours


# def generate_episode_point_clouds(episode_path):

#     full_pc_points = []
#     full_pc_colours = []

#     for file in os.listdir(episode_path+"/front_depth"):

#         front_depth = episode_path+"/front_depth/" + file
#         front_rgb = episode_path+"/front_rgb/" + file
#         front_points, front_colours = rgb_depth_to_pc(front_rgb, front_depth)
    
#         left_depth = episode_path+"/left_shoulder_depth/" + file
#         left_rgb = episode_path+"/left_shoulder_rgb/" + file
#         left_points, left_colours = rgb_depth_to_pc(left_rgb, left_depth)
#         full_points = np.concatenate((front_points, left_points), axis=0)
#         full_colours = np.concatenate((front_colours, left_colours), axis=0)

#         right_depth = episode_path+"/right_shoulder_depth/" + file
#         right_rgb = episode_path+"/right_shoulder_rgb/" + file
#         points, colours = rgb_depth_to_pc(right_rgb, right_depth)
#         full_points = np.concatenate((full_points, points), axis=0)
#         full_colours = np.concatenate((full_colours, colours), axis=0)
        
#         over_depth = episode_path+"/overhead_depth/" + file
#         over_rgb = episode_path+"/overhead_rgb/" + file
#         points, colours = rgb_depth_to_pc(over_rgb, over_depth)
#         full_points = np.concatenate((full_points, points), axis=0)
#         full_colours = np.concatenate((full_colours, colours), axis=0)
        
#         wrist_depth = episode_path+"/wrist_depth/" + file
#         wrist_rgb = episode_path+"/wrist_rgb/" + file
#         points, colours = rgb_depth_to_pc(wrist_rgb, wrist_depth)
#         full_points = np.concatenate((full_points, points), axis=0)
#         full_colours = np.concatenate((full_colours, colours), axis=0)

#         full_pc_points.append(full_points)
#         full_pc_colours.append(full_colours)

#         # print(len(full_pc_points))
    
#     return np.asarray(full_pc_points), np.asarray(full_pc_colours) 
        

# wrist_depth = "/home/nasir/Desktop/Demos/reach_target/variation0/episodes/episode0/wrist_depth/0.png" 
# wrist_colour = "/home/nasir/Desktop/Demos/reach_target/variation0/episodes/episode0/wrist_rgb/0.png"

# points, colour = rgb_depth_to_pc(wrist_colour, wrist_depth)
# visualise_pc(points, colour)

episode_path = "/home/nasir/Desktop/Demos/reach_target/variation0/episodes/episode0" 

# points, colour = generate_episode_point_clouds(episode_path)
# # print(points[0].shape)
# visualise_pc(points[0], colour[0])

# utils.get_stored_demos

def get_config(img_size):
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size

    # Store depth as 0 - 1
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    # We want to save the masks as rgb encodings.
    obs_config.left_shoulder_camera.masks_as_one_channel = False
    obs_config.right_shoulder_camera.masks_as_one_channel = False
    obs_config.overhead_camera.masks_as_one_channel = False
    obs_config.wrist_camera.masks_as_one_channel = False
    obs_config.front_camera.masks_as_one_channel = False

    
    # obs_config.left_shoulder_camera.point_cloud = True
    # obs_config.right_shoulder_camera.point_cloud = True
    # obs_config.overhead_camera.point_cloud = True
    # obs_config.wrist_camera.point_cloud = True
    # obs_config.front_camera.point_cloud = True


    return obs_config


def depth_to_pc(obs_config, task_name):

    # (amount: int, image_paths: bool, dataset_root: str,
    #                  variation_number: int, task_name: str,
    #                  obs_config: ObservationConfig,
    #                  random_selection: bool = True,
    #                  from_episode_number: int = 0) -> List[Demo]:
    
    # personal path "/home/nasir/Desktop/Demos"
    # lab path "/vol/bitbucket/nm219/Demos"
    # amount -1
    
    demos = get_stored_demos(-1, False, "/vol/bitbucket/nm219/Demos", 0, task_name, obs_config, random_selection=False)#, from_episode_number=90)
    return demos

def transform_between_frames(frame1, frame2):
    inverse = np.linalg.inv(frame1)
    return np.matmul(inverse, frame2)

def transform_point_cloud(transform, point_cloud)-> np.ndarray:

    num_points = point_cloud.shape[0]
    homogeneous_column = np.ones((num_points, 1))
    point_cloud_one_added = np.append(point_cloud, homogeneous_column, axis=1)

    transformed_point_cloud = np.matmul(transform, np.transpose(point_cloud_one_added))
    transformed_pc_one_removed = np.transpose(transformed_point_cloud)[:, :3]

    return transformed_pc_one_removed



def quaternion_rotation_matrix(quaternion):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
            This rotation matrix converts a point in the local reference 
            frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = quaternion[0]
    q1 = quaternion[1]
    q2 = quaternion[2]
    q3 = quaternion[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])

    return rot_matrix

def create_transform(translation, rotation):
    
    transform = np.identity(4)    
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    
    return transform

# PATH = "/home/nasir/Desktop/Demos/reach_target/variation0/episodes/episode0"

# with open(os.path.join(PATH, "low_dim_obs.pkl"), 'rb') as f:
#     obs = pickle.load(f)


# # Vision sensor gives point cloud in world frame 
# # Extrinsic - world frame to camera frame

# visualise_pc(full_pc_world)

# # (X,Y,Z,Qx,Qy,Qz,Qw)
# gripper_pos = obs[0].gripper_pose
# gripper_coord = gripper_pos[:3]
# gripper_rotation_quat = gripper_pos[3:]

# gripper_rotation_matrix = quaternion_rotation_matrix(gripper_rotation_quat)

# # print("Gripper pose is", gripper_pos)
# # print("Gripper coordinates is", gripper_coord)
# # print("Gripper rotation quaternion is", gripper_rotation_quat)
# # print("Grip matix is", gripper_rotation_matrix)

# gripper_transform = create_transform(gripper_coord, gripper_rotation_matrix)

# # print("Gripper transform is", gripper_transform)

# # full_pc_gripper = transform_point_cloud(gripper_transform, full_pc_world)
# # visualise_pc(full_pc_gripper)


# all_gripper_frames = []
# for i in range(len(obs)):

#     gripper_pos = obs[i].gripper_pose
#     gripper_coord = gripper_pos[:3]
#     gripper_rotation_quat = gripper_pos[3:]
#     gripper_rotation_matrix = quaternion_rotation_matrix(gripper_rotation_quat)

#     # print("Gripper pose is", gripper_pos)
#     # print("Gripper coordinates is", gripper_coord)
#     # print("Gripper rotation quaternion is", gripper_rotation_quat)
#     # print("Grip matix is", gripper_rotation_matrix)

#     gripper_frame = create_transform(gripper_coord, gripper_rotation_matrix)

#     all_gripper_frames.append(gripper_frame)

# all_actions = []

# for index, frame in enumerate(all_gripper_frames):

#     if index == len(all_gripper_frames)-1:
#         continue

#     next_frame = all_gripper_frames[index+1]
#     action = transform_between_frames(frame, next_frame)
#     all_actions.append(action)


# print("Action size is:", np.array(all_actions).shape)

# [transform_between_frames(all_gripper_transforms[i], all) for i in range(len(test_list))]
    # print("Gripper transform is", gripper_transform)


# print(np.matmul(all_gripper_frames[0], all_actions[0]) - all_gripper_frames[1])

demo_folder = "reach_target_10eps"
print(demo_folder)
config = get_config([128,128])
demos = depth_to_pc(config, demo_folder)

full_dataset = np.empty((1,2), dtype=object)

for i in range(len(demos)):

    dataset = []
    ls_mask = demos[i]._observations[0].left_shoulder_mask
    rs_mask = demos[i]._observations[0].right_shoulder_mask
    front_mask = demos[i]._observations[0].front_mask
    wrist_mask = demos[i]._observations[0].wrist_mask
    oh_mask = demos[i]._observations[0].overhead_mask

    # print(front_mask)

    ls_mask = np.array(ls_mask).reshape(-1)
    rs_mask = np.array(rs_mask).reshape(-1)
    front_mask = np.array(front_mask).reshape(-1)
    wrist_mask = np.array(wrist_mask).reshape(-1)
    oh_mask = np.array(oh_mask).reshape(-1)

    if i == 0:
        print("ls", np.unique(ls_mask))
        print("rs", np.unique(rs_mask))
        print("front", np.unique(front_mask))
        print("wrist", np.unique(wrist_mask))
        print("oh", np.unique(oh_mask))

    # print("Wrist mask is,", wrist_mask)
    # print("dims is:", np.array(wrist_mask).shape) 

    ls_pc_world = demos[i]._observations[0].left_shoulder_point_cloud
    rs_pc_world = demos[i]._observations[0].right_shoulder_point_cloud
    front_pc_world = demos[i]._observations[0].front_point_cloud
    wrist_pc_world = demos[i]._observations[0].wrist_point_cloud
    oh_pc_world = demos[i]._observations[0].overhead_point_cloud

    # print("Wrist pc world is,", wrist_pc_world)
    # test = (wrist_mask == 225)[..., None] * np.array(wrist_pc_world)
    # test2 = wrist_pc_world[wrist_mask == 225]
    # print("Test dims,", np.array(test2).shape)
    # print((wrist_mask == 225) * wrist_pc_world)
    # print(wrist_pc_world[wrist_mask == 225])

    #resize and then bool index


    ls_rgb = demos[i]._observations[0].left_shoulder_rgb
    rs_rgb = demos[i]._observations[0].right_shoulder_rgb
    front_rgb = demos[i]._observations[0].front_rgb
    wrist_rgb = demos[i]._observations[0].wrist_rgb
    oh_rgb = demos[i]._observations[0].overhead_rgb

    ls_colour_pc_world = np.concatenate((ls_pc_world, ls_rgb), axis=2).reshape(-1,6)
    rs_colour_pc_world = np.concatenate((rs_pc_world, rs_rgb), axis=2).reshape(-1,6)
    front_colour_pc_world = np.concatenate((front_pc_world, front_rgb), axis=2).reshape(-1,6)
    wrist_colour_pc_world = np.concatenate((wrist_pc_world, wrist_rgb), axis=2).reshape(-1,6)
    oh_colour_pc_world = np.concatenate((oh_pc_world, oh_rgb), axis=2).reshape(-1,6)

    mask_robot = True #False #True
    # 165 low limit
    maskNum = 213
    if mask_robot:
        ls_colour_pc_world = ls_colour_pc_world[ls_mask < maskNum]
        rs_colour_pc_world = rs_colour_pc_world[rs_mask < maskNum]
        front_colour_pc_world = front_colour_pc_world[np.logical_and(front_mask < 210, front_mask > 165)]
        wrist_colour_pc_world = wrist_colour_pc_world[wrist_mask < maskNum]
        oh_colour_pc_world = oh_colour_pc_world[oh_mask < maskNum]
        

    full_colour_pc_world = np.concatenate((ls_colour_pc_world, rs_colour_pc_world, front_colour_pc_world, wrist_colour_pc_world, oh_colour_pc_world))

    full_colour_pc_world = full_colour_pc_world[(full_colour_pc_world[:,0] > -1) & (full_colour_pc_world[:,2] > 0.755)]

    red_ball_only = False
    box_only = False
    yellow_cube_only = False
    charger_only = False
    scales_only = False

    if red_ball_only:
        full_colour_pc_world = full_colour_pc_world[(full_colour_pc_world[:,3] > 150) & (full_colour_pc_world[:,4] < 115) & (full_colour_pc_world[:,5] < 50)]
        # print(np.unique(full_colour_pc_world[:,3]))
    
    if box_only:
        full_colour_pc_world = full_colour_pc_world[(full_colour_pc_world[:,3] < 150) & (full_colour_pc_world[:,4] < 135) & (full_colour_pc_world[:,5] > 45)]

    if yellow_cube_only:
        full_colour_pc_world = full_colour_pc_world[(full_colour_pc_world[:,4] > 65) & (full_colour_pc_world[:,5] < 40)]

    if charger_only:
        full_colour_pc_world = full_colour_pc_world[(full_colour_pc_world[:,3] < 160)] 
        
    if scales_only:
        full_colour_pc_world = full_colour_pc_world[(full_colour_pc_world[:,3] < 160) & (full_colour_pc_world[:,4] < 150) & (full_colour_pc_world[:,5] < 150)] #& (full_colour_pc_world[:,4] < 135) & (full_colour_pc_world[:,5] > 45)]

    if i == 0:
        # visualise_pc_rgb(oh_colour_pc_world)
        visualise_pc_rgb(full_colour_pc_world)


    all_gripper_frames = []
    all_gripper_pc = []
    all_gripper_states = []
    for j in range(len(demos[i]._observations)):
        
        gripper_open = demos[i]._observations[j].gripper_open
        gripper_pos = demos[i]._observations[j].gripper_pose
        gripper_coord = gripper_pos[:3]
        gripper_rotation_quat = gripper_pos[3:]
        gripper_rotation_matrix = quaternion_rotation_matrix(gripper_rotation_quat)
        gripper_frame = create_transform(gripper_coord, gripper_rotation_matrix)

        full_pc_world_points, full_pc_world_colours = np.hsplit(full_colour_pc_world, 2)
        full_pc_gripper = transform_point_cloud(gripper_frame, full_pc_world_points)
        full_colour_pc_gripper = np.concatenate((full_pc_gripper, full_pc_world_colours), axis=1)


        all_gripper_frames.append(gripper_frame)
        all_gripper_pc.append(full_colour_pc_gripper)
        all_gripper_states.append(gripper_open)

    all_actions = []

    for index, frame in enumerate(all_gripper_frames):

        if index == len(all_gripper_frames)-1:
            continue

        next_frame = all_gripper_frames[index+1]
        next_gripper_state = all_gripper_states[index+1]
        action = transform_between_frames(frame, next_frame)
        
        if not next_gripper_state:
            action[3,0] = 1
        all_actions.append(action)

    all_actions.append(np.zeros((4,4)))

# print("Action size is:", np.array(all_actions).shape)
# print("Number of gripper pointclouds is:", np.array(all_gripper_pc).shape)

    pc_with_actions = list(zip(all_gripper_pc, all_actions))
    dataset = np.array(pc_with_actions, dtype=object)

    # print(np.array(dataset, dtype=object).shape)
    # print(np.array(dataset)[0])
    # full_dataset.append(dataset)

    full_dataset = np.append(full_dataset, dataset, axis=0)
    full_dataset = full_dataset[1:]



# print(np.array(full_dataset)[0].shape)
# print(np.array(full_dataset)[1])
# torch.save(full_dataset, '/vol/bitbucket/nm219/data/reach_target_64size.pt')


#/vol/bitbucket/nm219/data/
# print(len(dataset))


# pc, ac = pc_with_actions[0]
# print(np.array(pc).shape)
# print(np.array(ac).shape)  

#pyvista polydata