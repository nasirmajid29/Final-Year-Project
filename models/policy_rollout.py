import gym
import rlbench.gym
import numpy as np

import torch
from simple_pointnet_pp import Net, SAModule, GlobalSAModule
from torch_geometric.data import Data, Dataset
from GPUtil import showUtilization as gpu_usage

import torch_geometric.transforms as T

def transform_point_cloud(transform, point_cloud)-> np.ndarray:

    num_points = point_cloud.shape[0]
    homogeneous_column = np.ones((num_points, 1))
    point_cloud_one_added = np.append(point_cloud, homogeneous_column, axis=1)

    transformed_point_cloud = np.matmul(transform, np.transpose(point_cloud_one_added))
    transformed_pc_one_removed = np.transpose(transformed_point_cloud)[:, :3]

    return transformed_pc_one_removed


def quaternion_rotation_matrix(quaternion):

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

gpu_usage()
torch.cuda.empty_cache()
# print(torch.cuda.memory_allocated())
gpu_usage()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)

model = torch.load("/homes/nm219/Final-Year-Project/reach_target_100eps_pnpp.pt", map_location=device) #torch.device('cpu'))
model.eval()

env = gym.make('reach_target-state-v0', render_mode='human', observation_mode='vision')
 
training_steps = 200 #120
episode_length = 100 #100 #40

runs = 10 #100
episode_length = 75 #100

reach_goal = np.array([False]*runs)
time_to_goal = np.zeros(runs)

for i in range(runs): 
    for j in range(episode_length):
        if j == 0:
            print('Reset Episode')
            obs = env.reset()
        # print(action)
        # 8 nums, 3 translate, 4 quaternion, 1 gripper
        print("Step ", j, "of run ", i)


        front_colour_pc_world = np.append(np.array(obs["front_point_cloud"]).reshape(-1,3), np.array(obs["front_rgb"]).reshape(-1,3), axis=1)
        wrist_colour_pc_world = np.append(np.array(obs["wrist_point_cloud"]).reshape(-1,3), np.array(obs["wrist_rgb"]).reshape(-1,3), axis=1)
        ls_colour_pc_world = np.append(np.array(obs["left_shoulder_point_cloud"]).reshape(-1,3), np.array(obs["left_shoulder_rgb"]).reshape(-1,3), axis=1)
        rs_colour_pc_world = np.append(np.array(obs["right_shoulder_point_cloud"]).reshape(-1,3), np.array(obs["right_shoulder_rgb"]).reshape(-1,3), axis=1)
        oh_colour_pc_world = np.append(np.array(obs["overhead_point_cloud"]).reshape(-1,3), np.array(obs["overhead_rgb"]).reshape(-1,3), axis=1)
        
        front_mask = np.array(obs["front_mask"]).reshape(-1)
        wrist_mask = np.array(obs["wrist_mask"]).reshape(-1)
        left_shoulder_mask = np.array(obs["left_shoulder_mask"]).reshape(-1)
        right_shoulder_mask = np.array(obs["right_shoulder_mask"]).reshape(-1)
        overhead_mask = np.array(obs["overhead_mask"]).reshape(-1)
        
        

        # 165 low limit
        maskNum = 213
        ls_colour_pc_world = ls_colour_pc_world[left_shoulder_mask < maskNum]
        rs_colour_pc_world = rs_colour_pc_world[right_shoulder_mask < maskNum]
        front_colour_pc_world = front_colour_pc_world[np.logical_and(front_mask < 210, front_mask > 165)]
        wrist_colour_pc_world = wrist_colour_pc_world[wrist_mask < maskNum]
        oh_colour_pc_world = oh_colour_pc_world[overhead_mask < maskNum]
            

        full_colour_pc_world = np.concatenate((ls_colour_pc_world, rs_colour_pc_world, front_colour_pc_world, wrist_colour_pc_world, oh_colour_pc_world))
        full_colour_pc_world = full_colour_pc_world[(full_colour_pc_world[:,0] > -1) & (full_colour_pc_world[:,2] > 0.5)]
        full_colour_pc_world = full_colour_pc_world[(full_colour_pc_world[:,3] > 150) & (full_colour_pc_world[:,4] < 115) & (full_colour_pc_world[:,5] < 50)]


        gripper_pos = obs["gripper_pose"]
        gripper_coord, gripper_rotation_quat = gripper_pos[:3], gripper_pos[3:]
        gripper_rotation_matrix = quaternion_rotation_matrix(gripper_rotation_quat)
        gripper_frame = create_transform(gripper_coord, gripper_rotation_matrix)

        full_pc_world_points, full_pc_world_colours = np.hsplit(full_colour_pc_world, 2)
        full_pc_gripper = transform_point_cloud(gripper_frame, full_pc_world_points)
        full_colour_pc_gripper = np.concatenate((full_pc_gripper, full_pc_world_colours), axis=1)

        full_colour_pc_gripper = torch.tensor(full_colour_pc_gripper, dtype=torch.float32)

        data = Data(x = full_colour_pc_gripper)
        data.pos = full_colour_pc_gripper[:, :3]
        
        
        pre_transform = T.NormalizeScale()
        data = pre_transform(data)

        data.batch = torch.zeros(full_colour_pc_gripper.size(0), dtype=torch.int64)
        data = data.to(device)
        
        action = np.zeros(8)
        action[6] = 1
        action[7] = 1
        pred_action = model(data)
        action[:3] = pred_action.cpu().detach().numpy()
        
        # action = [0.1, 0, 0, 0, 0, 0, 1, 0]

        # action = env.action_space.sample()
        # print(action)
        # action[3:7] = [0, 0, 0, 1]
        
        # print("--------------------")
        # print("Gripper pose: ", gripper_pos)
        # print("Action taken: ", action)

        obs, reward, terminate, _ = env.step(action)
        
        if terminate:
            reach_goal[i] = True
            time_to_goal[i] = j
            break
        
        if j == episode_length - 1:
            reach_goal[i] = False
            time_to_goal[i] = -1            
        
        # print("Resulting gripper pose: ", obs["gripper_pose"])
        # difference = gripper_pos - obs["gripper_pose"]
        # print("Difference: ", difference)
        # print("How close to action: ", difference-action[:7])
        # print("--------------------")

        env.render()  # Note: rendering increases step time.


accuracy = reach_goal.sum() / runs

time_to_goal = time_to_goal[time_to_goal > 0]

avg_speed = np.mean(time_to_goal[time_to_goal != 0])

print('Done')
print("Accuracy = ", accuracy)
print("Average speed = ", avg_speed)

env.close()