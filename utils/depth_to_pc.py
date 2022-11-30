import os
import pickle
import open3d as o3d
import numpy as np
import pyvista

from PIL import Image

from rlbench.utils import get_stored_demos, ObservationConfig, _resize_if_needed
from rlbench.backend.utils import image_to_float_array, rgb_handles_to_mask
from pyrep.objects import VisionSensor

def rgb_depth_to_pc(colour_path, depth_path):
    depth = o3d.io.read_image(depth_path)
    colour = o3d.io.read_image(colour_path)


    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(colour, depth, convert_rgb_to_intensity = False)
    camera = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, camera)

    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # o3d.visualization.draw_geometries([pcd])



    points = np.asarray(pcd.points)
    colours = np.asarray(pcd.colors)

    return points, colours


def visualise_pc(points):
    
    plotter = pyvista.Plotter()
    plotter.add_points(points, opacity=2, point_size=3, render_points_as_spheres=True)
    plotter.show()

def visualise_pc_rgb(points, colours):
    
    plotter = pyvista.Plotter()
    plotter.add_points(points, opacity=1, point_size=1.5, render_points_as_spheres=True, scalars=colours, rgb=True)
    plotter.show()

def generate_pc_for_episode(episode_size, episode_path, camera_type):

    all_points = []
    all_colours = []

    for i in range(episode_size):
        depth_path = episode_path+"/wrist_depth/" +i+".png"
        rgb_path = episode_path+"/wrist_rgb/" +i+".png"
        
        points, colours = rgb_depth_to_pc(rgb_path, depth_path)
        all_points.append(points)
        all_colours.append(colours)
    return all_points, all_colours


def generate_episode_point_clouds(episode_path):

    full_pc_points = []
    full_pc_colours = []

    for file in os.listdir(episode_path+"/front_depth"):

        front_depth = episode_path+"/front_depth/" + file
        front_rgb = episode_path+"/front_rgb/" + file
        front_points, front_colours = rgb_depth_to_pc(front_rgb, front_depth)
    
        left_depth = episode_path+"/left_shoulder_depth/" + file
        left_rgb = episode_path+"/left_shoulder_rgb/" + file
        left_points, left_colours = rgb_depth_to_pc(left_rgb, left_depth)
        full_points = np.concatenate((front_points, left_points), axis=0)
        full_colours = np.concatenate((front_colours, left_colours), axis=0)

        right_depth = episode_path+"/right_shoulder_depth/" + file
        right_rgb = episode_path+"/right_shoulder_rgb/" + file
        points, colours = rgb_depth_to_pc(right_rgb, right_depth)
        full_points = np.concatenate((full_points, points), axis=0)
        full_colours = np.concatenate((full_colours, colours), axis=0)
        
        over_depth = episode_path+"/overhead_depth/" + file
        over_rgb = episode_path+"/overhead_rgb/" + file
        points, colours = rgb_depth_to_pc(over_rgb, over_depth)
        full_points = np.concatenate((full_points, points), axis=0)
        full_colours = np.concatenate((full_colours, colours), axis=0)
        
        wrist_depth = episode_path+"/wrist_depth/" + file
        wrist_rgb = episode_path+"/wrist_rgb/" + file
        points, colours = rgb_depth_to_pc(wrist_rgb, wrist_depth)
        full_points = np.concatenate((full_points, points), axis=0)
        full_colours = np.concatenate((full_colours, colours), axis=0)

        full_pc_points.append(full_points)
        full_pc_colours.append(full_colours)

        # print(len(full_pc_points))
    
    return np.asarray(full_pc_points), np.asarray(full_pc_colours) 
        

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


    return obs_config


def depth_to_pc(obs_config, task_name):

    # (amount: int, image_paths: bool, dataset_root: str,
    #                  variation_number: int, task_name: str,
    #                  obs_config: ObservationConfig,
    #                  random_selection: bool = True,
    #                  from_episode_number: int = 0) -> List[Demo]:
    demos = get_stored_demos(-1, False, "/home/nasir/Desktop/Demos", 0, task_name, obs_config, random_selection=False, from_episode_number=6)
    return demos


PATH = "/home/nasir/Desktop/Demos/reach_target/variation0/episodes/episode0"

with open(os.path.join(PATH, "low_dim_obs.pkl"), 'rb') as f:
    obs = pickle.load(f)

IMAGE_SIZE = [128, 128]
DEPTH_SCALE = 2**24 - 1
configuration = get_config(IMAGE_SIZE)

left_img_path = os.path.join(PATH, "left_shoulder_depth/0.png")
left_shoulder_extrinsic = obs[0].misc['left_shoulder_camera_extrinsics']
left_shoulder_intrinsic = obs[0].misc['left_shoulder_camera_intrinsics']
left_shoulder_near = obs[0].misc['left_shoulder_camera_near']
left_shoulder_far = obs[0].misc['left_shoulder_camera_far']

right_img_path = os.path.join(PATH, "right_shoulder_depth/0.png")
right_shoulder_extrinsic = obs[0].misc['right_shoulder_camera_extrinsics']
right_shoulder_intrinsic = obs[0].misc['right_shoulder_camera_intrinsics']
right_shoulder_near = obs[0].misc['right_shoulder_camera_near']
right_shoulder_far = obs[0].misc['right_shoulder_camera_far']

front_img_path = os.path.join(PATH, "front_depth/0.png")
front_extrinsic = obs[0].misc['front_camera_extrinsics']
front_intrinsic = obs[0].misc['front_camera_intrinsics']
front_near = obs[0].misc['front_camera_near']
front_far = obs[0].misc['front_camera_far']

wrist_img_path = os.path.join(PATH, "wrist_depth/0.png")
wrist_extrinsic = obs[0].misc['wrist_camera_extrinsics']
wrist_intrinsic = obs[0].misc['wrist_camera_intrinsics']
wrist_near = obs[0].misc['wrist_camera_near']
wrist_far = obs[0].misc['wrist_camera_far']

overhead_img_path = os.path.join(PATH, "overhead_depth/0.png")
overhead_extrinsic = obs[0].misc['overhead_camera_extrinsics']
overhead_intrinsic = obs[0].misc['overhead_camera_intrinsics']
overhead_near = obs[0].misc['overhead_camera_near']
overhead_far = obs[0].misc['overhead_camera_far']

left_shoulder_depth = image_to_float_array(_resize_if_needed(Image.open(left_img_path), configuration.left_shoulder_camera.image_size), DEPTH_SCALE)
left_shoulder_depth_m = left_shoulder_near + left_shoulder_depth * (left_shoulder_far - left_shoulder_near)

right_shoulder_depth = image_to_float_array(_resize_if_needed(Image.open(right_img_path), configuration.right_shoulder_camera.image_size), DEPTH_SCALE)
right_shoulder_depth_m = right_shoulder_near + right_shoulder_depth * (right_shoulder_far - right_shoulder_near)

front_depth = image_to_float_array(_resize_if_needed(Image.open(front_img_path), configuration.front_camera.image_size), DEPTH_SCALE)
front_depth_m = front_near + front_depth * (front_far - front_near)

wrist_depth = image_to_float_array(_resize_if_needed(Image.open(wrist_img_path), configuration.wrist_camera.image_size), DEPTH_SCALE)
wrist_depth_m = wrist_near + wrist_depth * (wrist_far - wrist_near)

overhead_depth = image_to_float_array(_resize_if_needed(Image.open(overhead_img_path), configuration.overhead_camera.image_size), DEPTH_SCALE)
overhead_depth_m = overhead_near + overhead_depth * (overhead_far - overhead_near)


left_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(left_shoulder_depth_m, left_shoulder_extrinsic, left_shoulder_intrinsic)
right_shoulder_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(right_shoulder_depth_m, right_shoulder_extrinsic, right_shoulder_intrinsic)
front_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(front_depth_m, front_extrinsic, front_intrinsic)
wrist_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(wrist_depth_m, wrist_extrinsic, wrist_intrinsic)
overhead_point_cloud = VisionSensor.pointcloud_from_depth_and_camera_params(overhead_depth_m, overhead_extrinsic, overhead_intrinsic)

# Extrinsic = world to camera


# visualise_pc(left_shoulder_point_cloud)
# visualise_pc(right_shoulder_point_cloud)
# visualise_pc(front_point_cloud)
visualise_pc(wrist_point_cloud)
# visualise_pc(overhead_point_cloud)


# print("Near:", near)
# print("Far: ", far)
# print("ext:", ext)
# print("int: ", int)
# print(len(obs))

# config = get_config([128,128])
# demos = depth_to_pc(config, "reach_target")

# print(demos[0])
# print(demos[0].left_shoulder_point_cloud)

# print(obs[10].left_shoulder_point_cloud)
# print(obs[10].right_shoulder_depth)
# print(obs[10])
# print(obs[10])
