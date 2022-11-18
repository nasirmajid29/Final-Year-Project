import os
import open3d as o3d
import numpy as np
import pyvista

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

def visualise_pc(points, colours):
    
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

points, colour = generate_episode_point_clouds(episode_path)
print(points[0].shape)
visualise_pc(points[0], colour[0])