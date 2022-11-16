import open3d as o3d
import numpy as np
import pyvista

def depth_to_pc(depth_path, colour_path):
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

wrist_depth = "/home/nasir/Desktop/Demos/reach_target/variation0/episodes/episode0/wrist_depth/0.png" 
wrist_colour = "/home/nasir/Desktop/Demos/reach_target/variation0/episodes/episode0/wrist_rgb/0.png"

points, colour = depth_to_pc(wrist_depth, wrist_colour)

visualise_pc(p, c)