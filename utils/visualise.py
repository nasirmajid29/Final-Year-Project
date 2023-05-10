
import numpy as np
import pyvista
pyvista.start_xvfb()


def visualise_pc(point_cloud):

    pyvista.plot(
        point_cloud,
        scalars=point_cloud[:, 2],
        render_points_as_spheres=True,
        point_size=4,
        show_scalar_bar=False
        )

def visualise_pc_old(points):
    
    plotter = pyvista.Plotter()
    plotter.add_points(points, opacity=2, point_size=3, render_points_as_spheres=True)
    plotter.show()

def visualise_pc_rgb(point_cloud):
    
    points, colours = np.hsplit(point_cloud, 2)
    plotter = pyvista.Plotter()
    plotter.add_points(points, opacity=1, point_size=4, render_points_as_spheres=True, scalars=colours.astype(int), rgb=True)
    plotter.show()

def visualise_pc_rgb_many(point_cloud_list):

    plotter = pyvista.Plotter()
    for point_cloud in point_cloud_list:
    
        points, colours = np.hsplit(point_cloud, 2)
        plotter.add_points(points, opacity=1, point_size=4, render_points_as_spheres=True, scalars=colours.astype(int), rgb=True)
    plotter.add_axes_at_origin()
    plotter.show()

def visualise_actions(actions):

    plotter = pyvista.Plotter()
    actions_list = []

    for action in actions:

        
        action_translate = action[:3, 3].reshape(-1)
        actions_list.append(action_translate)

    centres = np.cumsum(actions_list, axis=0)
    centres = [0,0,0] + centres[:-1]
    

    plotter.add_arrows(np.array(centres), np.array(actions_list[:-1]))
    plotter.add_axes_at_origin()
    plotter.show()


def visualise_over_time(point_cloud_list, actions):

    plotter = pyvista.Plotter()
    actions_list = []

    for point_cloud in point_cloud_list:
    
        points, colours = np.hsplit(point_cloud, 2)
        plotter.add_points(points, opacity=1, point_size=4, render_points_as_spheres=True, scalars=colours.astype(int), rgb=True)

    for action in actions:
    
        action_translate = action[:3, 3].reshape(-1)
        actions_list.append(action_translate)

    centres = np.cumsum(actions_list, axis=0)
    centres = [0,0,0] + centres[:-1]    


    plotter.add_points(centres, opacity=1, point_size=4, render_points_as_spheres=True)
    plotter.add_arrows(np.array(centres), np.array(actions_list[:-1]))
    plotter.add_axes_at_origin()
    plotter.show()
    

def fake_visualise_pc_rgb(point_cloud):
    
    points, colours = np.hsplit(point_cloud, 2)
    plotter = pyvista.Plotter(off_screen=True)
    plotter.add_points(points, opacity=1, point_size=4, render_points_as_spheres=True, scalars=colours.astype(int), rgb=True)
    # plotter.camera.zoom(2.5)
    plotter.screenshot(filename="visualised.png")

def cumulative_sum(actions):
    
    cumulative_sum = np.cumsum(actions, axis=0)
    result = [tuple(point) for point in cumulative_sum]
    
    return result

def visualise_policy(point_clouds, actions):
    
    gripper_points = cumulative_sum(actions)
    
    plotter = pyvista.Plotter()
    for point_cloud in point_clouds:
        points, colours = np.hsplit(point_cloud, 2)
        plotter.add_points(points, opacity=1, point_size=4, render_points_as_spheres=True, scalars=colours.astype(int), rgb=True)

    polydata = pyvista.PolyData(gripper_points)
    lines = polydata.lines
    plotter.add_mesh(lines, color='yellow', line_width=5)
    plotter.add_mesh(polydata, color='yellow', point_size=20)

    plotter.screenshot("policy.png")
