
import numpy as np
import pyvista


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
    

    plotter.add_arrows(np.array(centres), np.array(actions_list[:-1]), line_width = 5)
    plotter.add_axes_at_origin()
    plotter.show()