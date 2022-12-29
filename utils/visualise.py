
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
