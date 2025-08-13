import os
import time
import numpy as np
import cv2
import torch
import numpy as np
import open3d as o3d

filename = "Brick_Small"

mesh = o3d.io.read_triangle_mesh(filename + ".stl")
pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, 10000)
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Point Cloud")
vis.add_geometry(pcd)
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
vis.add_geometry(origin)
view_control = vis.get_view_control()
view_control.set_constant_z_far(1000)
x = input("press to save")
pcd_array = np.asarray(pcd.points)
np.save(filename, pcd_array)


