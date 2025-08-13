import os
import time
import numpy as np
import cv2
import torch
import numpy as np
import open3d as o3d

filename = "Brick_Small"

n_points = np.load(filename + ".npy")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(n_points)
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=1, origin=[0, 0, 0])
o3d.visualization.draw_geometries([pcd, mesh_frame])

input("press to close pcd")

