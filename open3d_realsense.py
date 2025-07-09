import numpy as np
import open3d as o3d
import traceback
import time
import pkgutil
print(list(pkgutil.iter_modules(o3d.__path__)))
o3d.t.io.RealSenseSensor.list_devices()
#depth_cam = o3d.t.io.RealSenseSensor()
#depth_cam.start_capture()

device = 'cuda:0' if o3d.core.cuda.is_available() else 'cpu:0'
o3d_device = o3d.core.Device(device)
intrinsic_matrix = o3d.core.Tensor(depth_cam.get_metadata().intrinsics.intrinsic_matrix, dtype=o3d.core.Dtype.Float32, device=o3d_device) 

# Initialize the pointcloud viewer
pcd = o3d.geometry.PointCloud()
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Point Cloud")
vis.add_geometry(pcd)
origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
vis.add_geometry(origin)
view_control = vis.get_view_control()
view_control.set_constant_z_far(1000)

while True:
    
    try:

        im_rgbd = depth_cam.capture_frame(True, True)
        new_pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(im_rgbd, intrinsic_matrix, depth_scale=depth_cam.get_metadata().depth_scale).to_legacy()
        R_camera_to_world = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).astype(np.float64)
        new_pcd.rotate(R_camera_to_world, center=np.array([0,0,0],dtype=np.float64))
        pcd.points = new_pcd.points
        pcd.colors = new_pcd.colors

        # Update the visualizer
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

    except KeyboardInterrupt:
        depth_cam.stop_capture()
        vis.destroy_window()
        break
    
    except:
        print(traceback.format_exc())
