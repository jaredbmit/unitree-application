import os
import time
import numpy as np
import cv2
import open3d as o3d
import torch
import pyrealsense2 as rs
import trimesh
import plotly.graph_objects as go
import plotly.io as pio

import matplotlib.pyplot as plt

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

class SimpleSAM:
    """
    Simple wrapper interface for segment-anything
    """
    # Use the smallest SAM model
    CHECKPOINT = "/home/unitree/drl/unitree-application/sam/checkpoints/sam_vit_b_01ec64.pth"  
    MODEL = "vit_b"

    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        sam = sam_model_registry[self.MODEL](checkpoint=self.CHECKPOINT)
        sam.to(device=self.device)
        self.mask_generator = self.mask_generator = SamAutomaticMaskGenerator(
            model=sam, 
            points_per_side=12,
            stability_score_thresh=0.9, 
            pred_iou_thresh=0.9,
        )

    def flush(self):
        del self.mask_generator
        gc.collect()
        torch.cuda.empty_cache() 

    def generate(self, image):
        """image -> list of annotations"""
        masks = self.mask_generator.generate(image)
        masks.sort(key=lambda x: x["predicted_iou"]+x["stability_score"], reverse=True)
        return masks
    
    def show_anns(self, anns, ax):
        """populates an axis with mask annotation"""
        if len(anns) == 0:
            print("zero anns")
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax.set_autoscale_on(False)
        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        #ax.imshow(img)


class SimpleRealSense:
    """
    Simple wrapper interface for RealSense camera
    """
    EXPOSURE_TIME = 500

    def __init__(self):
        # Start and configure RS pipeline
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)
        
        # Set exposure time
        sensor = self.pipeline.get_active_profile().get_device().query_sensors()[1]
        sensor.set_option(rs.option.exposure, self.EXPOSURE_TIME)
        
        # Get depth scale
        depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        self.intrinsics = None

    def get_images(self):
        """returns (color, depth)"""
        # Align frames
        align = rs.align(rs.stream.color)
        frames = self.pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Bookkeep intrinsics
        if self.intrinsics is None:
            self.intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        return color_image, depth_image
    
    def deproject_frame(self, depth, color, mask):
        """depth and mask are both arrays"""
        if self.intrinsics is None:
            raise RuntimeError("method `get_images` must be called at least once to calibrate `intrinsics`")
        
        # Mask
        h, w = depth.shape
        if mask is None:
            mask_2d = np.ones((h,w)).astype(bool)
        else:
            mask_2d = mask.astype(bool)
        #print("depth shape: " + str(depth.shape))
        #print("color shape: " + str(color.shape))
        Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        xs = X[mask_2d]
        ys = Y[mask_2d]
        zs = depth[mask_2d] * self.depth_scale

        # Filter invalid measurements
        valid = zs > 0
        xs, ys, zs = xs[valid], ys[valid], zs[valid]
        pts = []
        colors = []
        #print("xs shape: "+ str(xs.shape))

        # Deproject one at a time
        for point in zip(xs, ys, zs):
            pts += [rs.rs2_deproject_pixel_to_point(self.intrinsics, [float(point[0]), float(point[1])], point[2])]
            curr_color = color[point[1],point[0]]
            colors += [curr_color]
        return np.array(pts), np.array(colors)

    def stop(self):
        self.pipeline.stop()


class Perception:
    """
    Perception pipeline for brick pose estimation
    """
    BRICK_FILE = os.path.expanduser("~/drl/unitree-application/assets/Brick_Small.npy")
    ANGLE_THRESHOLD = 1
    NUM_MASKS = 10  # Number of segmentations to consider
    THRESHOLD = 1.0  # Furthest ICP correspondence
    MAX_ITER = 500  # Number of ICP iterations
    R_CAMERA_TO_REALSENSE = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    R_HEAD_TO_CAMERA = np.array([  # An assumption for ICP init
        [np.sqrt(2)/2, 0, np.sqrt(2)/2], 
        [0, 1, 0], 
        [-np.sqrt(2)/2, 0, np.sqrt(2)/2]]
    )
    T_INIT_GUESS = np.array(
        [[0.0119, 0.766, -0.643, 0.477],
        [-1, -0.005, -0.024, 0.016],
        [-0.022, 0.643, 0.765, -0.405],
        [0,0,0,1 ]]
    )

    def __init__(self, log_dir=None):
        # Init subsystems
        self.realsense = SimpleRealSense()
        self.sam = SimpleSAM()

        # Load brick point cloud
        pts_brick = np.load(self.BRICK_FILE)
        self.pcd_brick = o3d.geometry.PointCloud()
        self.pcd_brick.points = o3d.utility.Vector3dVector(pts_brick)
        self.pcd_brick.estimate_normals()

        # ~Approximate~ coordinate frames for ICP init
        self.T_init = self.T_INIT_GUESS
        #self.T_init = np.eye(4)
        #R_head_to_realsense = self.R_HEAD_TO_CAMERA @ self.R_CAMERA_TO_REALSENSE
        #self.T_init[:3, :3] = R_head_to_realsense

        self.log_dir = log_dir

    def get_color_pointcloud(self):
        
        color_image, depth_image = self.realsense.get_images()
        points, colors = self.realsense.deproject_frame(depth_image, color_image, None) 
        
        return points, colors

    def world_to_point(self, rs_to_point,T_world_to_camera):
        T_point_to_rs = np.eye(4)
        T_point_to_rs[:3,3] = rs_to_point       

        T_camera_to_realsense = np.eye(4)
        T_camera_to_realsense[:3, :3] = self.R_CAMERA_TO_REALSENSE

        T_camera_to_point = T_camera_to_realsense @ T_point_to_rs
        T_world_to_point = T_world_to_camera @ T_camera_to_point
        return T_world_to_point[:3,3]

    def camera_to_point(self, rs_to_point, debug=False):
        T_point_to_rs = np.eye(4)
        T_point_to_rs[:3,3] = rs_to_point
        

        T_camera_to_realsense = np.eye(4)
        T_camera_to_realsense[:3, :3] = self.R_CAMERA_TO_REALSENSE

        T_camera_to_point = T_camera_to_realsense @ T_point_to_rs
        if debug:
            print("T_point_to_rs:")
            print(T_point_to_rs)
            print("T_camera_to_point:")
            print(T_camera_to_point)
        return T_camera_to_point[:3,3]

    def transform_verts(self, verts, T):
        new_verts = np.zeros([verts.shape[0],3],dtype=np.float32)
        for i in range(verts.shape[0]):
            R = T[0:3,0:3]
            new_vert = R @ verts[i]
            new_vert = new_vert + T[0:3,3]
            new_verts[i] = new_vert    
        return new_verts
    
    def estimate_brick_pose(self):
        """pose of brick with respect to camera frame"""
        # Get current images
        color_image, depth_image = self.realsense.get_images()

        # Segment color image
        print("Segmenting image.")
        t_start = time.perf_counter()
        masks = self.sam.generate(color_image)
        t_stop = time.perf_counter()
        print(f"Segmentation completed, taking {t_stop - t_start} seconds and generating {len(masks)} masks.")
        print(f"Only considering the top {self.NUM_MASKS} masks, ranked by estimated quality.")
        masks = masks[:self.NUM_MASKS]

        if self.log_dir is not None:
            fig, ax = plt.subplots(1)
            #ax.imshow(color_image)
            self.sam.show_anns(masks, ax)
            fig.suptitle(f"Top {self.NUM_MASKS} annotations")
            fig.savefig(os.path.join(self.log_dir, "segmentation.png"))

        # De-project to point clouds and run ICP
        print("Computing best ICP match.")
        t_start = time.perf_counter()
        best_score = 0.0
        best_pcd = None
        T_brick_to_realsense = None
        point_clouds = []
        for i, mask in enumerate(masks):
            # De-project points
            pts, colors = self.realsense.deproject_frame(depth_image, color_image, mask["segmentation"])

            # Downsample and reject outliers
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)

            pcd = pcd.voxel_down_sample(voxel_size=0.002)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            point_clouds.append(pcd)

            # Compute ICP registration
            pcd.estimate_normals()
            reg = o3d.pipelines.registration.registration_icp(
                pcd, self.pcd_brick, self.THRESHOLD, self.T_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.MAX_ITER),
            )

            # Heuristic
            if reg.fitness == 0. or reg.inlier_rmse == 0.:
                continue
            score = reg.fitness / reg.inlier_rmse * len(pcd.points)
            if score > best_score:
                best_score = score
                best_pcd = pcd
                T_brick_to_realsense = reg.transformation
        
        t_stop = time.perf_counter()
        print(f"ICP matching completed, taking {t_stop - t_start} seconds and matching {len(masks)} point clouds.")

        if T_brick_to_realsense is None or not isinstance(T_brick_to_realsense, np.ndarray):
            raise RuntimeError("ICP failed to find a valid match among the segmented point clouds.")
        
        # Transform to camera frame
        T_realsense_to_brick = np.linalg.inv(T_brick_to_realsense)
        T_camera_to_realsense = np.eye(4)
        T_camera_to_realsense[:3, :3] = self.R_CAMERA_TO_REALSENSE
        T_camera_to_brick = T_camera_to_realsense @ T_realsense_to_brick

        if self.log_dir is not None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            for pcd in point_clouds:
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                colors = np.divide(colors.astype(np.float32),255)
                #print("color example: " + str(colors[0]))
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color=colors)

            points = np.asarray(best_pcd.points)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color='lightgreen')
            ax.set_aspect("equal")
            fig.savefig(os.path.join(self.log_dir, "projections1.png", dpi=300))
            #plt.show()

        print(f"ICP match found. Estimated brick pose wrt D435 camera frame: {T_camera_to_brick}")

        return T_camera_to_brick

    def estimate_brick_pose_debug(self, log = False):
        DEBUG_NUMBER = str(np.random.randint(999999))
        azm = -216.8
        ele = -16.75
        xlm = [0.418,0.822]
        ylm = [-0.143, 0.120]
        zlm = [-0.08, 0.164]

        """pose of brick with respect to camera frame"""
        # Get current images
        color_image, depth_image = self.realsense.get_images()

        # Segment color image
        print("Segmenting image.")
        t_start = time.perf_counter()
        masks = self.sam.generate(color_image)
        t_stop = time.perf_counter()
        masks = masks[:self.NUM_MASKS]
        if self.log_dir is not None:
            fig = plt.figure() 
            ax2 = fig.add_subplot(111)
            #ax2.imshow(color_image)
            self.sam.show_anns(masks, ax2)
            fig.suptitle(f"Top {self.NUM_MASKS} annotations")
            fig.savefig(os.path.join(self.log_dir, DEBUG_NUMBER + "segmentation.png"))
            plt.close(fig)

        # De-project to point clouds and run ICP
        print("Computing best ICP match.")
        t_start = time.perf_counter()
        best_score = 0.0
        best_pcd = None
        T_brick_to_realsense = None
        point_clouds = []
        for i, mask in enumerate(masks):
            # De-project points
            pts, colors = self.realsense.deproject_frame(depth_image, color_image, mask["segmentation"])

            # Downsample and reject outliers
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            bb = pcd.get_minimal_oriented_bounding_box()
            #print("@@@ EXTENT: " + str(bb.extent))
            if(np.max(bb.extent) > 0.75):
                continue
            if(np.min(bb.extent) < 0.03):
                continue
            pcd = pcd.voxel_down_sample(voxel_size=0.002)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            point_clouds.append(pcd)

            # Compute ICP registration
            pcd.estimate_normals()
            reg = o3d.pipelines.registration.registration_icp(
                pcd, self.pcd_brick, self.THRESHOLD, self.T_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.MAX_ITER),
            )

            # Heuristic
            if reg.fitness == 0. or reg.inlier_rmse == 0.:
                continue
            score = reg.fitness / reg.inlier_rmse * len(pcd.points)
            if score > best_score:
                best_score = score
                best_pcd = pcd
                T_brick_to_realsense = reg.transformation
                
        
        t_stop = time.perf_counter()
        print(f"ICP matching completed, taking {t_stop - t_start} seconds and matching {len(masks)} point clouds.")

        if T_brick_to_realsense is None or not isinstance(T_brick_to_realsense, np.ndarray):
            raise RuntimeError("ICP failed to find a valid match among the segmented point clouds.")
        #Print transforms debug

        #here, display best registration yet...
        fig3 = plt.figure()
        fig3.suptitle("Pointcloud Registration")
        ax = fig3.add_subplot(111, projection='3d')
        ax.set_clip_on(False)
        ax.set_aspect("equal")
        points = np.asarray(best_pcd.points)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color='lime')
        
        source_points = np.asarray(self.pcd_brick.points)
        transformed_source_points = self.transform_verts(source_points, T_brick_to_realsense)
        ax.scatter(transformed_source_points[:, 0], transformed_source_points[:, 1], transformed_source_points[:, 2], s=1, color='blue')

        #plt.show()
        

        # Transform to camera frame
        T_realsense_to_brick = np.linalg.inv(T_brick_to_realsense)
        T_camera_to_realsense = np.eye(4)
        T_camera_to_realsense[:3, :3] = self.R_CAMERA_TO_REALSENSE
        T_camera_to_brick = T_camera_to_realsense @ T_realsense_to_brick
        
        def plot_pose(T, ax, label):
            scale = 0.1
            ax.quiver(*T[:3,3], *T[:3,0], length=scale, color='r')
            ax.quiver(*T[:3,3], *T[:3,1], length=scale, color='g')
            ax.quiver(*T[:3,3], *T[:3,2], length=scale, color='b')
            ax.text(*T[:3,3], label, color='black', fontsize=10)

        if self.log_dir is not None:
            fig = plt.figure(dpi=1200)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_clip_on(False)
            ax.view_init(elev=-427.8, azim=-463.7) #Reproduce view
            ax.set_xlim3d(-0.231,0.188)     #Reproduce magnification
            ax.set_ylim3d(-0.105,0.193)     #...
            ax.set_zlim3d(0.55,0.806)
            ax.set_axis_off()

            plot_pose(self.T_init, ax, "T_init")
            
            
            
            for pcd in point_clouds:
                points = np.asarray(pcd.points)
                colors = np.asarray(pcd.colors)
                colors = np.divide(colors.astype(np.float32),255)
                #print("color example: " + str(colors[0]))
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color=colors)

            points = np.asarray(best_pcd.points)
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color='lightgreen')
            ax.set_aspect("equal")
            fig.savefig(os.path.join(self.log_dir, DEBUG_NUMBER + "projections.png"))
            plt.close(fig)

        
        
        def plot_cube(verts, ax, color):
            lines = [(0,1),(1,2),(2,3),(3,0),(0,4),(1,5),(2,6),(3,7),(4,5),(5,6),(6,7),(7,4)]
            for line in lines:
                x = []
                y = []
                z = []
                start = line[0]
                end = line[1]
                x.append(verts[start,0])
                x.append(verts[end,0])
                y.append(verts[start,1])
                y.append(verts[end,1])
                z.append(verts[start,2])
                z.append(verts[end,2])
                ax.plot3D(x,y,z,color)

        def plot_axis(verts, ax, color):
            ax.scatter(verts[:,0],verts[:,1],verts[:,2],c=color)
 
        #CAMERA FRAME POINTCLOUD DEBUG
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, projection='3d')

        points, colors = self.get_color_pointcloud()
        # Downsample pointcloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd = pcd.voxel_down_sample(voxel_size=0.005)

        points = np.asarray(pcd.points)
        
        for i in range(len(points)):
            points[i] = self.camera_to_point(points[i])
        colors = np.asarray(pcd.colors)
        colors = np.divide(colors.astype(np.float32),255)
        ax1.set_clip_on(False)
        ax1.view_init(elev=ele, azim=azm) #Reproduce view
        ax1.set_xlim3d(xlm[0],xlm[1])     #Reproduce magnification
        ax1.set_ylim3d(ylm[0],ylm[1])     #...
        ax1.set_zlim3d(zlm[0],zlm[1])
        ax1.set_axis_off()

        ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color=colors)
        ax1.set_aspect("equal")

        T_realsense_to_init = np.linalg.inv(self.T_init)
        T_camera_to_init = T_camera_to_realsense @ T_realsense_to_init

        plot_pose(T_camera_to_brick, ax1, "camera_to_Brick")
        plot_pose(T_realsense_to_brick, ax1, "realsense_to_Brick")
        plot_pose(T_camera_to_init, ax1, "camera_to_init")

        #DRAW WIREFRAME BRICK
        cube = np.zeros([8,3],dtype=np.float32)
        cube[:,1] = [0.045,0.045,-0.045,-0.045,0.045,0.045,-0.045,-0.045]
        cube[:,0] = [-0.03,0.03,0.03,-0.03,-0.03,0.03,0.03,-0.03]
        cube[:,2] = [0,0,0,0,0.03,0.03,0.03,0.03]
        transformed_cube = self.transform_verts(cube, T_camera_to_brick)
        plot_cube(transformed_cube,ax1,"lime")

        init_cube = self.transform_verts(cube, T_camera_to_init)
        plot_cube(init_cube,ax1,"yellow")
       
        #if log == False:    
            #plt.show()
        #else:
            #filename = DEBUG_NUMBER + "wireframe.png"
            #plt.savefig(os.path.join(self.log_dir, filename))
            

        print(f"ICP match found. Estimated brick pose wrt D435 camera frame: {T_camera_to_brick}")

        return T_camera_to_brick
    
    def estimate_brick_pose_color_debug(self, color, log = False):
        print(color)
        if color == "r" or "g":
            print("get valid color")
        else:
            raise ValueError("invalid color passed")
        """pose of brick with respect to camera frame"""
        def get_mean_color(array, m):
            extended_mask = m[..., np.newaxis]
            extended_mask = np.repeat(extended_mask, 3, axis=2)
            masked_array = np.ma.array(array, mask=np.invert(extended_mask))
            return masked_array.mean(axis=(0,1))

        def get_color_diff(a, b):
            dist = np.linalg.norm(a-b)
            return dist

        # Get current images
        color_image, depth_image = self.realsense.get_images()

        # Segment color image
        print("Segmenting image.")
        t_start = time.perf_counter()
        masks = self.sam.generate(color_image)
        t_stop = time.perf_counter()
        print(f"Segmentation completed, taking {t_stop - t_start} seconds and generating {len(masks)} masks.")
        #print(f"Only considering the top {self.NUM_MASKS} masks, ranked by estimated quality.")
        #masks = masks[:self.NUM_MASKS]
        target_color = [150,110,120] if color == 'r' else [150,190,200]
        target_masks = []
        #SUBSET THE MASKS BY COLOR
        for mask in masks:
            avg_color = get_mean_color(color_image,mask['segmentation'])
            print("Average color: " + str(avg_color))
            color_dist = get_color_diff(avg_color, target_color)
            if color_dist < 50:
                target_masks.append(mask)
        print("number of color-matched masks: " + str(len(target_masks)))
        #TODO: here, pass color-matched masks only
        if self.log_dir is not None:
            fig, ax = plt.subplots(1)
            #ax.imshow(color_image)
            self.sam.show_anns(masks, ax)
            fig.suptitle(f"Top {self.NUM_MASKS} annotations")
            fig.savefig(os.path.join(self.log_dir, str(np.random.randint(999999)) + "segmentation.png"))
        print("color matched masks...")
        if self.log_dir is not None:
            fig, ax = plt.subplots(1)
            #ax.imshow(color_image)
            self.sam.show_anns(target_masks, ax)
            fig.suptitle(f"Color-matched masks")
            fig.savefig(os.path.join(self.log_dir, str(np.random.randint(999999)) + "colormatch.png"))

        # De-project to point clouds and run ICP
        print("Computing best ICP match.")
        t_start = time.perf_counter()
        best_score = 0.0
        best_pcd = None
        T_brick_to_realsense = None
        point_clouds = []
        for i, mask in enumerate(target_masks):
            print("beginning icp round..")
            print(mask["segmentation"])
            # De-project points
            pts, colors = self.realsense.deproject_frame(depth_image, color_image, mask["segmentation"])

            # Downsample and reject outliers
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            bb = pcd.get_minimal_oriented_bounding_box()
            #print("@@@ EXTENT: " + str(bb.extent))
            if(np.max(bb.extent) > 0.75):
                continue
            if(np.min(bb.extent) < 0.03):
                continue

            pcd = pcd.voxel_down_sample(voxel_size=0.002)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            point_clouds.append(pcd)

            # Compute ICP registration
            pcd.estimate_normals()
            reg1 = o3d.pipelines.registration.registration_icp(
                pcd, self.pcd_brick, self.THRESHOLD, self.T_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.MAX_ITER),
            )
            reg2 = o3d.pipelines.registration.registration_icp(
                pcd, self.pcd_brick, self.THRESHOLD, self.T_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.MAX_ITER),
            )
            reg1_t = np.transpose(reg1.transformation[0:3,0:3])
            r_diff = reg1_t @ reg2.transformation[0:3,0:3]
            angle1 = np.abs(np.arccos((np.trace(r_diff) - 1)/2))    
            if angle1 < self.ANGLE_THRESHOLD:
                reg = reg2
            else:
                print("ICP difference larger than threshold, doing additional round..")
                reg3 = o3d.pipelines.registration.registration_icp(
                pcd, self.pcd_brick, self.THRESHOLD, self.T_init,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=self.MAX_ITER),
            )   
                reg2_t = np.transpose(reg2.transformation[0:3,0:3])
                r_diff2 = reg2_t @ reg3.transformation[0:3,0:3]
                angle2 = np.abs(np.arccos((np.trace(r_diff2) - 1)/2))
                
                reg3_t = np.transpose(reg3.transformation[0:3,0:3])
                r_diff3 = reg3_t @ reg1.transformation[0:3,0:3]
                angle3 = np.abs(np.arccos((np.trace(r_diff3) - 1)/2))
                
                angles = [angle1, angle2, angle3]
                if (angles.index(min(angles)) == 0):
                    reg = reg1
                else: 
                    reg = reg3 
            

            # Heuristic
            if reg.fitness == 0. or reg.inlier_rmse == 0.:
                continue
            score = reg.fitness / reg.inlier_rmse * len(pcd.points)
            if score > best_score:
                best_score = score
                best_pcd = pcd
                T_brick_to_realsense = reg.transformation
        
        t_stop = time.perf_counter()
        print(f"ICP matching completed, taking {t_stop - t_start} seconds and matching {len(target_masks)} point clouds.")

        if T_brick_to_realsense is None or not isinstance(T_brick_to_realsense, np.ndarray):
            raise RuntimeError("ICP failed to find a valid match among the segmented point clouds.")
        #Print transforms debug

        # Transform to camera frame
        T_realsense_to_brick = np.linalg.inv(T_brick_to_realsense)
        T_camera_to_realsense = np.eye(4)
        T_camera_to_realsense[:3, :3] = self.R_CAMERA_TO_REALSENSE
        T_camera_to_brick = T_camera_to_realsense @ T_realsense_to_brick
        
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #ax.set_title("Brick Transform (T_brick_to_realsense) and pointcloud in native frame")

        def plot_pose(T, ax, label):
            scale = 0.1
            print("plot pose debug")
            print("transform: ")
            print(T)
            print("T[:3,3]")
            print(T[:3,3])
            print("T[:3,0]")
            print(T[:3,0])
            ax.quiver(*T[:3,3], *T[:3,0], length=scale, color='r')
            ax.quiver(*T[:3,3], *T[:3,1], length=scale, color='g')
            ax.quiver(*T[:3,3], *T[:3,2], length=scale, color='b')
            ax.text(*T[:3,3], label, color='black', fontsize=10)
        
        def plot_cube(verts, ax):
            lines = [(0,1),(1,2),(2,3),(3,0),(0,4),(1,5),(2,6),(3,7),(4,5),(5,6),(6,7),(7,4)]
            for line in lines:
                x = []
                y = []
                z = []
                start = line[0]
                end = line[1]
                x.append(verts[start,0])
                x.append(verts[end,0])
                y.append(verts[start,1])
                y.append(verts[end,1])
                z.append(verts[start,2])
                z.append(verts[end,2])
                ax.plot3D(x,y,z,'lime')

        def plot_axis(verts, ax, color):
            ax.scatter(verts[:,0],verts[:,1],verts[:,2],c=color)
        
         
            
        #for pcd in point_clouds:
        #    points = np.asarray(pcd.points)
        #    colors = np.asarray(pcd.colors)
        #    colors = np.divide(colors.astype(np.float32),255)
        #    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color=colors)

        #points = np.asarray(best_pcd.points)
        #ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color='lightgreen')
        #ax.set_aspect("equal")
        print("REALSENSE TO BRICK")
        print(T_realsense_to_brick)
        print("CAMERA TO BRICK")
        print(T_camera_to_brick)
        print("CAMERA TO REALSENSE")
        print(T_camera_to_realsense)

        
        #CAMERA FRAME POINTCLOUD DEBUG
        #fig1 = plt.figure()
        #ax1 = fig1.add_subplot(111, projection='3d')

        points, colors = self.get_color_pointcloud()
        # Downsample pointcloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd = pcd.voxel_down_sample(voxel_size=0.005)

        points = np.asarray(pcd.points)
        #print("REALSENSE TO POINT 0")
        #print(points[0])
        
        for i in range(len(points)):
            points[i] = self.camera_to_point(points[i])
        print(points[0])
        colors = np.asarray(pcd.colors)
        colors = np.divide(colors.astype(np.float32),255)
        #ax1.set_clip_on(False)
        #plt.close()
        
        azm = -216.8
        ele = -16.75
        xlm = [0.418,0.822]
        ylm = [-0.143, 0.120]
        zlm = [-0.08, 0.164]
        #ax1.view_init(elev=ele, azim=azm) #Reproduce view
        #ax1.set_xlim3d(xlm[0],xlm[1])     #Reproduce magnification
        #ax1.set_ylim3d(ylm[0],ylm[1])     #...
        #ax1.set_zlim3d(zlm[0],zlm[1])
        
        #ax1.set_axis_off()
        #ax1.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, color=colors)
        #ax1.set_aspect("equal")
        #plot_pose(T_camera_to_brick, ax1, "camera_to_Brick")
        #plot_pose(T_realsense_to_brick, ax1, "realsense_to_Brick")

        #DRAW WIREFRAME BRICK
        cube = np.zeros([8,3],dtype=np.float32)
        cube[:,1] = [0.045,0.045,-0.045,-0.045,0.045,0.045,-0.045,-0.045]
        cube[:,0] = [-0.03,0.03,0.03,-0.03,-0.03,0.03,0.03,-0.03]
        cube[:,2] = [0,0,0,0,0.03,0.03,0.03,0.03]
        #transformed_cube = self.transform_verts(cube, T_camera_to_brick)
        #plot_cube(transformed_cube,ax1)
        #x_verts = np.zeros([3,3],dtype=np.float32)
        #x_verts[:,0] = [0,0.03,0.06]
        #y_verts = np.zeros([3,3],dtype=np.float32)
        #y_verts[:,1] = [0,0.03,0.06]
        #z_verts = np.zeros([3,3],dtype=np.float32)
        #z_verts[:,2] = [0,0.03,0.06]
        #transformed_x = self.transform_verts(x_verts, T_camera_to_brick)
        #transformed_y = self.transform_verts(y_verts, T_camera_to_brick)
        #transformed_z = self.transform_verts(z_verts, T_camera_to_brick)
        #plot_axis(transformed_x, ax1, 'red')
        #plot_axis(transformed_y, ax1, 'lime')
        #plot_axis(transformed_z, ax1, 'blue')
       
    
        #if log == False:    
            #plt.show()
            #xlm=ax1.get_xlim3d() #These are two tupples
            #ylm=ax1.get_ylim3d() #we use them in the next
            #zlm=ax1.get_zlim3d() #graph to reproduce the magnification from mousing
            #azm=ax1.azim
            #ele=ax1.elev
            #print(str(xlm) + " " + str(ylm) + " " + str(zlm) + " " + str(azm) + " " + str(ele))
        #else:
            #filename = str(np.random.randint(999999)) + ".png"
            #plt.savefig(os.path.join(self.log_dir, filename))
            

        print(f"ICP match found. Estimated brick pose wrt D435 camera frame: {T_camera_to_brick}")

        return T_camera_to_brick
    
    def stop(self):
        self.realsense.stop()


if __name__ == "__main__":
    perception = Perception(log_dir="../log")
    for i in range(10):
        t_start = time.perf_counter()
        #T_brick_to_world = perception.estimate_brick_pose_color_debug("r",log = True)
        T_brick_to_world = perception.estimate_brick_pose_debug(log = True)
        t_stop = time.perf_counter()
        print(f"Total perception run took {t_stop - t_start} seconds")

    perception.stop()
