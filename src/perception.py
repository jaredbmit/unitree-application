import numpy as np
import cv2
import torch
import pyrealsense2 as rs
import open3d as o3d
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
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam, 
            points_per_side=20,
            stability_score_thresh=0.9, 
            pred_iou_thresh=0.9,
        )

    def generate(self, image):
        """image -> list of annotations"""
        masks = self.mask_generator.generate(image)
        masks.sort(key=lambda x: x["predicted_iou"]+x["stability_score"], reverse=True)
        return masks
    
    def show_anns(self, anns, ax):
        """populates an axis with mask annotation"""
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)


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
    
    def deproject_frame(self, depth, mask):
        """depth and mask are both arrays"""
        if self.intrinsics is None:
            raise RuntimeError("method `get_images` must be called at least once to calibrate `intrinsics`")
        
        # Mask
        mask_2d = mask.astype(bool)
        h, w = depth.shape
        Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        xs = X[mask_2d]
        ys = Y[mask_2d]
        zs = depth[mask_2d] * self.depth_scale

        # Filter invalid measurements
        valid = zs > 0
        xs, ys, zs = xs[valid], ys[valid], zs[valid]

        # Deproject one at a time
        pts = [
            rs.rs2_deproject_pixel_to_point(self.intrinsics, [float(x), float(y)], z)
            for x, y, z in zip(xs, ys, zs)
        ]

        return np.array(pts)

    def stop(self):
        self.pipeline.stop()


class Perception:
    """
    Perception pipeline for brick pose estimation
    """
    BRICK_FILE = "../assets/brick.npy"
    NUM_MASKS = 5  # Number of segmentations to consider
    THRESHOLD = 1.0  # Furthest ICP correspondence
    MAX_ITER = 500  # Number of ICP iterations
    R_CAMERA_TO_REALSENSE = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    R_HEAD_TO_CAMERA = np.array([  # An assumption for ICP init
        [np.sqrt(2)/2, 0, np.sqrt(2)/2], 
        [0, 1, 0], 
        [-np.sqrt(2)/2, 0, np.sqrt(2)/2]]
    )

    def __init__(self):
        # Init subsystems
        self.realsense = SimpleRealSense()
        self.sam = SimpleSAM()

        # Load brick point cloud
        pts_brick = np.load("../assets/brick.npy")
        self.pcd_brick = o3d.geometry.PointCloud()
        self.pcd_brick.points = o3d.utility.Vector3dVector(pts_brick)
        self.pcd_brick.estimate_normals()

        # ~Approximate~ coordinate frames for ICP init
        self.T_init = np.eye(4)
        R_head_to_realsense = self.R_HEAD_TO_CAMERA @ self.R_CAMERA_TO_REALSENSE
        self.T_init[:3, :3] = R_head_to_realsense.T
    
    def estimate_brick_pose(self):
        """pose of brick with respect to camera frame"""
        # Get current images
        color_image, depth_image = self.realsense.get_images()

        # Segment color image
        masks = self.sam.generate(color_image)
        masks = masks[:self.NUM_MASKS]

        # De-project to point clouds and run ICP
        best_score = 0.0
        T_realsense_to_brick = None
        for i, mask in enumerate(masks):
            # De-project points
            pts = self.realsense.deproject_frame(depth_image, mask["segmentation"])

            # Downsample and reject outliers
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd = pcd.voxel_down_sample(voxel_size=0.002)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

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
                T_realsense_to_brick = reg.transformation

        # Transform to camera frame
        T_camera_to_realsense = np.eye(4)
        T_camera_to_realsense[:3, :3] = self.R_CAMERA_TO_REALSENSE
        T_camera_to_brick = T_camera_to_realsense @ T_realsense_to_brick

        return T_camera_to_brick

    def stop(self):
        self.realsense.stop()