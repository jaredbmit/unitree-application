import numpy as np
import open3d as o3d
import cv2
import pyrealsense2 as rs
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

#LOAD MODEL
sam_checkpoint = "/home/unitree/drl/unitree-application/sam/checkpoints/sam_vit_b_01ec64.pth"  # sam_vit_h_4b8939.pth
model_type = "vit_b"  # "vit_h"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

def show_anns(anns, ax):
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

def show_ann(ann, ax):
    ax.set_autoscale_on(False)
    img = np.ones((ann['segmentation'].shape[0], ann['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    color_mask = np.concatenate([np.random.random(3), [0.35]])
    img[ann['segmentation']] = color_mask
    ax.imshow(img)

def visualize_point_cloud(points, idx=0):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=points[:, 2], cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    fig.savefig(f"point_cloud_{i}.png")

def project_depth(depth, mask, intrinsics, depth_scale, clip_distance_max=2.0):
    depth = depth * depth_scale
    rows, cols  = depth.shape

    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    r = r.astype(float)
    c = c.astype(float)

    valid = (depth > 0) & (depth < clip_distance_max)
    valid = valid & mask
    valid = np.ravel(valid)
    z = depth
    x =  z * (c - intrinsics.ppx) / intrinsics.fx
    y =  z * (r - intrinsics.ppy) / intrinsics.fy

    z = np.ravel(z)[valid]
    x = np.ravel(x)[valid]
    y = np.ravel(y)[valid]

    points = np.dstack((x, y, z))
    points = points.reshape(-1,3)

    return points

def deproject_frame(depth, mask, intrinsics, depth_scale):
    mask_2d = mask["segmentation"].astype(bool)
    h, w = depth.shape
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    xs = X[mask_2d]
    ys = Y[mask_2d]
    zs = depth[mask_2d] * depth_scale
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]
    pts = [rs.rs2_deproject_pixel_to_point(intrinsics, [float(x), float(y)], z) for x, y, z in zip(xs, ys, zs)]
    return np.array(pts)

if __name__ == "__main__":
    # Start and configure RS pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
    sensor.set_option(rs.option.exposure, 500)
    depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Get aligned frames
    align = rs.align(rs.stream.color)
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Segment color image
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam, 
        points_per_side=20,
        stability_score_thresh=0.9, 
        pred_iou_thresh=0.9,
    )
    image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    print("Segmenting image...")
    masks = mask_generator.generate(image)
    print("Segmentation complete!")

    masks.sort(key=lambda x: x["predicted_iou"]+x["stability_score"], reverse=True)
    num_masks = 5  # Only consider a few
    masks = masks[:num_masks]

    # Pre-process point clouds
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    point_clouds = []
    for i, mask in enumerate(masks):
        # De-project points
        pts = deproject_frame(depth_image, mask, intrinsics, depth_scale)
        # Convert to point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        # Downsample by voxel
        pcd = pcd.voxel_down_sample(voxel_size=0.002)
        # Outlier rejection
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # Estimate normals
        # pcd = o3d.geometry.estimate_normals(pcd)
        pcd.estimate_normals()
        point_clouds.append(pcd)

        # Visualize
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        show_ann(mask, ax)
        fig.savefig(f"fig/segmentation_{i}.png")
        plt.close()
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        pts = np.asarray(pcd.points)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, c='dodgerblue', alpha=0.8)
        ax.set_aspect('equal')
        plt.tight_layout()
        fig.savefig(f"fig/cloud_{i}.png")
        plt.close()

    # Run ICP on point clouds
    pts_brick = np.load("../assets/Brick_Small.npy")
    pcd_brick = o3d.geometry.PointCloud()
    pcd_brick.points = o3d.utility.Vector3dVector(pts_brick)
    pcd_brick.estimate_normals()

    # ~Approximate~ coordinate frames for ICP init
    T_init = np.eye(4)
    rotation_head_to_camera = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    r2 = np.sqrt(2)/2
    rotation_world_to_head = np.array([[r2, 0, r2], [0, 1, 0], [-r2, 0, r2]])  # Nominally, an assumption
    rotation_world_to_camera = rotation_world_to_head @ rotation_head_to_camera
    T_init[:3, :3] = rotation_world_to_camera

    threshold = 1.0
    best_score = 0.0
    for i, pcd in enumerate(point_clouds):
        reg = o3d.pipelines.registration.registration_icp(
            pcd, pcd_brick, threshold, T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500),
        )
        print("Fitness: ", reg.fitness)
        print("Inlier RMSE: ", reg.inlier_rmse)
        print("Length: ", len(pcd.points))
        if reg.fitness == 0. or reg.inlier_rmse == 0.:
            continue
        score = reg.fitness / reg.inlier_rmse * len(pcd.points)
        if score > best_score:
            best_score = score
            best_idx = i
            best_pcd = pcd
            best_T = reg.transformation
        pcd_tf = pcd.transform(reg.transformation)
        pts_tf = np.asarray(pcd_tf.points)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pts_tf[:, 0], pts_tf[:, 1], pts_tf[:, 2], s=1, c='lightgreen', alpha=0.8)
        ax.scatter(pts_brick[:, 0], pts_brick[:, 1], pts_brick[:, 2], s=1, c='dodgerblue', alpha=0.8)
        ax.set_aspect('equal')
        plt.tight_layout()
        fig.savefig(f"fig/match_{i}.png")
        plt.close()

    print(f"Best match: PCD {best_idx}")
    print(f"Pose of brick wrt camera: {best_T}")

    pipeline.stop()
