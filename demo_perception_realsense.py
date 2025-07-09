import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import open3d as o3d

# Avoid package shadowing
parent_dir = os.path.abspath(os.path.join(os.getcwd()))
sam_dir = os.path.abspath(os.path.join(os.getcwd(), "sam2"))
if parent_dir in sys.path:
    sys.path.remove(parent_dir)
if sam_dir not in sys.path:
    sys.path.append(sam_dir)

from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry


#REALSENSE SETUP
import pyrealsense2 as rs
# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
pipeline.start()
sensor = pipeline.get_active_profile().get_device().query_sensors()[1]
sensor.set_option(rs.option.exposure, 500)

def get_depth_frame(frames):
    depth = frames.get_depth_frame()
    depth_data = depth.as_frame().get_data()
    np_image = np.asanyarray(depth_data)
    return np_image

def get_color_frame(frames):
    color = frames.get_color_frame()
    color_data = color.as_frame().get_data()
    np_image = np.asanyarray(color_data)
    return np_image

#LOAD MODEL
sam_checkpoint = "/home/unitree/drl/unitree-application/sam/checkpoints/sam_vit_b_01ec64.pth"  # sam_vit_h_4b8939.pth
model_type = "vit_b"  # "vit_h"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def show_ann(ann):
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((ann['segmentation'].shape[0], ann['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    m = ann['segmentation']
    color_mask = np.array([0.0,0.0,0.0,1.0])
    img[np.invert(m)] = color_mask
    ax.imshow(img)
   
def get_color_diff(a, b):
    dist = np.linalg.norm(a-b)
    return dist

def get_mean_color(array, m):
    extended_mask = m[..., np.newaxis]
    extended_mask = np.repeat(extended_mask, 3, axis=2)
    masked_array = np.ma.array(array, mask=np.invert(extended_mask))
    return masked_array.mean(axis=(0,1))

if __name__ == "__main__":
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pc = rs.pointcloud()
    points = rs.points()
    
    # Start streaming
    profile = pipeline.start(config)
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    # Align the depth frame to color frame
    aligned_frames = align.process(frames)
    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()
    aligned_color_frame = aligned_frames.get_color_frame()

    points = pc.calculate(aligned_depth_frame)

    vertices = np.asanyarray(points.get_vertices(dims=2))


    w = aligned_depth_frame.get_width()
    image_Points = np.reshape(vertices , (-1,w,3))

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(aligned_color_frame.get_data())
    rgb = color_image[...,::-1]

    #CREATE SEGMENTATION
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=16, pred_iou_thresh=.95)
    image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)

    #SEGMENTED POINTCLOUDS
    for i, mask in enumerate(masks):
        plt.figure(figsize=(20, 20))
        plt.imshow(image)
        show_ann(mask)
        plt.axis('off')
        plt.savefig("test"+ str(i) + ".png")
        #print("mask shape: " + str(mask['segmentation'].shape))
        #print("image shape: " + str(image_Points.shape))
        vertices_interest = image_Points[np.array(mask['segmentation']),:].reshape(-1,3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices_interest)
        o3d.io.write_point_cloud(str(i)+'.ply', pcd)
    """
    
    #CULL SEGMENTATION
    greylist = []
    for mask, n in enumerate(masks):
        plt.figure(figsize=(20, 20))
        plt.imshow(img)
        show_ann(mask)
        plt.axis('off')
        plt.savefig("test"+ str(n) + ".png")
        avg_color = get_mean_color(img,mask['segmentation'])
        color_dist_grey = get_color_diff(avg_color, [59,72,56])
        if color_dist_grey < 35:
            greylist.append(n)
        plt.close()"""

    pipeline.stop()
