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
    #masked = cv2.bitwise_and(img,img, mask=m)
    color_mask = np.array([0.0,0.0,0.0,1.0])
    img[np.invert(m)] = color_mask
    ax.imshow(img)
   
def get_color_diff(a, b):
    dist = np.linalg.norm(a-b)
    return dist

def get_mean_color(array, m):
    print("array shape:")
    print(array.shape)
    
    extended_mask = m[..., np.newaxis]
    extended_mask = np.repeat(extended_mask, 3, axis=2)
    print(extended_mask.shape)

    #print(array)
    masked_array = np.ma.array(array, mask=np.invert(extended_mask))
    return masked_array.mean(axis=(0,1))
if __name__ == "__main__":
    align = rs.align(rs.stream.depth)
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    color_frame = get_color_frame(frames)
    color_aligned_to_depth = aligned.first(rs.stream.color)
    depth_frame = frames.first(rs.stream.depth)

   
    #color = o3d.geometry.Image(cur_frame)
    #depth = o3d.geometry.Image(cur_frame_depth)
    
    #print("currframe shape: " + str(color_frame.shape))
    #print("depthframe shape: " + str(cur_frame_depth.shape))
    #img = color_aligned_to_depth
    img = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    #cv2.imwrite('realsense_img.jpg', img)

    #generate pointcloud
    colorizer = rs.colorizer()
    colorized = colorizer.process(frames)
    ply = rs.save_to_ply("1.ply")
    color_frame = frames.get_color_frame()
    ply.process(frames)
    pc = rs.pointcloud()    
    #pc.map_to(color_aligned_to_depth)
    #points = pc.calculate(depth_frame)
    #print("yippie!")
    pointcloud = o3d.io.read_point_cloud("1.ply")
    print("read pointcloud")
    
    o3d.visualization.draw_geometries([pointcloud],
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
    input("press enter to continue..")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=16, pred_iou_thresh=.95)

    import time
    start = time.perf_counter()
    masks = mask_generator.generate(img)
    #print("Masks") 
    #print(masks)
    end = time.perf_counter()
    print(f"Time: {end - start}")
    n = 0
    threshhold = 20
    greylist = []
    print("Masks length:")
    print(len(masks))
    for mask in masks:
        print("Mask data specs:")
        print("Type: " + str(type(mask)))
        print("Keys: " + str(mask.keys()))
        print("Segmentation type: " + str(type(mask['segmentation'])))
        print("Segmentation shape: " + str(np.shape(mask['segmentation'])))
        print("Sample [0][0]: " + str(mask['segmentation'][0,0]))
        plt.figure(figsize=(20, 20))
        plt.imshow(img)
        show_ann(mask)
        plt.axis('off')
        plt.savefig("test"+ str(n) + ".png")
        print("calculated average value of segment " + str(n))
        avg_color = get_mean_color(img,mask['segmentation'])
        print(avg_color)
        color_dist_brown = get_color_diff(avg_color, [85,61,39])
        color_dist_grey = get_color_diff(avg_color, [59,72,56])
        if color_dist_grey < 35:
            greylist.append(n)
        print("difference from brown: " + str(color_dist_brown))
        print("difference from grey: " + str(color_dist_grey))
        plt.close()
        n = n + 1
    print("list of grey brick segments: ")
    print(greylist)
    pipeline.stop()
