import time
import numpy as np
import cv2
import pyrealsense2 as rs

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from cam_data import ImageChunk_


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


def chunk_image(image: np.ndarray, chunk_height: int):
    """Split image into horizontal chunks."""
    if image.shape[0] % chunk_height != 0:
        raise ValueError("chunk_height should divide image height")
    chunks = [image[i:i+chunk_height] for i in range(image.shape[0] // chunk_height)]
    return chunks


if __name__ == "__main__":
    realsense = SimpleRealSense()

    ChannelFactoryInitialize()
    pub = ChannelPublisher("image_topic", ImageChunk_)
    pub.Init()

    for i in range(10):

        color_image, _ = realsense.get_images()
        chunk_height = 32
        chunks = chunk_image(color_image, chunk_height)

        for idx, chunk in enumerate(chunks):
            msg = ImageChunk_(
                height = chunk.shape[0],
                width = chunk.shape[1],
                depth = chunk.shape[2],
                chunk_index = idx,           
                num_chunks = len(chunks),    
                data = chunk.flatten().tolist(),
            )

            # Publish message
            if pub.Write(msg, 0.5):
                print("Published color image.")
            else:
                print("Waitting for subscriber.")
            time.sleep(0.05)

        time.sleep(1)

    pub.Close()
