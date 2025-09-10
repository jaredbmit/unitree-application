import time
import cv2

import pyrealsense2 as rs
import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from cam_data import Image_, Time_
# from unitree_sdk2py.idl.sensor_msgs import PointCloud2_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import Header_
from unitree_sdk2py.idl.default import std_msgs_msg_dds__Header_

# Common encoding constants (you may need to adjust these based on your SDK)
ENCODING_RGB8 = 1
ENCODING_BGR8 = 2
ENCODING_MONO8 = 3
ENCODING_MONO16 = 4
ENCODING_DEPTH_16UC1 = 5

def package_image_to_unitree_format(image_array, encoding_type, frame_id="camera", timestamp=None):
    """
    Package numpy image array into Unitree Image_ format
    
    Args:
        image_array: numpy array (H, W, C) for color or (H, W) for depth
        encoding_type: encoding format (see encoding constants below)
        frame_id: frame identifier string
        timestamp: optional timestamp, uses current time if None
    """
    # time_stamp = Time_()
    # time_stamp.sec = int(timestamp // 1_000_000_000)
    # time_stamp.nanosec = int(timestamp % 1_000_000_000)

    # Create header with Time_ object
    # header = Header_()
    # header.stamp = time_stamp
    # header.frame_id = frame_id    

    # header = std_msgs_msg_dds__Header_()
    if timestamp is None:
        timestamp = time.time_ns()
    
    # Create Time_ object
    time_stamp = Time_()
    time_stamp.sec = int(timestamp // 1_000_000_000)
    time_stamp.nanosec = int(timestamp % 1_000_000_000)
    
    # Create header with proper Time_ object
    header = std_msgs_msg_dds__Header_()
    header.stamp = time_stamp
    header.frame_id = frame_id
    # Create header
    # header = Header_(timestamp, frame_id)
    # header.stamp = time_stamp
    # header.frame_id = frame_id

    # Get image dimensions
    if len(image_array.shape) == 3:
        height, width, channels = image_array.shape
    else:
        height, width = image_array.shape
        channels = 1
    
    # Get image dimensions
    if len(image_array.shape) == 3:
        height, width, channels = image_array.shape
    else:
        height, width = image_array.shape
        channels = 1
    
    # Calculate step (bytes per row)
    bytes_per_pixel = image_array.dtype.itemsize * (channels if len(image_array.shape) == 3 else 1)
    step = width * bytes_per_pixel
    
    # Convert image to bytes
    image_data = image_array.tobytes()
    
    # Create Image_ object
    image_msg = Image_(
            header=header,
            height=height,
            width=width,
            encoding=encoding_type,
            is_bigendian=False,
            step=step,
            data=list(image_data)
        )
        
    return image_msg

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
    
    def create_header(self):
        """Create DDS Header message with current timestamp"""
        header = Header_()
        
        # Get current time
        current_time = time.time()
        sec = int(current_time)
        nanosec = int((current_time - sec) * 1e9)
        
        # Create timestamp
        timestamp = Time_()
        timestamp.sec = sec
        timestamp.nanosec = nanosec
        
        header.stamp = timestamp
        header.frame_id = "camera_link"
        
        return header

    def create_image_msg(self, cv_image, encoding='rgb8'):
        """Convert OpenCV image to DDS Image message"""
        img_msg = Image_()
        
        # Set header
        img_msg.header = self.create_header()
        
        # Set image properties
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = encoding
        
        # Set step (bytes per row)
        if encoding == 'rgb8':
            img_msg.step = cv_image.shape[1] * 3
        elif encoding == '16UC1':
            img_msg.step = cv_image.shape[1] * 2
        else:
            img_msg.step = cv_image.shape[1]
        
        # Convert image data to bytes
        img_msg.data = cv_image.tobytes()
        img_msg.is_bigendian = False
        
        return img_msg

    def stop(self):
        self.pipeline.stop()


    def get_images_packaged(self):
        """Returns packaged (color, depth) in Unitree Image_ format"""
        # Get your existing images
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
        
        # Get timestamp from frames (more accurate than system time)
        timestamp_ns = color_frame.get_timestamp() * 1_000_000  # Convert ms to ns
        
        # Package images
        color_msg = package_image_to_unitree_format(
            color_image, 
            ENCODING_RGB8, 
            "camera_color",
            timestamp_ns
        )
        
        depth_msg = package_image_to_unitree_format(
            depth_image,
            ENCODING_DEPTH_16UC1,
            "camera_depth", 
            timestamp_ns
        )
        
        return color_msg, depth_msg


if __name__ == "__main__":
    ChannelFactoryInitialize()
    realsense = SimpleRealSense()
    # color_image, depth_image = realsense.get_images()

    # Create a publisher to publish the data defined in UserData class
    pub = ChannelPublisher("image_topic", Image_)
    pub.Init()
# PointCloud2_(std_msgs_msg_dds__Header_(), 0, 0, [], False, 0, 0, [], False)

    for i in range(30):
        # Create a Userdata message
        # height, width = color_image.shape[0], color_image.shape[1]
        
        # msg = Image_(Header_, height, width, 0, False, 0, [])
        color_msg, depth_msg = realsense.get_images_packaged()
        # Publish message
        if pub.Write(color_msg, 0.5):
            print("Published color image.")
        else:
            print("Waitting for subscriber.")

        time.sleep(1)

    pub.Close()
