import time
import os
import numpy as np
import cv2
from datetime import datetime
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from cam_data import Image_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import Header_

def decode_image_message(image_msg):
    """
    Decode Unitree Image_ message back to numpy array
    
    Args:
        image_msg: Image_ message object
        
    Returns:
        numpy array of the image
    """
    # Convert data list back to bytes
    image_bytes = bytes(image_msg.data)
    
    # Determine data type based on encoding
    if image_msg.encoding in [1, 2]:  # RGB8, BGR8
        dtype = np.uint8
        channels = 3
    elif image_msg.encoding == 3:  # MONO8
        dtype = np.uint8
        channels = 1
    elif image_msg.encoding in [4, 5]:  # MONO16, DEPTH_16UC1
        dtype = np.uint16
        channels = 1
    else:
        print(f"Unknown encoding: {image_msg.encoding}")
        dtype = np.uint8
        channels = 1
    
    # Convert bytes to numpy array
    if channels == 1:
        image_array = np.frombuffer(image_bytes, dtype=dtype).reshape(
            (image_msg.height, image_msg.width)
        )
    else:
        image_array = np.frombuffer(image_bytes, dtype=dtype).reshape(
            (image_msg.height, image_msg.width, channels)
        )
    
    return image_array

def save_image(image_array, encoding, frame_id, timestamp, save_dir):
    """
    Save image array to file
    
    Args:
        image_array: numpy array of the image
        encoding: encoding type
        frame_id: frame identifier
        timestamp: message timestamp
        save_dir: directory to save images
    """
    # Create timestamp string for filename
    dt = datetime.fromtimestamp(timestamp.sec + timestamp.nanosec / 1e9)
    timestamp_str = dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
    
    # Determine file extension and processing based on encoding
    if encoding in [1, 2]:  # RGB8, BGR8
        if encoding == 1:  # RGB8
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:  # BGR8
            image_bgr = image_array
        
        filename = f"{frame_id}_{timestamp_str}_color.jpg"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, image_bgr)
        
    elif encoding == 3:  # MONO8
        filename = f"{frame_id}_{timestamp_str}_mono.jpg"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, image_array)
        
    elif encoding in [4, 5]:  # MONO16, DEPTH_16UC1
        # For depth images, save as PNG to preserve 16-bit data
        filename = f"{frame_id}_{timestamp_str}_depth.png"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, image_array)
        
        # Also save a normalized version for visualization
        depth_normalized = cv2.normalize(image_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        filename_norm = f"{frame_id}_{timestamp_str}_depth_norm.jpg"
        filepath_norm = os.path.join(save_dir, filename_norm)
        cv2.imwrite(filepath_norm, depth_normalized)
    
    print(f"Saved image: {filename}")

if __name__ == "__main__":
    # Initialize channel factory
    ChannelFactoryInitialize()
    
    # Create save directory
    save_directory = "saved_images"
    os.makedirs(save_directory, exist_ok=True)
    
    # Create subscriber for image messages
    # Replace "image_topic" with your actual topic name
    sub = ChannelSubscriber("image_topic", Image_)
    sub.Init()
    
    print(f"Subscriber initialized. Saving images to: {save_directory}")
    print("Press Ctrl+C to stop...")
    
    try:
        while True:
            msg = sub.Read()
            if msg is not None:
                print(f"Received image: {msg.width}x{msg.height}, encoding: {msg.encoding}, frame: {msg.header.frame_id}")
                
                # Decode image message
                image_array = decode_image_message(msg)
                
                # Save image
                save_image(image_array, msg.encoding, msg.header.frame_id, msg.header.stamp, save_directory)
                
            else:
                print("No data subscribed.")
                time.sleep(0.01)  # Small delay to prevent busy waiting
                
    except KeyboardInterrupt:
        print("\nShutting down subscriber...")
    finally:
        sub.Close()
        print("Subscriber closed.")