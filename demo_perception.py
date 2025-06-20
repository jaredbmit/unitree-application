import os
import sys
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Avoid package shadowing
parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
sam2_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "sam2"))
if parent_dir in sys.path:
    sys.path.remove(parent_dir)
if sam2_dir not in sys.path:
    sys.path.append(sam2_dir)
import sam2

if __name__ == "__main__":
    print(sam2.__path__)
