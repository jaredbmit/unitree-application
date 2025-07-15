import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Avoid package shadowing
parent_dir = os.path.abspath(os.path.join(os.getcwd()))
sam_dir = os.path.abspath(os.path.join(os.getcwd(), "sam2"))
if parent_dir in sys.path:
    sys.path.remove(parent_dir)
if sam_dir not in sys.path:
    sys.path.append(sam_dir)

from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

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

if __name__ == "__main__":
    image = cv2.imread('../sam/notebooks/images/dog.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=16)

    import time
    start = time.perf_counter()
    masks = mask_generator.generate(image)
    end = time.perf_counter()
    print(f"Time: {end - start}")
    
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig("test1.png")

