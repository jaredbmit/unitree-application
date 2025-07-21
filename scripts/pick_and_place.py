import os
import sys
import time
import numpy as np
base_dir = os.path.expanduser("~/drl/unitree-application")
if base_dir not in sys.path:
    sys.path.append(base_dir)

from src.unitree import Unitree

robot_id = 165
unitree = Unitree(robot_id)
unitree.move_home()
p_place = np.array([0.52, 0., 0.075])
unitree.pick_and_place(p_place)
p_place = np.array([0.52, 0., 0.15])
unitree.pick_and_place(p_place)
unitree.stop()

