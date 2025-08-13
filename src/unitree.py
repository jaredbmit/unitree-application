import os
import shutil
import time
from copy import deepcopy
import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.controller import Controller
from src.planner import Planner
from src.perception import Perception

model_path = os.path.expanduser("~/drl/unitree-application/avp_teleoperate/assets/g1")
urdf_filename = "g1_body29_hand14.urdf"
urdf_path = model_path + "/" + urdf_filename

class Unitree:
    def __init__(self, robot_id: int):
        assert robot_id in [164, 165]  # End of IP address

        # Load static URDF
        self.robot = RobotWrapper.BuildFromURDF(urdf_path, [model_path])
        model = self.robot.model
        data = self.robot.data
        q_neutral = pin.neutral(model)
        pin.forwardKinematics(model, data, q_neutral)
        pin.updateFramePlacements(model, data)
        
        # log output
        self.log_dir = os.path.abspath(os.path.join(os.getcwd(), "../log/"))
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        os.mkdir(self.log_dir)
        
        # Initialize subsystems
        self.perception = Perception(self.log_dir)
        self.planner = Planner(robot_id, self.log_dir)
        self.controller = Controller()
        print("Waiting for controller warm-up.")
        time.sleep(2)
        print("Finished waiting.")

        # Ensure robot moves home first
        self.initialized = False

    def get_pose(self, frame_name):
        frame_id = self.robot.model.getFrameId(frame_name)
        T_world_to_frame = self.robot.data.oMf[frame_id].homogeneous
        return T_world_to_frame

    def move_home(self):
        print("Moving to home pose.")
        traj_init = self.planner.plan_init()
        self.controller.load_trajectory(traj_init)
        self.controller.run_trajectory()
        self.initialized = True

    def pick_and_place(self, p_place):
        if not self.initialized:
            raise RuntimeError("`move_home` must be called first.")
        
        # Find brick
        T_camera_to_brick = self.perception.estimate_brick_pose()

        # Transform coordinates
        T_world_to_camera = self.get_pose("d435_link")
        T_world_to_brick = T_world_to_camera @ T_camera_to_brick
        print(f"Estimated brick pose wrt world frame: {T_world_to_brick}")
        
        # Log transforms
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        def plot_pose(T, ax, label):
            scale = 0.1
            ax.quiver(*T[:3,3], *T[:3,0], length=scale, color='r')
            ax.quiver(*T[:3,3], *T[:3,1], length=scale, color='g')
            ax.quiver(*T[:3,3], *T[:3,2], length=scale, color='b')
            ax.text(*T[:3,3], label, color='black', fontsize=10)
        plot_pose(np.eye(4), ax, "origin")
        plot_pose(T_world_to_brick, ax, "brick")
        plot_pose(T_world_to_camera, ax, "camera")
        ax.set_aspect("equal")
        fig.savefig(os.path.join(self.log_dir, "transforms.png"))

        # Temporary ~
        T_place = deepcopy(T_world_to_brick)
        T_place[:3, 3] = p_place

        # Plan trajectory
        print("Planning trajectory.")
        print("T_pick:")
        print(str(T_world_to_brick))
        print("T_place:")
        print(str(T_place))
        
        if T_world_to_brick[1, 3] > 0:  # LHS or RHS
            side = "left"
            T_place[1, 3] += -0.05
        else:
            side = "right"
            T_place[1, 3] += 0.05
        traj = self.planner.plan_pick_and_place(T_world_to_brick, T_place, side=side)

        # Control arms
        print("Running trajectory.")
        self.controller.load_trajectory(traj)
        self.controller.run_trajectory()

        # Log
        print(f"Program finished. Find outputs in {self.log_dir}")

    def stop(self):
        self.perception.stop()
