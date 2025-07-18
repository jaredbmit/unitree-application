import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

from src.controller import Controller
from src.planner import Planner
from src.perception import Perception

model_path = "../unitree_ros/robots/g1_description"
urdf_filename = "g1_29dof.urdf"
urdf_path = model_path + "/" + urdf_filename

class Unitree:
    def __init__(self, robot_id: int):
        assert robot_id in [164, 165]  # End of IP address

        # Initialize subsystems
        self.controller = Controller()
        self.planner = Planner(robot_id)
        self.perception = Perception()

        # Ensure robot moves home first
        self.initialized = False

        # Load static URDF
        self.robot = RobotWrapper.BuildFromURDF(urdf_path, [model_path])
        model = self.robot.model
        data = self.robot.data
        q_neutral = pin.neutral(model)
        pin.forwardKinematics(model, data, q_neutral)
        pin.updateFramePlacements(model, data)
        
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

    def pick_and_place(self, T_place):
        if not self.initialized:
            raise RuntimeError("`move_home` must be called first.")
        
        # Find brick
        print("Looking for brick.")
        T_camera_to_brick = self.perception.estimate_brick_pose()
        if T_camera_to_brick is None:
            raise RuntimeError("No valid brick found from `estimate_brick_pose`")
        print(f"Brick found at: {T_camera_to_brick}.")

        # Transform coordinates
        T_world_to_camera = self.get_pose("d435_link")
        T_world_to_brick = T_world_to_camera @ T_camera_to_brick

        # Plan trajectory
        print("Planning trajectory.")
        side = "left" if T_world_to_brick[1, 3] > 0 else "right"  # LHS or RHS
        traj = self.planner.plan_pick_place(T_world_to_brick, T_place, side=side)

        # Control arms
        print("Running trajectory.")
        self.controller.load_trajectory(traj)
        self.controller.run_trajectory()

    def stop(self):
        self.perception.stop()
