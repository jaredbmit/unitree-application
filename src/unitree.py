from controller import Controller
from planner import Planner
from perception import Perception

class Unitree:
    def __init__(self, robot_id: int):
        assert robot_id in [164, 165]  # End of IP address

        # Initialize subsystems
        self.controller = Controller()
        self.planner = Planner(robot_id)
        self.perception = Perception()

        # Ensure robot moves home first
        self.initialized = False

    def move_home(self):
        print("Moving to home pose.")
        traj_init = self.planner.plan_init()
        self.controller.load_trajectory(traj_init)
        self.controller.run_trajectory()
        self.initialized = True

    def pick_and_place(self):
        if not self.initialized:
            raise RuntimeError("`move_home` must be called first.")
        
        # Find brick
        print("Looking for brick.")
        T_C_to_brick = self.perception.estimate_brick_pose()
        if T_C_to_brick is None:
            raise RuntimeError("No valid brick found from `estimate_brick_pose`")
        print(f"Brick found at: {T_C_to_brick}.")

        raise NotImplementedError

        # Transform to world
        # TODO get unitree world to camera pose
        pass

        # Plan trajectory
        # TODO how should we choose place location?
        pass

        # Control arms
        # TODO
        pass

    def stop(self):
        self.perception.stop()