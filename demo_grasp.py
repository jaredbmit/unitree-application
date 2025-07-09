import time
import numpy as np
import pinocchio as pin
from multiprocessing import Array, Lock
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from typing import List

from avp_teleoperate.teleop.robot_control.robot_arm import G1_29_ArmController
from avp_teleoperate.teleop.robot_control.robot_arm_ik import G1_29_ArmIK
from avp_teleoperate.teleop.robot_control.robot_hand_unitree import Dex3_1_Simple_Controller

# I have no idea why but the hand joint mappings seem completely different
# depending on which unitree is being used. These were manually tested to find
# (roughly) equivalent notions of "closing" and "opening" the gripper
robot_id = 165
if robot_id == 165:
    left_q_close = np.array([0., -0.7, -0.7, 0.1, 1., -0.7, -0.7]) * np.pi / 3
    left_q_open = np.array([0., 0.1, 0.1, -0.3, 0., 0.1, 0.1]) * np.pi / 3
    right_q_close = np.array([0., 0.7, 0.7, 0.7, 0.7, -0.1, -1.]) * np.pi / 3
    right_q_open = np.array([0., -0.1, -0.1, -0.1, -0.1, 0.3, 0.]) * np.pi / 3
elif robot_id == 164:
    left_q_close = np.array([0, 0.25, 1, 1.5, 1.5, 0.4, 1.5]) * np.pi / 3
    left_q_open = np.array([0, -0.5, 0, 0, 0, -1.1, 0]) * np.pi / 3
    right_q_close = np.array([0, 0.25, 1, 0.4, 1.5, 1.5, 1.5]) * np.pi / 3
    right_q_open = np.array([0, -0.5, 0, -1.1, 0, 0, 0]) * np.pi / 3
else:
    raise ValueError("Specify which unitree you're using!!! (164 or 165).")

class Trajectory:
    def __init__(self):
        self.time = []
        self.T_right = []
        self.T_left = []
        self.G_right = []
        self.G_left = []

    def build_from_trajectory(
        self,
        time: List,
        T_right: List[np.ndarray],
        T_left: List[np.ndarray],
        G_right: List[np.ndarray],
        G_left: List[np.ndarray],
    ):
        """Stores pre-built trajectory"""
        assert len(time) == len(T_right) == len(T_left) == len(G_right) == len(G_left)
        self.time = time
        self.T_right = T_right
        self.T_left = T_left
        self.G_right = G_right
        self.G_left = G_left

    def build_from_keypoints(
        self,
        T_right: List[np.ndarray],
        T_left: List[np.ndarray],
        G_right: List[np.ndarray],
        G_left: List[np.ndarray],
        segment_durations: List[float],
        dt: float
    ):
        """Interpolates trajectory from keyframes. ZOH for gripper."""
        assert len(T_right) == len(T_left) == len(G_right) == len(G_left) == len(segment_durations) + 1
        t = 0
        self.time = []
        self.T_right = []
        self.T_left = []
        self.G_right = []
        self.G_left = []

        for i in range(len(T_right) - 1):
            duration_per_segment = segment_durations[i]
            segment_T_right, segment_time = interpolate_pose(T_right[i], T_right[i+1], duration_per_segment, dt)
            segment_T_left, _ = interpolate_pose(T_left[i], T_left[i+1], duration_per_segment, dt)

            if i == 0:
                self.time += [t + st for st in segment_time]
                self.T_right += segment_T_right
                self.T_left += segment_T_left
                self.G_right += [G_right[i]] * len(segment_T_right)
                self.G_left += [G_left[i]] * len(segment_T_left)
            else:
                self.time += [t + st for st in segment_time][1:]
                self.T_right += segment_T_right[1:]
                self.T_left += segment_T_left[1:]
                self.G_right += [G_right[i]] * (len(segment_T_right) - 1)
                self.G_left += [G_left[i]] * (len(segment_T_left) - 1)

            t = self.time[-1]

        self.time.append(sum(segment_durations))
        self.T_right.append(T_right[-1])
        self.T_left.append(T_left[-1])
        self.G_right.append(G_right[-1])
        self.G_left.append(G_left[-1])

    def __getitem__(self, i):
        return self.time[i], self.T_right[i], self.T_left[i], self.G_right[i], self.G_left[i]

def interpolate_pose(T_now: np.ndarray, T_goal: np.ndarray, duration: float, dt: float):
    """
    Interpolates linearly in position and spherically in rotation.
    """
    n_steps = int(duration / dt) + 1
    times = np.linspace(0, 1, n_steps)

    # Extract positions
    p_now = T_now[:3, 3]
    p_goal = T_goal[:3, 3]

    # Extract rotations and create Slerp object
    rot_now = R.from_matrix(T_now[:3, :3])
    rot_goal = R.from_matrix(T_goal[:3, :3])
    key_rots = R.from_matrix([T_now[:3, :3], T_goal[:3, :3]])
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)

    # Interpolate
    interp_rots = slerp(times)
    poses = []

    for i, t in enumerate(times):
        p_interp = (1 - t) * p_now + t * p_goal
        r_interp = interp_rots[i].as_matrix()

        T_interp = np.eye(4)
        T_interp[:3, :3] = r_interp
        T_interp[:3, 3] = p_interp
        poses.append(T_interp)

    return poses, times.tolist()

if __name__ == "__main__":
    arm_ctrl = G1_29_ArmController()
    arm_ik = G1_29_ArmIK()
    hand_ctrl = Dex3_1_Simple_Controller()

    # pre-init
    T_left_preinit = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, 0.15, 0.1]),
    )
    T_right_preinit = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, -0.15, 0.1]),
    )

    # initial positon
    T_left_init = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.15, 0.25, 0.1]),
    )
    T_right_init = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.15, -0.25, 0.1]),
    )

    # Rotations
    left_q1 = pin.Quaternion(np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0)
    left_q2 = pin.Quaternion(np.cos(np.pi / 16), 0, np.sin(np.pi / 16), 0)
    left_q = left_q2 * left_q1
    right_q1 = pin.Quaternion(np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0, 0)
    right_q2 = pin.Quaternion(np.cos(np.pi / 16), 0, np.sin(np.pi / 16), 0)
    right_q = right_q2 * right_q1
    
    # Left gripper keypoints
    T_left_prepick = pin.SE3(left_q, np.array([0.35, 0.25, 0.35]))
    T_left_pick1 = pin.SE3(left_q, np.array([0.45, 0.25, 0.2]))
    T_left_pick2 = pin.SE3(left_q, np.array([0.45, 0.25, 0.14]))
    T_left_predrop = pin.SE3(left_q, np.array([0.35, 0, 0.425]))
    T_left_drop1 = pin.SE3(left_q, np.array([0.45, 0, 0.21]))
    T_left_drop2 = pin.SE3(left_q, np.array([0.45, 0, 0.34]))
    T_left_keypoints_pick1 = [
        T_left_init.homogeneous,
        T_left_prepick.homogeneous,
        T_left_pick1.homogeneous,
        T_left_prepick.homogeneous,
        T_left_predrop.homogeneous,
        T_left_drop1.homogeneous,
        T_left_predrop.homogeneous,
        T_left_prepick.homogeneous,
        T_left_init.homogeneous,
    ]
    T_left_keypoints_pick2 = [
        T_left_init.homogeneous,
        T_left_prepick.homogeneous,
        T_left_pick2.homogeneous,
        T_left_prepick.homogeneous,
        T_left_predrop.homogeneous,
        T_left_drop2.homogeneous,
        T_left_predrop.homogeneous,
        T_left_prepick.homogeneous,
        T_left_init.homogeneous,
    ]
    T_left_keypoints = [] \
        + [T_left_preinit.homogeneous] \
        + [T_left_init.homogeneous] * (len(T_left_keypoints_pick1) - 2) \
        + T_left_keypoints_pick1 \
        + [T_left_init.homogeneous] * (len(T_left_keypoints_pick1) - 4) \
        + T_left_keypoints_pick2 \
        + [T_left_preinit.homogeneous] \

    # Right gripper keypoints
    T_right_prepick = pin.SE3(right_q, np.array([0.35, -0.25, 0.35]))
    T_right_pick1 = pin.SE3(right_q, np.array([0.45, -0.25, 0.20]))
    T_right_pick2 = pin.SE3(right_q, np.array([0.45, -0.25, 0.14]))
    T_right_predrop = pin.SE3(right_q, np.array([0.35, 0, 0.425]))
    T_right_drop1 = pin.SE3(right_q, np.array([0.45, 0, 0.15]))
    T_right_drop2 = pin.SE3(right_q, np.array([0.45, 0, 0.28]))
    T_right_keypoints_pick1 = [
        T_right_init.homogeneous, 
        T_right_prepick.homogeneous, 
        T_right_pick1.homogeneous, 
        T_right_prepick.homogeneous, 
        T_right_predrop.homogeneous, 
        T_right_drop1.homogeneous, 
        T_right_predrop.homogeneous,
        T_right_prepick.homogeneous,
        T_right_init.homogeneous,
    ]
    T_right_keypoints_pick2 = [
        T_right_init.homogeneous,
        T_right_prepick.homogeneous, 
        T_right_pick2.homogeneous, 
        T_right_prepick.homogeneous, 
        T_right_predrop.homogeneous, 
        T_right_drop2.homogeneous, 
        T_right_predrop.homogeneous, 
        T_right_prepick.homogeneous,
        T_right_init.homogeneous,
    ]
    T_right_keypoints = [] \
        + [T_right_preinit.homogeneous] \
        + T_right_keypoints_pick1 \
        + [T_right_init.homogeneous] * (len(T_right_keypoints_pick1) - 4) \
        + T_right_keypoints_pick2 \
        + [T_right_init.homogeneous] * (len(T_right_keypoints_pick1) - 2) \
        + [T_right_preinit.homogeneous] \

    # Left gripper commands
    G_left_keypoints_pick = [
        left_q_open,
        left_q_open,
        left_q_close,
        left_q_close,
        left_q_close,
        left_q_open,
        left_q_open,
        left_q_open,
        left_q_open,
    ]
    G_left_keypoints = [] \
        + [left_q_open] \
        + [left_q_open] * (len(G_left_keypoints_pick) - 2) \
        + G_left_keypoints_pick \
        + [left_q_open] * (len(G_left_keypoints_pick) - 4) \
        + G_left_keypoints_pick \
        + [left_q_open] \
    
    # Right gripper commands
    G_right_keypoints_pick = [
        right_q_open,
        right_q_open,
        right_q_close,
        right_q_close,
        right_q_close,
        right_q_open,
        right_q_open,
        right_q_open,
        right_q_open,
    ]
    G_right_keypoints = [] \
        + [right_q_open] \
        + G_right_keypoints_pick \
        + [right_q_open] * (len(G_right_keypoints_pick) - 4) \
        + G_right_keypoints_pick \
        + [right_q_open] * (len(G_right_keypoints_pick) - 2) \
        + [right_q_open] \

    segment_durations = [1.2] * (len(T_right_keypoints) - 1)
    dt = 0.01

    # Build trajectory
    traj = Trajectory()
    traj.build_from_keypoints(
        T_right_keypoints,
        T_left_keypoints,
        G_right_keypoints,
        G_left_keypoints,
        segment_durations,
        dt,
    )

    print("Wait for warmup")
    time.sleep(2)

    print("Run trajectory")
    arm_ctrl.speed_gradual_max(t=1.0)  # Avoid jerky start
    for t, TR, TL, GR, GL in traj:
        start_time = time.time()
        hand_ctrl.ctrl_dual_hand(GL, GR)

        # get current state data.
        current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
        current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

        # solve ik using motor data and wrist pose, then use ik results to control arms.
        sol_q, sol_tauff  = arm_ik.solve_ik(TL, TR, current_lr_arm_q, current_lr_arm_dq)
        arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

        current_time = time.time()
        time_elapsed = current_time - start_time
        sleep_time = max(0, dt - time_elapsed)
        time.sleep(sleep_time)
    
    print("Finished")
    time.sleep(1)
