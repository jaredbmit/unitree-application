import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from typing import List


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


class Trajectory:
    def __init__(self):
        self.time = []
        self.T_left = []
        self.T_right = []
        self.G_left = []
        self.G_right = []

    def build_from_trajectory(
        self,
        time: List,
        T_left: List[np.ndarray],
        T_right: List[np.ndarray],
        G_left: List[np.ndarray],
        G_right: List[np.ndarray],
    ):
        """Stores pre-built trajectory"""
        assert len(time) == len(T_left) == len(T_right) == len(G_left) == len(G_right)
        self.time = time
        self.T_left = T_left
        self.T_right = T_right
        self.G_left = G_left
        self.G_right = G_right

    def build_from_keypoints(
        self,
        T_left: List[np.ndarray],
        T_right: List[np.ndarray],
        G_left: List[np.ndarray],
        G_right: List[np.ndarray],
        segment_durations: List[float],
        dt: float
    ):
        """Interpolates trajectory from keyframes. ZOH for gripper."""
        assert len(T_left) == len(T_right) == len(G_left) == len(G_right) == len(segment_durations) + 1
        t = 0
        self.time = []
        self.T_left = []
        self.T_right = []
        self.G_left = []
        self.G_right = []

        for i in range(len(T_right) - 1):
            duration_per_segment = segment_durations[i]
            segment_T_left, segment_time = interpolate_pose(T_left[i], T_left[i+1], duration_per_segment, dt)
            segment_T_right, _ = interpolate_pose(T_right[i], T_right[i+1], duration_per_segment, dt)

            if i == 0:
                self.time += [t + st for st in segment_time]
                self.T_left += segment_T_left
                self.T_right += segment_T_right
                self.G_left += [G_left[i]] * len(segment_T_left)
                self.G_right += [G_right[i]] * len(segment_T_right)
            else:
                self.time += [t + st for st in segment_time][1:]
                self.T_left += segment_T_left[1:]
                self.T_right += segment_T_right[1:]
                self.G_left += [G_left[i]] * (len(segment_T_left) - 1)
                self.G_right += [G_right[i]] * (len(segment_T_right) - 1)

            t = self.time[-1]

        self.time.append(sum(segment_durations))
        self.T_left.append(T_left[-1])
        self.T_right.append(T_right[-1])
        self.G_left.append(G_left[-1])
        self.G_right.append(G_right[-1])

    def __getitem__(self, i):
        return self.time[i], self.T_left[i], self.T_right[i], self.G_left[i], self.G_right[i]


class Planner:
    """
    Simple gripper trajectory planning for pick and place
    """
    DT = 0.01

    def __init__(self, robot_id: int):
        assert robot_id in [164, 165]  # End of IP address

        # Gripper configuration is hardware dependent
        if robot_id == 165:
            self.cmd_close_left = np.array([0., -0.7, -0.7, 0.1, 1., -0.7, -0.7]) * np.pi / 3
            self.cmd_open_left = np.array([0., 0.1, 0.1, -0.3, 0., 0.1, 0.1]) * np.pi / 3
            self.cmd_close_right = np.array([0., 0.7, 0.7, 0.7, 0.7, -0.1, -1.]) * np.pi / 3
            self.cmd_open_right = np.array([0., -0.1, -0.1, -0.1, -0.1, 0.3, 0.]) * np.pi / 3
        elif robot_id == 164:
            self.cmd_close_left = np.array([0, 0.25, 1, 1.5, 1.5, 0.4, 1.5]) * np.pi / 3
            self.cmd_open_left = np.array([0, -0.5, 0, 0, 0, -1.1, 0]) * np.pi / 3
            self.cmd_close_right = np.array([0, 0.25, 1, 0.4, 1.5, 1.5, 1.5]) * np.pi / 3
            self.cmd_open_right = np.array([0, -0.5, 0, -1.1, 0, 0, 0]) * np.pi / 3

        # Initialization poses
        self.T_left_preinit = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            np.array([0.25, 0.15, 0.1]),
        )
        self.T_right_preinit = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            np.array([0.25, -0.15, 0.1]),
        )
        self.T_left_init = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            np.array([0.15, 0.25, 0.1]),
        )
        self.T_right_init = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            np.array([0.15, -0.25, 0.1]),
        )

        # Nominal gripper rotations
        left_q1 = pin.Quaternion(np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0)
        left_q2 = pin.Quaternion(np.cos(np.pi / 16), 0, np.sin(np.pi / 16), 0)
        self.left_q = left_q2 * left_q1
        right_q1 = pin.Quaternion(np.cos(-np.pi / 4), np.sin(-np.pi / 4), 0, 0)
        right_q2 = pin.Quaternion(np.cos(np.pi / 16), 0, np.sin(np.pi / 16), 0)
        self.right_q = right_q2 * right_q1

        # Offsets
        self.p_B_to_G_W = np.array([0., 0., 0.05])  # TODO tune
        self.p_point_to_pre_W = np.array([0., 0., 0.1])  # TODO tune

    def plan_init(self):
        """Initial trajectory"""
        # Construct sequences
        T_W_to_G_left = [self.T_left_preinit.homogeneous, self.T_left_init.homogeneous]
        T_W_to_G_right = [self.T_right_preinit.homogeneous, self.T_right_init.homogeneous]
        cmd_left = [self.cmd_open_left] * 2
        cmd_right = [self.cmd_open_right] * 2
        segment_durations = [1.5, 1.5]

        # Build trajectory
        traj = Trajectory()
        traj.build_from_keypoints(
            T_W_to_G_left,
            T_W_to_G_right,
            cmd_left,
            cmd_right,
            segment_durations,
            self.DT,
        )

        return traj

    def plan_pick_place(self, T_pick: np.ndarray, T_place: np.ndarray, side: str):
        """
        T_pick and T_place are brick poses wrt world
        Planning is gripper pose wrt world
        """
        if side == "left":
            T_W_to_init = self.T_left_init
            T_W_to_init_off = self.T_right_init
            quat_W_to_G = self.left_q
            cmd_open_side = self.cmd_open_left
            cmd_close_side = self.cmd_close_left
            cmd_open_off = self.cmd_open_right
        elif side == "right":
            T_W_to_init = self.T_right_init
            T_W_to_init_off = self.T_left_init
            quat_W_to_G = self.right_q
            cmd_open_side = self.cmd_open_right
            cmd_close_side = self.cmd_close_right
            cmd_open_off = self.cmd_open_left
        else:
            raise ValueError("`side` must be either 'left' or 'right'")

        # Define keypoint poses
        p_W_to_pick_W = T_pick[:3, 3] + self.p_B_to_G_W
        p_W_to_place_W = T_place[:3, 3] + self.p_B_to_G_W
        p_W_to_prepick_W = p_W_to_pick_W + self.p_point_to_pre_W
        p_W_to_preplace_W = p_W_to_place_W + self.p_point_to_pre_W
        T_W_to_pick = pin.SE3(quat_W_to_G, p_W_to_pick_W)
        T_W_to_place = pin.SE3(quat_W_to_G, p_W_to_place_W)
        T_W_to_prepick = pin.SE3(quat_W_to_G, p_W_to_prepick_W)
        T_W_to_preplace = pin.SE3(quat_W_to_G, p_W_to_preplace_W)

        # Construct sequences
        T_W_to_G_side = [
            T_W_to_init.homogeneous,
            T_W_to_prepick.homogeneous,
            T_W_to_pick.homogeneous,
            T_W_to_prepick.homogeneous,
            T_W_to_preplace.homogeneous,
            T_W_to_place.homogeneous,
            T_W_to_preplace.homogeneous,
            T_W_to_prepick.homogeneous,
            T_W_to_init.homogeneous,
        ]
        cmd_side = [
            cmd_open_side,
            cmd_open_side,
            cmd_close_side,
            cmd_close_side,
            cmd_close_side,
            cmd_open_side,
            cmd_open_side,
            cmd_open_side,
            cmd_open_side,
        ]
        T_W_to_G_off = [T_W_to_init_off.homogeneous] * 9
        cmd_off = [cmd_open_off] * 9

        # Timing
        segment_durations = [1.5] * (len(T_W_to_G_side) - 1)

        # Build trajectory
        traj = Trajectory()
        if side == "left":
            T_left = T_W_to_G_side
            T_right = T_W_to_G_off
            cmd_left = cmd_side
            cmd_right = cmd_off
        elif side == "right":
            T_left = T_W_to_G_off
            T_right = T_W_to_G_side
            cmd_left = cmd_off
            cmd_right = cmd_side
        traj.build_from_keypoints(
            T_left,
            T_right,
            cmd_left,
            cmd_right,
            segment_durations,
            self.DT,
        )

        return traj