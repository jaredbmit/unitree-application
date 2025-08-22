import os
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from typing import List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    
    def build_from_keypoints_debug(
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

            self.time += [-1.0] + [t + st for st in segment_time][1:]
            self.T_left += segment_T_left
            self.T_right += segment_T_right
            self.G_left += [G_left[i]] * len(segment_T_left)
            self.G_right += [G_right[i]] * len(segment_T_right)

            t = self.time[-1]
        keypoints_left = []
        keypoints_right = []
        self.time.append(sum(segment_durations))
        print("======= PLANNED TRAJECTORY =======")
        for t in range(len(T_right) - 1):
            print(str(self.time[t]) + " pose: " + str(T_left[t]))
            keypoints_left.append(T_left[t][0:3,3])
            keypoints_right.append(T_right[t][0:3,3])
        return np.asarray(keypoints_left), np.asarray(keypoints_right)

    def plot(self, ax, downsample=10):
        scale = 0.1
        for i in range(len(self) // downsample):
            j = i * downsample
            TL = self.T_left[j]
            TR = self.T_right[j]
            ax.quiver(*TL[:3,3], *TL[:3,0], length=scale, color='r')
            ax.quiver(*TL[:3,3], *TL[:3,1], length=scale, color='g')
            ax.quiver(*TL[:3,3], *TL[:3,2], length=scale, color='b')
            ax.quiver(*TR[:3,3], *TR[:3,0], length=scale, color='r')
            ax.quiver(*TR[:3,3], *TR[:3,1], length=scale, color='g')
            ax.quiver(*TR[:3,3], *TR[:3,2], length=scale, color='b')

    def __len__(self):
        return len(self.T_left)

    def __getitem__(self, i):
        return self.time[i], self.T_left[i], self.T_right[i], self.G_left[i], self.G_right[i]


class Planner:
    """
    Simple gripper trajectory planning for pick and place
    """
    DT = 0.01

    def __init__(self, robot_id: int, log_dir=None):
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
        self.p_B_to_G_W = np.array([-0.05, 0., 0.11])  # TODO tune
        self.p_point_to_pre_W = np.array([-0.05, 0., 0.2])  # TODO tune

        # Logging
        self.log_dir = log_dir

    def plan_init(self):
        """Initial trajectory"""
        # Construct sequences
        T_W_to_G_left = [self.T_left_preinit.homogeneous, self.T_left_init.homogeneous]
        T_W_to_G_right = [self.T_right_preinit.homogeneous, self.T_right_init.homogeneous]
        cmd_left = [self.cmd_open_left] * 2
        cmd_right = [self.cmd_open_right] * 2
        segment_durations = [1.5]

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

        if self.log_dir is not None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            traj.plot(ax)
            ax.set_aspect('equal')
            fig.suptitle("initialization trajectory")
            fig.savefig(os.path.join(self.log_dir, "traj_init.png"))

        return traj

    def plan_pick_and_place(self, T_pick: np.ndarray, T_place: np.ndarray, side: str):
        """
        T_pick and T_place are brick poses wrt world
        Planning is gripper pose wrt world
        """
        # TODO !!
        # Currently brick is assumed to be "aligned" with robot, rotationally
        # Extend to arbitary brick rotations by creating a map from brick rotation to gripper rotation

        # Define keypoint positions
        p_W_to_pick_W = T_pick[:3, 3] + self.p_B_to_G_W
        p_W_to_place_W = T_place[:3, 3] + self.p_B_to_G_W
        p_W_to_prepick_W = p_W_to_pick_W + self.p_point_to_pre_W
        p_W_to_preplace_W = p_W_to_place_W + self.p_point_to_pre_W

        if side == "left":
            # Keypoint poses
            T_W_to_pick = pin.SE3(self.left_q, p_W_to_pick_W)
            T_W_to_place = pin.SE3(self.left_q, p_W_to_place_W)
            T_W_to_prepick = pin.SE3(self.left_q, p_W_to_prepick_W)
            T_W_to_preplace = pin.SE3(self.left_q, p_W_to_preplace_W)
            
            # Construct sequences
            T_W_to_G_left = [
                self.T_left_init.homogeneous,
                T_W_to_prepick.homogeneous,
                T_W_to_pick.homogeneous,
                T_W_to_prepick.homogeneous,
                T_W_to_preplace.homogeneous,
                T_W_to_place.homogeneous,
                T_W_to_preplace.homogeneous,
                T_W_to_prepick.homogeneous,
                self.T_left_init.homogeneous,
            ]
            cmd_left = [
                self.cmd_open_left,
                self.cmd_open_left,
                self.cmd_close_left,
                self.cmd_close_left,
                self.cmd_close_left,
                self.cmd_open_left,
                self.cmd_open_left,
                self.cmd_open_left,
                self.cmd_open_left,
            ]
            T_W_to_G_right = [self.T_right_init.homogeneous] * 9
            cmd_right = [self.cmd_open_right] * 9
        elif side == "right":
            # Keypoint poses
            T_W_to_pick = pin.SE3(self.right_q, p_W_to_pick_W)
            T_W_to_place = pin.SE3(self.right_q, p_W_to_place_W)
            T_W_to_prepick = pin.SE3(self.right_q, p_W_to_prepick_W)
            T_W_to_preplace = pin.SE3(self.right_q, p_W_to_preplace_W)

            # Construct sequences
            T_W_to_G_right = [
                self.T_right_init.homogeneous,
                T_W_to_prepick.homogeneous,
                T_W_to_pick.homogeneous,
                T_W_to_prepick.homogeneous,
                T_W_to_preplace.homogeneous,
                T_W_to_place.homogeneous,
                T_W_to_preplace.homogeneous,
                T_W_to_prepick.homogeneous,
                self.T_right_init.homogeneous,
            ]
            cmd_right = [
                self.cmd_open_right,
                self.cmd_open_right,
                self.cmd_close_right,
                self.cmd_close_right,
                self.cmd_close_right,
                self.cmd_open_right,
                self.cmd_open_right,
                self.cmd_open_right,
                self.cmd_open_right,
            ]
            T_W_to_G_left = [self.T_left_init.homogeneous] * 9
            cmd_left = [self.cmd_open_left] * 9
        else:
            raise ValueError("`side` must be either 'left' or 'right'")
        
        # Timing
        segment_durations = [1.5] * 8

        # Build trajectory
        traj = Trajectory()
        traj.build_from_keypoints_debug(
            T_W_to_G_left,
            T_W_to_G_right,
            cmd_left,
            cmd_right,
            segment_durations,
            self.DT,
        )
        
        if self.log_dir is not None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            traj.plot(ax, downsample=30)
            ax.set_aspect('equal')
            fig.suptitle("pick and place trajectory")
            fig.savefig(os.path.join(self.log_dir, "traj_pick_and_place.png"))

        return traj

    def plan_pick_and_place_debug(self, T_pick: np.ndarray, T_place: np.ndarray, side: str):
        """
        T_pick and T_place are brick poses wrt world
        Planning is gripper pose wrt world
        """
        # TODO !!
        # Currently brick is assumed to be "aligned" with robot, rotationally
        # Extend to arbitary brick rotations by creating a map from brick rotation to gripper rotation

        # Define keypoint positions
        p_W_to_pick_W = T_pick[:3, 3] + self.p_B_to_G_W
        p_W_to_place_W = T_place[:3, 3] + self.p_B_to_G_W
        p_W_to_prepick_W = p_W_to_pick_W + self.p_point_to_pre_W
        p_W_to_preplace_W = p_W_to_place_W + self.p_point_to_pre_W

        if side == "left":
            # Keypoint poses
            T_W_to_pick = pin.SE3(self.left_q, p_W_to_pick_W)
            T_W_to_place = pin.SE3(self.left_q, p_W_to_place_W)
            T_W_to_prepick = pin.SE3(self.left_q, p_W_to_prepick_W)
            T_W_to_preplace = pin.SE3(self.left_q, p_W_to_preplace_W)
            
            # Construct sequences
            T_W_to_G_left = [
                self.T_left_init.homogeneous,
                T_W_to_prepick.homogeneous,
                T_W_to_pick.homogeneous,
                T_W_to_prepick.homogeneous,
                T_W_to_preplace.homogeneous,
                T_W_to_place.homogeneous,
                T_W_to_preplace.homogeneous,
                T_W_to_prepick.homogeneous,
                self.T_left_init.homogeneous,
            ]
            cmd_left = [
                self.cmd_open_left,
                self.cmd_open_left,
                self.cmd_close_left,
                self.cmd_close_left,
                self.cmd_close_left,
                self.cmd_open_left,
                self.cmd_open_left,
                self.cmd_open_left,
                self.cmd_open_left,
            ]
            T_W_to_G_right = [self.T_right_init.homogeneous] * 9
            cmd_right = [self.cmd_open_right] * 9
        elif side == "right":
            # Keypoint poses
            T_W_to_pick = pin.SE3(self.right_q, p_W_to_pick_W)
            T_W_to_place = pin.SE3(self.right_q, p_W_to_place_W)
            T_W_to_prepick = pin.SE3(self.right_q, p_W_to_prepick_W)
            T_W_to_preplace = pin.SE3(self.right_q, p_W_to_preplace_W)

            # Construct sequences
            T_W_to_G_right = [
                self.T_right_init.homogeneous,
                T_W_to_prepick.homogeneous,
                T_W_to_pick.homogeneous,
                T_W_to_prepick.homogeneous,
                T_W_to_preplace.homogeneous,
                T_W_to_place.homogeneous,
                T_W_to_preplace.homogeneous,
                T_W_to_prepick.homogeneous,
                self.T_right_init.homogeneous,
            ]
            cmd_right = [
                self.cmd_open_right,
                self.cmd_open_right,
                self.cmd_close_right,
                self.cmd_close_right,
                self.cmd_close_right,
                self.cmd_open_right,
                self.cmd_open_right,
                self.cmd_open_right,
                self.cmd_open_right,
            ]
            T_W_to_G_left = [self.T_left_init.homogeneous] * 9
            cmd_left = [self.cmd_open_left] * 9
        else:
            raise ValueError("`side` must be either 'left' or 'right'")
        
        # Timing
        segment_durations = [1.5] * 8

        # Build trajectory
        traj = Trajectory()
        keypoints_left, keypoints_right = traj.build_from_keypoints_debug(
            T_W_to_G_left,
            T_W_to_G_right,
            cmd_left,
            cmd_right,
            segment_durations,
            self.DT,
        )
        
        if self.log_dir is not None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            traj.plot(ax, downsample=30)
            ax.set_aspect('equal')
            fig.suptitle("pick and place trajectory")
            fig.savefig(os.path.join(self.log_dir, "traj_pick_and_place.png"))
        keypoints = []
        if side == "right":
            keypoints = keypoints_right
        else:
            keypoints = keypoints_left
        return traj, keypoints

if __name__ == "__main__":
    planner = Planner(robot_id = 165)
    T_pick  = np.array([[0.99, 0.07, -0.14, 0.41],[-0.08, 0.99, -0.05, 0.02],[0.13, 0.06, 0.98, 0.02],[0, 0, 0, 1]])
    T_place = np.array([[0.99, 0.07, -0.14, 0.52],[-0.08, 0.99, -0.05, 0.00],[0.13, 0.06, 0.98, 0.15],[0, 0, 0, 1]])
    planner.plan_pick_and_place(T_pick,T_place,"left")
