import time
import numpy as np
import pinocchio as pin
from multiprocessing import Array, Lock

from avp_teleoperate.teleop.robot_control.robot_arm import G1_29_ArmController
from avp_teleoperate.teleop.robot_control.robot_arm_ik import G1_29_ArmIK
from avp_teleoperate.teleop.robot_control.robot_hand_unitree import Dex3_1_Controller

if __name__ == "__main__":
    print("Instantiate controllers")
    arm_ctrl = G1_29_ArmController()
    arm_ik = G1_29_ArmIK()

    print("Return to home pose")
    arm_ctrl.ctrl_dual_arm_go_home()

    # initial positon
    T_left_init = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, +0.25, 0.1]),
    )
    T_right_init = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, -0.25, 0.1]),
    )

    dt = 0.05
    duration = 5
    n = int(duration / dt)
    delta_translation = np.array([0.1, 0, 0])
    T_left = [pin.SE3(T_left_init.rotation, T_left_init.translation + delta_translation * i / n) for i in range(n)]
    T_right = [pin.SE3(T_right_init.rotation, T_right_init.translation + delta_translation * i / n) for i in range(n)]

    print("Run trajectory")
    arm_ctrl.speed_gradual_max(t=2.0)  # Avoid jerky start
    for i in range(n):
        start_time = time.time()

        # get current goal pose
        left_wrist = T_left[i]
        right_wrist = T_right[i]

        # get current state data.
        current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
        current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

        # solve ik using motor data and wrist pose, then use ik results to control arms.
        sol_q, sol_tauff  = arm_ik.solve_ik(left_wrist, right_wrist, current_lr_arm_q, current_lr_arm_dq)
        arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

        current_time = time.time()
        time_elapsed = current_time - start_time
        sleep_time = max(0, dt - time_elapsed)
        time.sleep(sleep_time)

    # # TODO actuate grippers
    # left_hand_array = Array('d', 75, lock = True)          # [input]
    # right_hand_array = Array('d', 75, lock = True)         # [input]
    # dual_hand_data_lock = Lock()
    # dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
    # dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
    # hand_ctrl = Dex3_1_Controller(left_hand_array, right_hand_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array)

    print("Return to home pose")
    arm_ctrl.ctrl_dual_arm_go_home()
