import time
import numpy as np
import pinocchio as pin
from multiprocessing import Array, Lock

from avp_teleoperate.teleop.robot_control.robot_arm import G1_29_ArmController
from avp_teleoperate.teleop.robot_control.robot_arm_ik import G1_29_ArmIK
from avp_teleoperate.teleop.robot_control.robot_hand_unitree import Dex3_1_Simple_Controller

if __name__ == "__main__":
    print("Instantiate controllers")
    arm_ctrl = G1_29_ArmController()
    arm_ik = G1_29_ArmIK()
    hand_ctrl = Dex3_1_Simple_Controller()

    # initial positon
    T_left_init = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, +0.25, 0.1]),
    )
    T_right_init = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, -0.25, 0.1]),
    )

    dt = 0.01
    duration = 2
    n = int(duration / dt)
    delta_translation = np.array([0.1, 0, 0.25])
    delta_angle = - np.pi / 2
    T_left = [T_left_init for i in range(n)]
    T_right = [
        pin.SE3(
            pin.Quaternion(np.cos(delta_angle * i / (n // 1) / 2), np.sin(delta_angle * i / (n // 2) / 2), 0, 0),
            T_right_init.translation + delta_translation * i / (n // 2)
        ) 
        for i in range(n // 2)
    ] + [
        pin.SE3(
            pin.Quaternion(np.cos(delta_angle / 2), np.sin(delta_angle / 2), 0, 0),
            T_right_init.translation + delta_translation - np.array([-0.1, 0, 0.15]) * i / (n // 2)
        )
        for i in range(n // 2)
    ]

    T_left = T_left * 2
    T_right = T_right + (T_right[::-1])[1:]

    print("Wait for warmup")
    time.sleep(2.5)

    print("Command 0")
    hand_ctrl.ctrl_dual_hand_binary(False, False)
    time.sleep(1)

    print("Run trajectory")
    arm_ctrl.speed_gradual_max(t=1.0)  # Avoid jerky start
    for i in range(len(T_right)):
        if i == len(T_right) // 2:
            hand_ctrl.ctrl_dual_hand_binary(True, True)
            time.sleep(0.5)

        start_time = time.time()

        # get current goal pose
        left_wrist = T_left[i]
        right_wrist = T_right[i]

        # get current state data.
        current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
        current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

        # solve ik using motor data and wrist pose, then use ik results to control arms.
        sol_q, sol_tauff  = arm_ik.solve_ik(left_wrist.homogeneous, right_wrist.homogeneous, current_lr_arm_q, current_lr_arm_dq)
        arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

        current_time = time.time()
        time_elapsed = current_time - start_time
        sleep_time = max(0, dt - time_elapsed)
        time.sleep(sleep_time)
    time.sleep(0.5)

    print("Command 2")
    hand_ctrl.ctrl_dual_hand_binary(False, False)
    time.sleep(1)

    print("Finished")
