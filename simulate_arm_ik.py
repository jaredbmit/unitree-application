import time
import numpy as np
import pinocchio as pin

from avp_teleoperate.teleop.robot_control.robot_arm_ik import G1_29_ArmIK

if __name__ == "__main__":
    print("Instantiate IK")
    arm_ik = G1_29_ArmIK(Visualization = True)

    # Simulate starting from rest
    current_lr_arm_q = arm_ik.init_data.copy()
    current_lr_arm_dq = np.zeros_like(current_lr_arm_q)

    # Parameters
    delta_x = 0.1  # trajectory distance
    dt = 0.05  # 20 Hz

    # initial positon
    T_left_init = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, +0.25, 0.1]),
    )
    T_right_init = pin.SE3(
        pin.Quaternion(1, 0, 0, 0),
        np.array([0.25, -0.25, 0.1]),
    )

    for i in range(50):
        # Solve for wrist target
        T_left_target = pin.SE3(
            T_left_init.rotation,
            T_left_init.translation + np.array([delta_x / 100 * i, 0, 0]),
        ).homogeneous
        T_right_target = pin.SE3(
            T_right_init.rotation,
            T_right_init.translation + np.array([delta_x / 100 * i, 0, 0]),
        ).homogeneous

        # Automatically visualizes in meshcat
        sol_q, sol_tauff  = arm_ik.solve_ik(T_left_target, T_right_target, current_lr_arm_q, current_lr_arm_dq)

        # Update simulated current state
        current_lr_arm_dq = (sol_q - current_lr_arm_q) / dt
        current_lr_arm_q = sol_q.copy()

        time.sleep(dt)

    init_lr_arm_q = current_lr_arm_q.copy()
    d_l_hand_q = np.array([-1, -1, -1, -1, 0, 0, 1]) * 2/3
    d_r_hand_q = np.array([1, 1, 1, 1, 0, 0, -1]) * 2/3
    for i in range(50):
        current_lr_arm_q = init_lr_arm_q + np.concatenate(
            [np.zeros(7), i / 50 * d_l_hand_q, np.zeros(7), i / 50 * d_r_hand_q]
        )
        arm_ik.visualize_lr_arm_motor_q(current_lr_arm_q)
        time.sleep(dt)
