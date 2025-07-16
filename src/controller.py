import time

from planner import Trajectory
from avp_teleoperate.teleop.robot_control.robot_arm import G1_29_ArmController
from avp_teleoperate.teleop.robot_control.robot_arm_ik import G1_29_ArmIK
from avp_teleoperate.teleop.robot_control.robot_hand_unitree import Dex3_1_Simple_Controller

class Controller:
    def __init__(self):
        # Teleop interfaces
        self.arm_ctrl = G1_29_ArmController()
        self.arm_ik = G1_29_ArmIK()
        self.hand_ctrl = Dex3_1_Simple_Controller()

        # Trajectory buffer
        self.traj = None

    def load_trajectory(self, traj: Trajectory):
        self.traj = traj

    def run_trajectory(self):
        if self.traj is None:
            print("No trajectory loaded. returning..")
            return 
        
        print("Wait for warmup")
        time.sleep(2)

        print("Running trajectory")
        self.arm_ctrl.speed_gradual_max(t=1.0)  # Avoid jerky start
        t_prev = 0.
        for t, TR, TL, GR, GL in self.traj:
            start_time = time.time()
            self.hand_ctrl.ctrl_dual_hand(GL, GR)

            # get current state data.
            current_lr_arm_q  = self.arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = self.arm_ctrl.get_current_dual_arm_dq()

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            sol_q, sol_tauff  = self.arm_ik.solve_ik(TL, TR, current_lr_arm_q, current_lr_arm_dq)
            self.arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (t - t_prev) - time_elapsed)
            time.sleep(sleep_time)
            t_prev = t