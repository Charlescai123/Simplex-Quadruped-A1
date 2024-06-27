import ctypes
import traceback

import matlab
# import matlab.engine
from omegaconf import DictConfig
from numpy import linalg as LA
import matplotlib.pyplot as plt
from collections import deque
from numpy.linalg import pinv

from locomotion.robots.motors import MotorCommand
from locomotion.robots.motors import MotorControlMode
from scipy.linalg import solve_continuous_are, inv
from cvxopt import matrix, solvers

import multiprocessing as mp
import numpy as np
import asyncio
import cvxpy as cp
# import cvxpy as cp
from typing import Tuple, Any
import copy
import enum
import time
import sys

# global lock
# global state_triggered
# global state_update_flag

default_kp = np.diag((0., 0., 100., 100., 100., 0.))
default_kd = np.diag((40., 30., 10., 10., 10., 30.))

# default_kp = np.array([[0, 0, 0, 0, 0, 0],
#                        [0, 0, 0, 0, 0, 0],
#                        [0, 0, 128, 0, 0, 0],
#                        [0, 0, 0, 83, -25, -2],
#                        [0, 0, 0, -33, 80, 2],
#                        [0, 0, 0, 1, 0, 80]])
#
# default_kd = np.array([[39, 0, 0, 0, 0, 0],
#                        [0, 35, 0, 0, 0, 0],
#                        [0, 0, 35, 0, 0, 0],
#                        [0, 0, 0, 37, -1, -9],
#                        [0, 0, 0, -1, 37, 9],
#                        [0, 0, 0, 0, 0, 40]])

manager = mp.Manager()
lock = manager.Lock()
triggered_roll = manager.Value('d', 0)
triggered_pitch = manager.Value('d', 0)
triggered_yaw = manager.Value('d', 0)
state_update_flag = manager.Value('b', 0)
f_kp = manager.list(copy.deepcopy(default_kp.reshape(36)))
f_kd = manager.list(copy.deepcopy(default_kd.reshape(36)))


# queue = manager.Queue()
# queue = asyncio.Queue()


# state_triggered = mp.RawArray(ctypes.c_double, np.array([0] * 12))
# state_triggered = mp.RawArray(ctypes.c_double, np.array([0] * 12))

def mp_start():
    print("creating mat process")
    mat_process = mp.Process(target=update_feedback_gain2,
                             args=(state_update_flag, triggered_roll, triggered_pitch, triggered_yaw, f_kp, f_kd, lock))

    mat_process.daemon = True
    print("starting mat process")
    mat_process.start()
    print(f"Pid of mat process: {mat_process.pid}")
    # mat_process.join()


def mp_start2():
    print("starting mat process pool")
    lmi_run()
    # with mp.Pool(1) as pool:
    #     x = pool.apply_async(update_feedback_gain2,
    #                          args=(state_update_flag, triggered_roll, triggered_pitch, triggered_yaw,))
    #     # x = pool.starmap_async(update_feedback_gain,
    #     #                        [state_update_flag, triggered_roll, triggered_pitch, triggered_yaw])
    #     x.get()


def update_feedback_gain(args):
    # print(f"args input: {args_input}")
    # args = args_input.value
    update_flag, roll, pitch, yaw = args[0], args[1], args[2], args[3]
    try:
        print("Starting a subprocess for LMI computation...")
        path = "./locomotion/ha_teacher"
        # mat_engine = matlab.engine.start_matlab()
        # mat_engine.cd(path)
        # print("Matlab current working directory is ---->>>", mat_engine.pwd())
        while True:
            # _state_trig = await _queue.get()  # get updated trigger state
            print("LMI process computing...")
            print(f"update_flag value: {update_flag.value}")
            print(f"roll value: {roll.value}")
            print(f"pitch value: {pitch.value}")
            print(f"yaw value: {yaw.value}")

            if update_flag.value == 1:
                # lock.acquire()
                print("success!!!")
                # if _lock.acquire():
                # print("Acquiring lock")
                roll_v, pitch_v, yaw_v = roll.value, pitch.value, yaw.value
                print("Obtained new state, updating the feedback gain using matlab engine")
                F_kp, F_kd = feedback_law2(roll_v, pitch_v, yaw_v)
                # F_kp, F_kd = mat_engine.feedback_law2(roll, pitch, yaw, nargout=2)
                # _update_flag.value = 0
                print("Feedback gain is updated now ---->>>")

                # _lock.release()
            time.sleep(1)
    except:
        # traceback.print_exc(file=open("suberror.txt", "w+"))
        error = traceback.format_exc()
        print(f"subprocess error: {error}")
        sys.stdout.flush()
        return error


async def lmi_run():
    task = asyncio.create_task(update_feedback_gain2(state_update_flag, triggered_roll, triggered_pitch, triggered_yaw))
    await asyncio.sleep(2)
    asyncio.gather(task)


def update_feedback_gain2(update_flag, roll, pitch, yaw, _f_kp, _f_kd, lock):
    try:
        print("Starting a subprocess for LMI computation...")
        path = "./locomotion/ha_teacher"
        # mat_engine = matlab.engine.start_matlab()
        # mat_engine.cd(path)
        # print("Matlab current working directory is ---->>>", mat_engine.pwd())
        while True:
            # _state_trig = await _queue.get()  # get updated trigger state
            print("LMI process computing...")
            # print(f"update_flag value: {update_flag.value}")
            # print(f"roll value: {roll.value}")
            # print(f"pitch value: {pitch.value}")
            # print(f"yaw value: {yaw.value}")
            # print(f"_f_kp value: {_f_kp}")
            # print(f"_f_kd value: {_f_kd}")

            if update_flag.value == 1:
                print("success!!!")
                roll_v, pitch_v, yaw_v = roll.value, pitch.value, yaw.value
                print("Obtained new state, updating the feedback gain using matlab engine")
                F_kp, F_kd = feedback_law2(roll_v, pitch_v, yaw_v)
                # lock.acquire()
                for i in range(36):
                    _f_kp[i] = np.asarray(F_kp).reshape(36)[i]
                    _f_kd[i] = np.asarray(F_kd).reshape(36)[i]
                # print(f"np.asarray(F_kp): {np.asarray(F_kp)}")
                # lock.release()
                print("Feedback gain is updated now ---->>>")

                # _lock.release()
            time.sleep(0.04)
    except:
        # traceback.print_exc(file=open("suberror.txt", "w+"))
        error = traceback.format_exc()
        print(f"subprocess error: {error}")
        sys.stdout.flush()
        return error


def feedback_law2(roll, pitch, yaw):
    """
    Computes the feedback gains for a 3-axis attitude control system.

    Args:
      roll: Roll angle (rad).
      pitch: Pitch angle (rad).
      yaw: Yaw angle (rad).

    Returns:
      F_kp: Proportional feedback gain matrix.
      F_kd: Derivative feedback gain matrix.
    """

    # Rotation matrices
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    Rzyx = Rz.dot(Ry.dot(Rx))
    # print(f"Rzyx: {Rzyx}")

    # Sampling period
    T = 1 / 35

    # Control output matrix
    C = np.array([1, 0, 0])

    # System matrices (continuous-time)
    aA = np.zeros((10, 10))
    aA[0, 4] = 1
    # A[:3, 3:6] = C
    # A[3:, 3:] = np.eye(6)
    aA[1:4, 7:10] = Rzyx
    aB = np.zeros((10, 6))
    aB[4:, :] = np.eye(6)

    # System matrices (discrete-time)
    B = np.vstack((np.zeros((4, 6)), np.eye(6))) * T
    A = np.eye(10) + T * aA

    bP = np.array([[122.164786064669, 0, 0, 0, 2.48716597374493, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 480.62107526958, 0, 0, 0, 0, 0, 155.295455907449],
                   [2.48716597374493, 0, 0, 0, 3.21760325418695, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 155.295455907449, 0, 0, 0, 0, 0, 156.306807893237]])

    eta = 2
    # beta = 0.25
    beta = 0.24
    # beta = 0.3        # also works in simulation
    kappa = 0.001

    Q = cp.Variable((10, 10), PSD=True)
    R = cp.Variable((6, 10))

    # Define constraints
    # constraints = [
    #     (beta - 2 * eta * kappa) * np.eye(10) << Q,
    #     A @ Q + B @ R >> 0,
    #     Q >> 0.95 * np.eye(10),
    #     cp.trace(bP @ Q) >= 1
    # ]

    constraints = [
        cp.bmat([
            [(beta - 2 * kappa * eta) * Q, A.T + R.T @ B.T],
            [A @ Q + B @ R, Q / 2]
        ]) >> 0,
        # (beta - 2 * eta * kappa) * np.eye(10) << Q,
        # A @ Q + B @ R >> 0,
        Q >> 0.95 * np.eye(10),
        cp.trace(bP @ Q) >= 1
    ]

    # Define problem and objective
    problem = cp.Problem(cp.Minimize(0), constraints)

    # Solve the problem
    problem.solve()

    # Check if the problem is solved successfully
    if problem.status == 'optimal':
        print("Optimization successful.")
    else:
        print("Optimization failed.")

    # Extract optimal values
    optimal_Q = Q.value
    optimal_R = R.value

    print(f"Q: {optimal_Q}")
    print(f"R: {optimal_R}")
    # Compute P
    P = np.linalg.inv(optimal_Q)

    # Compute aF
    aF = np.round(aB @ optimal_R @ P, 0)
    print(f"aF: {aF}")
    # Extract submatrices Fb1 and Fb2
    Fb1 = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, aF[4, 0]]
    ])
    Fb2 = aF[7:10, 1:4]
    print(f"Fb1: {Fb1}")
    print(f"Fb2: {Fb2}")

    # Compute F_kp
    F_kp = -np.block([
        [Fb1, np.zeros((3, 3))],
        [np.zeros((3, 3)), Fb2]
    ])

    # Compute F_kd
    F_kd = -aF[4:10, 4:10]

    return F_kp, F_kd


class HACActionMode(enum.Enum):
    """The state of a leg during locomotion."""
    MPC = 0
    PHYDRL = 1
    SIMPLEX = 2


class HATeacher:
    def __init__(self,
                 robot: Any,
                 mat_engine: Any):

        self._robot = robot

        # HAC Configure
        self.chi = 0.25
        self.epsilon = 0.4
        self.dwell_step_max = 100
        self.teacher_enable = False
        self.continual_learn = False
        self.teacher_learn = False
        self.p_mat = np.array([[122.164786064669, 0, 0, 0, 2.48716597374493, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 480.62107526958, 0, 0, 0, 0, 0, 155.295455907449],
                               [2.48716597374493, 0, 0, 0, 3.21760325418695, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 155.295455907449, 0, 0, 0, 0, 0, 156.306807893237]])

        # HAC Runtime
        self._dwell_time = 0
        self._last_action_mode = None
        self.action_mode_list = []

        # Multiprocessing
        # self.manager = mp.Manager()

        self._triggered_state = None
        # self.mat_process = mp.Process(target=self.update_feedback_gain,
        #                               args=(state_update_flag, queue, lock,))
        # self.mat_process.daemon = True  # Set to be daemon process

        self._F_kp = np.diag((0., 0., 100., 100., 100., 0.))
        self._F_kd = np.diag((40., 30., 10., 10., 10., 30.))

        # self._F_kp = np.array([[0, 0, 0, 0, 0, 0],
        #                        [0, 0, 0, 0, 0, 0],
        #                        [0, 0, 128, 0, 0, 0],
        #                        [0, 0, 0, 83, -25, -2],
        #                        [0, 0, 0, -33, 80, 2],
        #                        [0, 0, 0, 1, 0, 80]])
        #
        # self._F_kd = np.array([[39, 0, 0, 0, 0, 0],
        #                        [0, 35, 0, 0, 0, 0],
        #                        [0, 0, 35, 0, 0, 0],
        #                        [0, 0, 0, 37, -1, -9],
        #                        [0, 0, 0, -1, 37, 9],
        #                        [0, 0, 0, 0, 0, 40]])

        # f_kp = copy.deepcopy(self._F_kp)
        # f_kd = copy.deepcopy(self._F_kd)

        self.mat_engine = mat_engine

        # self.loop = asyncio.get_event_loop()
        # asyncio.run_coroutine_threadsafe(self.update_feedback_gain2(queue), self.loop)
        # self.loop.run_until_complete(asyncio.wait(self.task))
        # self.mat_process.start()
        # self.mat_process.join()

        # Matlab Engine
        # self.cvx_setup()
        # self.matlab_engine_launch()

    def matlab_engine_launch(self, path="./locomotion/ha_teacher"):
        self.mat_engine = matlab.engine.start_matlab()
        self.mat_engine.cd(path)
        print("Matlab current working directory is ---->>>", self.mat_engine.pwd())

    def feedback_law(self, roll, pitch, yaw):
        # roll = matlab.double(roll)s
        # pitch = matlab.double(pitch)
        # yaw = matlab.double(yaw)
        # self._F_kp, self._F_kd = np.array(self.mat_engine.feedback_law2(roll, pitch, yaw, nargout=2))
        # return self._F_kp, self._F_kd
        return np.asarray(f_kp).reshape(6, 6), np.asarray(f_kd).reshape(6, 6)

    def safety_value(self, states, p_mat=None):
        if p_mat is None:
            p_mat = self.p_mat

        states_shrink = states[2:]
        safety_val = np.squeeze(states_shrink.transpose() @ p_mat @ states_shrink)

        print(f"states: {states}")
        print(f"safety val: {safety_val}")
        return safety_val

    def get_hac_action(self, states):
        # print(f"state_trig value: {state_trig}")
        safety_val = self.safety_value(states)

        state_val = state_update_flag

        # Inside Safety Envelope (bounded by epsilon)
        if safety_val <= self.epsilon:

            # HAC Dwell-Time
            if self._last_action_mode == HACActionMode.SIMPLEX:

                if self._dwell_time < self.dwell_step_max:
                    print(f"current dwell time: {self._dwell_time}")
                    self._dwell_time += 1

                # Switch back to HPC (If Simplex enabled)
                elif self.continual_learn:
                    self._dwell_time = 0
                    self._triggered_state = None
                    # self._lock.acquire()
                    # state_triggered.value = None
                    state_update_flag.value = 0
                    # self._lock.release()

                    s_action = time.time()
                    if self._robot.controller.ddpg_agent is not None:
                        self._last_action_mode = HACActionMode.PHYDRL
                    else:
                        self._last_action_mode = HACActionMode.MPC
                    e_action = time.time()
                    print(f"get action duration: {e_action - s_action}")
                    print(f"Simplex control switch back to {self._last_action_mode} control")

            else:
                self._dwell_time = 0
                s_action = time.time()
                if self._robot.controller.ddpg_agent is not None:
                    self._last_action_mode = HACActionMode.PHYDRL
                else:
                    self._last_action_mode = HACActionMode.MPC
                e_action = time.time()
                print(f"get action duration: {e_action - s_action}")

        # Outside Safety Envelope (bounded by epsilon)
        else:
            if self._last_action_mode != HACActionMode.SIMPLEX:
                print(f"Safety value {safety_val} is "
                      f"outside epsilon range: {self.epsilon}, switch to simplex")
                # self._lock.acquire()
                # self._state_trig = mp.Array('d', list(states.reshape(12, 1)))
                self._triggered_state = states.reshape(12, 1)
                triggered_roll.value = states[3].squeeze()
                triggered_pitch.value = states[4].squeeze()
                triggered_yaw.value = states[5].squeeze()
                # s = time.time()
                # queue.put(self._state_triggered)
                # e = time.time()
                # print(f"queue time: {e - s}")
                state_update_flag.value = 1
                # self._lock.release()
                self._last_action_mode = HACActionMode.SIMPLEX

        print(f"In main thread: state update flag:{state_update_flag.value}")

        # Append for record
        self.action_mode_list.append(self._last_action_mode)

        # Get action
        if self._last_action_mode == HACActionMode.PHYDRL:
            action, qp_sol = self._robot.controller.get_action(phydrl=True)
            return action, qp_sol

        elif self._last_action_mode == HACActionMode.MPC:
            action, qp_sol = self._robot.controller.get_action(phydrl=False)
            return action, qp_sol

        elif self._last_action_mode == HACActionMode.SIMPLEX:
            # Swing action
            s = time.time()
            swing_action = self._robot.controller.swing_leg_controller.get_action()
            e_swing = time.time()

            # Stance action
            rpy = np.asarray(states[3:6])
            s = time.time()
            F_kp, F_kd = self.feedback_law(rpy[0], rpy[1], rpy[2])
            e = time.time()
            print(f"F_kp is: {F_kp}")
            print(f"F_kd is: {F_kd}")
            # print(f"LMI time duration: {e - s}")
            stance_action, qp_sol = self._robot.controller.stance_leg_controller.get_hac_action(
                chi=self.chi,
                state_trig=np.asarray(self._triggered_state),
                F_kp=F_kp,
                F_kd=F_kd
            )

            e_stance = time.time()
            print(f"swing_action time: {e_swing - s}")
            print(f"stance_action time: {e_stance - e_swing}")
            print(f"total get_action time: {e_stance - s}")

            actions = []
            for joint_id in range(self._robot.num_motors):
                if joint_id in swing_action:
                    actions.append(swing_action[joint_id])
                else:
                    assert joint_id in stance_action
                    actions.append(stance_action[joint_id])

            vectorized_action = MotorCommand(
                desired_position=[action.desired_position for action in actions],
                kp=[action.kp for action in actions],
                desired_velocity=[action.desired_velocity for action in actions],
                kd=[action.kd for action in actions],
                desired_torque=[
                    action.desired_torque for action in actions
                ])

            return vectorized_action, dict(qp_sol=qp_sol)

        else:
            raise RuntimeError(f"Unsupported HAC action {self._last_action_mode}")

    @property
    def feedback_gain(self):
        return self._F_kp, self._F_kd

    @property
    def last_action_mode(self):
        return self._last_action_mode

    @property
    def dwell_time(self):
        return self._dwell_time

    @last_action_mode.setter
    def last_action_mode(self, action_mode: HACActionMode):
        self._last_action_mode = action_mode


if __name__ == '__main__':
    As = np.array([[0, 1, 0, 0],
                   [0, 0, -1.42281786576776, 0.182898194776782],
                   [0, 0, 0, 1],
                   [0, 0, 25.1798795199119, 0.385056459685276]])

    Bs = np.array([[0,
                    0.970107410065162,
                    0,
                    -2.04237185222105]])

    Ak = np.array([[1, 0.0100000000000000, 0, 0],
                   [0, 1, -0.0142281786576776, 0.00182898194776782],
                   [0, 0, 1, 0.0100000000000000],
                   [0, 0, 0.251798795199119, 0.996149435403147]])

    Bk = np.array([[0,
                    0.00970107410065163,
                    0,
                    -0.0204237185222105]])

    sd = np.array([[0.234343490000000,
                    0,
                    -0.226448960000000,
                    0]])

    # ha_teacher = HATeacher()
    # K = ha_teacher.feedback_law(0, 0, 0)
    # print(K)
    F_kp, F_kd = feedback_law2(0.05, 0.05, 0.05)
    print(f"F_kp is: {F_kp}")
    print(f"F_kd is: {F_kd}")
