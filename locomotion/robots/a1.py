"""Class for Unitree A1 robot."""
import ml_collections
import numpy as np
from typing import Any
from typing import Sequence
from typing import Tuple, Optional
import json
import time
from locomotion.robots import kinematics
from locomotion.robots.motors import MotorControlMode
from locomotion.robots.motors import MotorGroup
from locomotion.robots.motors import MotorModel
from locomotion.robots.motors import MotorCommand
from locomotion.robots.quadruped import QuadrupedRobot
from config.locomotion.robots.a1_params import A1Params
from config.locomotion.robots.motor_params import MotorGroupParams
from config.locomotion.controllers.swing_params import SwingControllerParams
from config.locomotion.controllers.stance_params import StanceControllerParams
from config.locomotion.robots.pose import Pose

from locomotion.mpc_controller import swing_leg_controller
from locomotion.gait_scheduler import offset_gait_scheduler
from locomotion.state_estimator import com_velocity_estimator
from config.phydrl.ddpg_params import DDPGParams
from config.phydrl.taylor_params import TaylorParams
from agents.phydrl.policies.ddpg import DDPGAgent
from locomotion import wbc_controller

class A1(QuadrupedRobot):
    """A1 Simulation Robot."""

    def __init__(
            self,
            pybullet_client: Any = None,
            ddpg_agent: DDPGAgent = None,
            mat_engine: Any = None,
            a1_params: A1Params = None,
            motor_params: MotorGroupParams = None,
            swing_params: SwingControllerParams = None,
            stance_params: StanceControllerParams = None,
            logdir='./logs'
    ) -> None:
        """Constructs an A1 robot and resets it to the initial states.
        Initializes a tuple with a single MotorGroup containing 12 MotorModels.
        Each MotorModel is by default configured for the robot of the A1.
        """

        # Robot robot
        self._a1_params = a1_params
        self._swing_params = swing_params
        self._stance_params = stance_params
        self._motor_params = motor_params

        # Pybullet client
        self._pybullet_client = pybullet_client

        # Robot Dynamics robot
        self._mpc_body_mass = a1_params.mpc_body_mass
        self._mpc_body_height = a1_params.mpc_body_height
        self._mpc_body_inertia = a1_params.mpc_body_inertia

        # Motor
        self._motor_init_position = a1_params.motor_init_position
        self._motor_control_mode = a1_params.motor_control_mode
        motors = self._load_a1_motors(self._motor_params)

        super().__init__(
            motors=motors,
            base_joint_names=a1_params.base_joint_names,
            foot_joint_names=a1_params.foot_joint_names,
        )
        self._last_timestamp = 0
        self._control_interval = (a1_params.time_step * a1_params.action_repeat)
        self._control_freq = 1 / self._control_interval

        # Robot urdf phydrl
        self._chassis_link_ids = [-1]
        self._motor_joint_ids = []  # Actuators for robot in Simulation
        self._foot_link_ids = []  # For calculating foot location in Simulation
        self.quadruped = self._load_robot_urdf(a1_params.urdf_path)

        self._foot_contact_history = self.foot_positions_in_body_frame.copy()
        self._foot_contact_history[:, 2] = -self.mpc_body_height

        # Robot desired speed/twisting speed
        self._desired_vx = a1_params.desired_vx
        self._desired_vy = a1_params.desired_vy
        self._desired_wz = a1_params.desired_wz

        # Robot Controller
        self._controller = wbc_controller.WholeBodyController(
            robot=self,
            ddpg_agent=ddpg_agent,
            mat_engine=mat_engine,
            desired_speed=(self._desired_vx, self._desired_vy),
            desired_twisting_speed=self._desired_wz,
            desired_com_height=self._mpc_body_height,
            mpc_body_mass=self._mpc_body_mass,
            mpc_body_inertia=self._mpc_body_inertia,
            swing_params=self._swing_params,
            stance_params=self._stance_params,
            logdir=logdir
        )

        # Reset robot to init pose
        self.reset()

    def reset(self,
              hard_reset: bool = False,
              num_reset_steps: Optional[int] = None) -> None:
        """Resets the robot."""

        # self._reset_pybullet_client()

        self._reset_robot_pose(hard_reset=hard_reset, num_reset_steps=num_reset_steps)

        self._last_timestamp = self.time_since_reset

    def _reset_pybullet_client(self):

        # Reset the pybullet_client
        num_solver_iterations_ = self._a1_params.num_solver_iterations
        enable_cone_friction_ = self._a1_params.enable_cone_friction

        # self._pybullet_client.resetSimulation()
        self._pybullet_client.setPhysicsEngineParameter(numSolverIterations=num_solver_iterations_)
        self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=enable_cone_friction_)
        self._pybullet_client.setTimeStep(self._a1_params.time_step)
        self._pybullet_client.setGravity(0, 0, -9.8)

    def _reset_robot_pose(self, hard_reset: bool = False,
                          num_reset_steps: Optional[int] = None):

        # Reset the robot
        if hard_reset:
            # This assumes that resetSimulation() is already called.
            self.quadruped = self._load_robot_urdf(self._a1_params.urdf_path)

        else:
            init_position = (self._a1_params.init_rack_position
                             if self._a1_params.on_rack else
                             self._a1_params.init_position)
            self._pybullet_client.resetBasePositionAndOrientation(
                self.quadruped, init_position, [0.0, 0.0, 0.0, 1.0])

        num_joints = self._pybullet_client.getNumJoints(self.quadruped)
        for joint_id in range(num_joints):
            self._pybullet_client.setJointMotorControl2(
                bodyIndex=self.quadruped,
                jointIndex=joint_id,
                controlMode=self._pybullet_client.VELOCITY_CONTROL,
                targetVelocity=0,
                force=0,
            )

        # Set motors to the initial position
        # TODO: these should be set already when they are instantiated?
        for i in range(len(self._motor_joint_ids)):
            self._pybullet_client.resetJointState(
                self.quadruped,
                self._motor_joint_ids[i],
                self._motor_group.init_positions[i],
                targetVelocity=0,
            )

        # Steps the robot with position command
        if num_reset_steps is None:
            num_reset_steps = int(self._a1_params.reset_time / self._control_interval)

        # Interpolate to the standing position
        for i in range(num_reset_steps):
            rate = i / num_reset_steps
            p = self.joint_linear_interpolation(self._motor_group.init_positions,
                                                self._a1_params.motor_init_target_position, rate)
            motor_cmd = MotorCommand(desired_position=p)

            self.step(motor_cmd, MotorControlMode.POSITION)
            print(f"step counter: {self._step_counter}")
            print(f"action counter: {self._action_counter}")

    def _load_robot_urdf(self, urdf_path: str) -> int:

        if not self._pybullet_client:
            raise AttributeError("No pybullet client specified!")
        if self._a1_params.on_rack:
            quadruped = self._pybullet_client.loadURDF(urdf_path,
                                                       self._a1_params.init_rack_position,
                                                       useFixedBase=True)
        else:
            quadruped = self._pybullet_client.loadURDF(urdf_path, self._a1_params.init_position)
        self._build_urdf_ids(quadruped)
        return quadruped

    def _build_urdf_ids(self, quadruped) -> None:
        """Records ids of base link, foot links and motor joints.

        For detailed documentation of links and joints, please refer to the
        pybullet documentation.
        """
        num_joints = self._pybullet_client.getNumJoints(quadruped)
        for joint_id in range(num_joints):
            joint_info = self._pybullet_client.getJointInfo(quadruped, joint_id)
            joint_name = joint_info[1].decode("UTF-8")
            if joint_name in self._base_joint_names:
                self._chassis_link_ids.append(joint_id)
            elif joint_name in self._motor_group.motor_joint_names:
                self._motor_joint_ids.append(joint_id)
            elif joint_name in self._foot_joint_names:
                self._foot_link_ids.append(joint_id)

    def _apply_action(self, action, motor_control_mode=None) -> None:
        torques, observed_torques = self._motor_group.convert_to_torque(
            action, self.motor_angles, self.motor_velocities, motor_control_mode)
        # import pdb
        # pdb.set_trace()
        self._pybullet_client.setJointMotorControlArray(
            bodyIndex=self.quadruped,
            jointIndices=self._motor_joint_ids,
            controlMode=self._pybullet_client.TORQUE_CONTROL,
            forces=torques,
        )
        self._motor_torques = observed_torques

    def step(self, action, motor_control_mode=None) -> None:
        # print("Entering a1.step!!!")
        self._step_counter += 1
        for _ in range(self._a1_params.action_repeat):
            self._apply_action(action, motor_control_mode)
            self._pybullet_client.stepSimulation()  # Step in simulation
            self._action_counter += 1
            self._update_contact_history()

        # Sync time in simulation and real world
        if self._a1_params.sync_gui:
            # time.sleep(self.control_timestep)
            time.sleep(0.002)

        # Camera setup:
        if (self._a1_params.camera_fixed):
            self._pybullet_client.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=30 + self.base_orientation_rpy[2] / np.pi * 180,
                cameraPitch=-30,
                cameraTargetPosition=self.base_position,
            )

    def _update_contact_history(self):

        dt = self.time_since_reset - self._last_timestamp
        self._last_timestamp = self.time_since_reset
        # print(f"time_since_reset: {self.time_since_reset}")
        # print(f"last_timestamp: {self._last_timestamp}")
        # print(f"dt: {dt}")

        base_orientation = self.base_orientation_quaternion
        rot_mat = self.pybullet_client.getMatrixFromQuaternion(base_orientation)
        rot_mat = np.array(rot_mat).reshape((3, 3))
        base_vel_body_frame = rot_mat.T.dot(self.base_linear_velocity)

        foot_contacts = self.foot_contacts.copy()
        foot_positions = self.foot_positions_in_body_frame.copy()
        for leg_id in range(4):
            if foot_contacts[leg_id]:
                self._foot_contact_history[leg_id] = foot_positions[leg_id]
            else:
                self._foot_contact_history[leg_id] -= base_vel_body_frame * dt
        # print(f"foot_contact_history: {self._foot_contact_history}")

    @property
    def base_position(self):
        return np.array(
            self._pybullet_client.getBasePositionAndOrientation(self.quadruped)[0])

    # @property
    # def base_velocity(self):
    #     return self._pybullet_client.getBaseVelocity(self.robot)[0]

    @property
    def base_linear_velocity(self):
        return self._pybullet_client.getBaseVelocity(self.quadruped)[0]

    @property
    def base_angular_velocity(self):
        return self._pybullet_client.getBaseVelocity(self.quadruped)[1]

    @property
    def base_orientation_rpy(self):
        return self._pybullet_client.getEulerFromQuaternion(
            self.base_orientation_quaternion)

    @property
    def base_orientation_quaternion(self):
        return np.array(
            self._pybullet_client.getBasePositionAndOrientation(self.quadruped)[1])

    @property
    def controller(self):
        return self._controller

    @property
    def desired_speed(self):
        return (self._desired_vx, self._desired_vy)

    @desired_speed.setter
    def desired_speed(self, speed: Tuple[float, float]):
        self._desired_vx, self._desired_vy = speed[0], speed[1]

    @property
    def desired_twisting_speed(self):
        return self._desired_wz

    @desired_twisting_speed.setter
    def desired_twisting_speed(self, twisting_speed: int):
        self._desired_wz = twisting_speed

    @property
    def motor_angles(self):
        joint_states = self._pybullet_client.getJointStates(
            self.quadruped, self._motor_joint_ids)
        return np.array([s[0] for s in joint_states])

    @property
    def motor_velocities(self):
        joint_states = self._pybullet_client.getJointStates(
            self.quadruped, self._motor_joint_ids)
        return np.array([s[1] for s in joint_states])

    @property
    def motor_torques(self):
        return self._motor_torques

    @property
    def foot_contact_history(self):
        return self._foot_contact_history

    @property
    def base_angular_velocity_in_body_frame(self):
        angular_velocity = self.base_angular_velocity
        orientation = self.base_orientation_quaternion
        _, orientation_inversed = self._pybullet_client.invertTransform(
            [0, 0, 0], orientation)
        relative_velocity, _ = self._pybullet_client.multiplyTransforms(
            [0, 0, 0],
            orientation_inversed,
            angular_velocity,
            self._pybullet_client.getQuaternionFromEuler([0, 0, 0]),
        )
        return np.asarray(relative_velocity)

    @property
    def foot_positions_in_body_frame(self):
        foot_positions = []
        for foot_id in self._foot_link_ids:
            foot_position = kinematics.link_position_in_body_frame(
                robot=self,
                link_id=foot_id,
            )
            foot_positions.append(foot_position)
        return np.array(foot_positions)

    @property
    def foot_contacts(self):
        all_contacts = self._pybullet_client.getContactPoints(bodyA=self.quadruped)

        contacts = [False, False, False, False]
        for contact in all_contacts:
            # Ignore self contacts
            if contact[2] == self.quadruped:
                continue
            try:
                toe_link_index = self._foot_link_ids.index(contact[3])
                contacts[toe_link_index] = True
            except ValueError:
                continue
        return contacts

    def compute_foot_jacobian(self, leg_id):
        """Compute the Jacobian for a given leg."""
        full_jacobian = kinematics.compute_jacobian(
            robot=self,
            link_id=self._foot_link_ids[leg_id],
        )
        motors_per_leg = self.num_motors // self.num_legs
        com_dof = 6
        return full_jacobian[:, com_dof + leg_id * motors_per_leg:com_dof +
                                                                  (leg_id + 1) * motors_per_leg]

    def get_motor_angles_from_foot_position(self, leg_id, foot_local_position):
        toe_id = self._foot_link_ids[leg_id]

        motors_per_leg = self.num_motors // self.num_legs
        joint_position_indexes = list(
            range(leg_id * motors_per_leg, leg_id * motors_per_leg + motors_per_leg))

        joint_angles = kinematics.joint_angles_from_link_position(
            robot=self,
            link_position=foot_local_position,
            link_id=toe_id,
            joint_ids=joint_position_indexes,
        )
        # Return the joint index (the same as when calling GetMotorAngles) as well
        # as the angles.
        return joint_position_indexes, joint_angles

    def map_contact_force_to_joint_torques(self, leg_id, contact_force):
        """Maps the foot contact force to the leg joint torques."""
        jv = self.compute_foot_jacobian(leg_id)
        motor_torques_list = np.matmul(contact_force, jv)
        motor_torques_dict = {}
        motors_per_leg = self.num_motors // self.num_legs
        for torque_id, joint_id in enumerate(
                range(leg_id * motors_per_leg, (leg_id + 1) * motors_per_leg)):
            motor_torques_dict[joint_id] = motor_torques_list[torque_id]
        return motor_torques_dict

    def _load_a1_motors(self, motor_params: MotorGroupParams):
        a1_motors = MotorGroup(motor_params.motors)
        a1_motors.init_positions = self._motor_init_position
        a1_motors.motor_control_mode = self._motor_control_mode

        return a1_motors

    @property
    def gait_generator(self):
        return self._gait_generator

    @property
    def swing_params(self):
        return self._swing_params

    @property
    def stance_params(self):
        return self._stance_params

    @property
    def robot_params(self):
        return self._a1_params

    @property
    def pybullet_client(self):
        return self._pybullet_client

    @property
    def control_timestep(self):
        return self._control_interval

    @property
    def time_since_reset(self):
        # return self._step_counter * self.control_timestep
        return self._action_counter * self.control_timestep

    @property
    def mpc_body_height(self):
        return self._mpc_body_height

    @mpc_body_height.setter
    def mpc_body_height(self, mpc_body_height: float):
        self._mpc_body_height = mpc_body_height

    @property
    def mpc_body_mass(self):
        return self._mpc_body_mass

    @mpc_body_mass.setter
    def mpc_body_mass(self, mpc_body_mass: float):
        self._mpc_body_mass = mpc_body_mass

    @property
    def mpc_body_inertia(self):
        return self._mpc_body_inertia

    @mpc_body_inertia.setter
    def mpc_body_inertia(self, mpc_body_inertia: Sequence[float]):
        self._mpc_body_inertia = mpc_body_inertia

    @property
    def swing_reference_positions(self):
        return self._a1_params.swing_reference_positions

    @property
    def num_motors(self):
        return 12
