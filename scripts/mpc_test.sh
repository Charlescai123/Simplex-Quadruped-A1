#!/bin/bash

MODEL_NAME="phydrl-test-1"
SYNC_GUI=true
DESIRED_VX=0.3
FRICTION=0.4
ROBOT="robot_sim"
SIM_ENVS="sim_v2"
MOTOR_INIT_POSITION='${robot.constant.pose.laying}'
RESET_TIME=3
FIXED_TIME_STEP=0.002
#TIMESTEP=0.001
ACTION_REPEAT=1

python3 -m examples.main_mpc_example  \
  robot=${ROBOT} \
  phydrl.id=${MODEL_NAME} \
  phydrl.mode="test" \
  envs=${SIM_ENVS} \
  envs.fixed_time_step=${FIXED_TIME_STEP} \
  envs.friction=${FRICTION} \
  robot.command.desired_vx=${DESIRED_VX} \
  robot.robot_model.reset_time=${RESET_TIME} \
  robot.robot_model.sync_gui=${SYNC_GUI} \
  robot.robot_model.motor_init_position=${MOTOR_INIT_POSITION} \
  robot.robot_model.action_repeat=${ACTION_REPEAT}

#  robot/interface="a1" \
#  robot/state_estimator="a1_velocity_estimator" --help --resolve
#  robot.interface.a1_model.motor_init_position:=[${MOTOR_INIT_POSITION}]

