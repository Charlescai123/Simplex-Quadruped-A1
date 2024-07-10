#!/bin/bash

ID="phydrl-test-1"
RUN_NAME="real_plant"
RUNTIME=10
DESIRED_VX=0.3
DEPLOAY_PHYDRL=true
MODEL_NAME="phydrl-test-2_1"
#STANCE_CONTROL="stance_control_qpOASES"
STANCE_CONTROL="stance_control_quadprog"
SIM_ENVS="sim_v2"
FRICTION=0.7
FIXED_TIME_STEP=0.002
ACTION_REPEAT=1
RESET_TIME=3


python3 -m examples.a1_locomotion_example  \
  general.id=${ID} \
  general.name=${RUN_NAME} \
  general.runtime=${RUNTIME} \
  general.deploy_phydrl=${DEPLOAY_PHYDRL} \
  general.id=${MODEL_NAME} \
  envs/simulator=${SIM_ENVS} \
  envs.simulator.fixed_time_step=${FIXED_TIME_STEP} \
  envs.simulator.friction=${FRICTION} \
  envs/robot="a1_real" \
  envs.robot.command.desired_vx=${DESIRED_VX} \
  envs/robot/mpc_controller/stance_leg=${STANCE_CONTROL} \
  envs.robot.interface.reset_time=${RESET_TIME} \
  envs.robot.interface.action_repeat=${ACTION_REPEAT}

#  robot/state_estimator="a1_velocity_estimator" --help --resolve
#  robot.interface.a1_model.motor_init_position:=[${MOTOR_INIT_POSITION}]

