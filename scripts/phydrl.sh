#!/bin/bash

ID="phydrl-test-2_2"
MODE="train"
DESIRED_VX=0.3
DEPLOAY_PHYDRL=false
TEACHER_ENABLE=false
SIM_ENVS="sim_v2"
#STANCE_CONTROL="stance_control_qpOASES"
STANCE_CONTROL="stance_control_quadprog"
RANDOM_INIT=true
FIXED_TIME_STEP=0.002
ROBOT="a1_sim"
FRICTION=0.7
ACTION_REPEAT=1
RESET_TIME=3

# For training
python3 -m examples.main_drl_example \
  general.id=${ID} \
  general.mode=${MODE} \
  general.deploy_phydrl=${DEPLOAY_PHYDRL} \
  ha_teacher.teacher_enable=${TEACHER_ENABLE} \
  envs/simulator=${SIM_ENVS} \
  envs.simulator.fixed_time_step=${FIXED_TIME_STEP} \
  envs.simulator.friction=${FRICTION} \
  envs/robot=${ROBOT} \
  envs.robot.command.desired_vx=${DESIRED_VX} \
  envs/robot/mpc_controller/stance_leg=${STANCE_CONTROL} \
  envs.robot.interface.reset_time=${RESET_TIME} \
  envs.robot.interface.action_repeat=${ACTION_REPEAT}


# For testing (inference)
#python3 -m examples.main_drl_example  \
#  robot=${ROBOT} \
#  phydrl.id=${MODEL_NAME} \
#  phydrl.mode="test" \
#  simulator.friction=${FRICTION} \
#  simulator.fixed_time_step=${FIXED_TIME_STEP} \
#  robot.command.desired_vx=${DESIRED_VX} \
#  robot/mpc_controller/stance_leg@mpc_controller.stance_leg=${STANCE_CONTROL}
