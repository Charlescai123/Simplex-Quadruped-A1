#!/bin/bash

ID='SeCLM'
RUNTIME=10
DESIRED_VX=0.3
DEPLOAY_PHYDRL=true
CHECKPOINT="model/trained_model"
#STANCE_CONTROL="stance_control_qpOASES"
STANCE_CONTROL="stance_control_quadprog"
SIM_ENVS="sim_v2"
ROBOT="a1_real"
FRICTION=0.7
FIXED_TIME_STEP=0.002
ACTION_REPEAT=1
RESET_TIME=3


python -m examples.a1_locomotion_example  \
  general.id=${ID} \
  general.runtime=${RUNTIME} \
  general.checkpoint=${CHECKPOINT} \
  general.deploy_phydrl=${DEPLOAY_PHYDRL} \
  envs/simulator=${SIM_ENVS} \
  envs.simulator.fixed_time_step=${FIXED_TIME_STEP} \
  envs.simulator.friction=${FRICTION} \
  envs/robot=${ROBOT} \
  envs.robot.command.desired_vx=${DESIRED_VX} \
  envs/robot/mpc_controller/stance_leg=${STANCE_CONTROL} \
  envs.robot.interface.reset_time=${RESET_TIME} \
  envs.robot.interface.action_repeat=${ACTION_REPEAT}

#  robot/interface="a1" \
#  robot/state_estimator="a1_velocity_estimator" --help --resolve
#  robot.interface.a1_model.motor_init_position:=[${MOTOR_INIT_POSITION}]

