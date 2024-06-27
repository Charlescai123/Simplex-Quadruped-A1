# Simplex-Quadruped

![Tensorflow](https://img.shields.io/badge/Tensorflow-2.5.0-orange?logo=tensorflow)
![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Linux](https://img.shields.io/badge/Linux-22.04-yellow?logo=linux)
![Pybullet](https://img.shields.io/badge/Pybullet-3.2.6-brightgreen)

---

## Table of Content

* [Code Structure](#code-structure)
* [Environment Setup](#environment-setup)
* [PhyDRL Runtime](#phydrl-runtime)
* [Running Convex MPC Controller](#running-convex-mpc-controller)
    * [In Simulation](#in-simulation)
    * [In Real A1 Robot](#in-real-a1-robot)
* [Trouble Shootings](#trouble-shootings)

---

## Code Structure

This repo adapts from the codebase of [fast_and_efficient](https://github.com/yxyang/fast_and_efficient), with a
modularized structure to validate the Simplex-enabled Safe Continual Learning Machine on quadruped robot. All the
configuration settings are under the folder *config*, and the parameters are spawn in
hierarchy for each instance:

```
├── agents
│      ├── cmaes                                      <- CMA-ES framework
│      └── phydrl                                     <- PhyDRL framework
│           ├── ...
│           ├── envs                                  <- Training environments
│           ├── learners                              <- PhyDRL learner
│           ├── networks                              <- Talyor neural network & MLP
│           ├── replay_buffers                        <- Replay memory
│           ├── policies                              <- DDPG algorithm
│           └── utils                                 <- Utilities (logger/add training noise)
├── config                                            
│      ├── phydrl
│           ├── ddpg_params.py                        <- DDPG parameters
│           ├── logger_params.py                      <- Logger parameters
│           ├── taylor_params.py                      <- Taylor neural network parameters
│           └── trainer_params.py                     <- PhyDRL trainer parameters
│      ├── locomotion
│           ├── controllers                           <- Low-level controller parameters
│           ├── gait_scheduler                        <- Gait scheduler parameters
│           └── robots                                <- Robot model and motor parameters
│      ├── a1_real_params.py                          <- Config for real A1 robot
│      ├── a1_sim_params.py                           <- Config for simulated A1 robot
│      └── a1_trainer_params.py                       <- Config for trained A1 robot
├── examples                                
│      ├── a1_exercise_example.py                     <- Robot makes a sinuous move
│      ├── a1_sim_to_real_example.py                  <- Robot sim-to-real (for testing)
│      ├── a1_mpc_controller_example.py               <- Running MPC controller in simulator/real plant
│      ├── main_drl.py                                <- Training A1 with PhyDRL
│      └── main_mpc.py                                <- Testing trained PhyDRL policy
├── locomotion
│      ├── gait_scheduler                            
│           ├── gait_scheduler.py                     <- An abstract class
│           └── offset_gait_scheduler.py              <- Actual gait generator
│      ├── ha_teacher       
│           ├── ...
│           └── ha_teacher.py                         <- HA Teacher   
│      ├── mpc_controllers                      
│           ├── mpc_osqp.cc                           <- OSQP library for stance state controller
│           ├── qp_torque_optimizer.py                <- QP solver for stance acceleration controller
│           ├── stance_leg_controller_mpc.py          <- Stance controller (objective func -> state)
│           ├── stance_leg_controller_quadprog.py     <- Stance controller (objective func -> acceleration)
│           └── swing_leg_controller.py               <- Swing controller (using Raibert formula)
│      ├── robots
│           ├── ...
│           ├── a1.py                                 <- A1 robot (for simulation)
│           ├── a1_robot.py                           <- A1 robot (for real plant)
│           ├── a1_robot_phydrl.py                    <- A1 robot (for PhyDRL training)
│           ├── motors.py                             <- A1 motor model
│           └── quadruped.py                          <- An abstract base class for all robots
│      ├── state_estimators
│           ├── a1_robot_state_estimator.py           <- State estimator for real A1
│           ├── com_velocity_estimator.py             <- CoM velocity estimator simulator/real plant 
│           └── moving_window_fillter.py              <- A filter used in CoM velocity estimator
│      ├── wbc_controller.py                          <- robot whole-body controller
│      └── wbc_controller_cl.py                       <- robot whole-body controller (For continual learning)
├── ...
├── logs                                              <- Log files for training
├── models                                            <- Trained model saved path
├── third_party                                       <- Code by third parties (unitree, qpsolver, etc.)
├── requirements.txt                                  <- Depencencies for code environment
├── setup.py
└── utils.py                         
```

[//]: # (## Running Convex MPC Controller:)

[//]: # ()

[//]: # (### Setup the environment)

[//]: # ()

[//]: # (First, make sure the environment is setup by following the steps in the [Setup]&#40;#Setup&#41; section.)

[//]: # ()

[//]: # (### Run the code:)

[//]: # ()

[//]: # (```bash)

[//]: # (python -m src.convex_mpc_controller.convex_mpc_controller_example --show_gui=True --max_time_secs=10 --world=plane)

[//]: # (```)

[//]: # ()

[//]: # (change `world` argument to be one of `[plane, slope, stair, uneven]` for different worlds. The current MPC controller)

[//]: # (has been tuned for all four worlds.)

## Environment Setup

### Setup for Local PC

It is recommended to create a separate virtualenv or conda environment to avoid conflicting with existing system
packages. The required packages have been tested under Python 3.8.5, though they should be compatible with other Python
versions.

Follow the steps below to build the Python environment:

1. First, install all dependent packages by running:

   ```bash
   pip install -r requirements.txt
   ```

2. Second, install the C++ binding for the convex MPC controller:

   ```bash
   python setup.py install
   ```

3. Lastly, build and install the interface to Unitree's SDK. The
   Unitree [repo](https://github.com/unitreerobotics/unitree_legged_sdk) keeps releasing new SDK versions. For
   convenience, we have included the version that we used in `third_party/unitree_legged_sdk`.

   First, make sure the required packages are installed, following
   Unitree's [guide](https://github.com/unitreerobotics/unitree_legged_sdk?tab=readme-ov-file#dependencies). Most
   nostably, please make sure to
   install `Boost` and `LCM`:

   ```bash
   sudo apt install libboost-all-dev liblcm-dev
   ```

   Then, go to `third_party/unitree_legged_sdk` and create a build folder:
   ```bash
   cd third_party/unitree_legged_sdk
   mkdir build && cd build
   ```

   Now, build the libraries and move them to the main directory by running:
   ```bash
   cmake ..
   make
   mv robot_interface* ../../..
   ```

### Setup for Real Robot

Follow the steps if you want to run controllers on the real robot:

1. **Setup correct permissions for non-sudo user (optional)**

   Since the Unitree SDK requires memory locking and high process priority, root priority with `sudo` is usually
   required to execute commands. To run the SDK without `sudo`, write the following
   to `/etc/security/limits.d/90-unitree.conf`:

   ```bash
   <username> soft memlock unlimited
   <username> hard memlock unlimited
   <username> soft nice eip
   <username> hard nice eip
   ```

   Log out and log back in for the above changes to take effect.

2. **Connect to the real robot**

   Configure the wireless on the real robot with the [manual](docs/A1_Wireless_Configuration.pdf), and make sure
   you can ping into the robot's low-level controller (IP:`192.168.123.10`) on your local PC.

[//]: # (3. **Test connection**)

[//]: # ()

[//]: # (   Start up the robot. After the robot stands up, enter joint-damping mode by pressing L2+B on the remote controller.)

[//]: # (   Then, run the following:)

[//]: # (   ```bash)

[//]: # (   python -m src.robots.a1_robot_exercise_example --use_real_robot=True)

[//]: # (   ```)

[//]: # ()

[//]: # (   The robot should be moving its body up and down following a pre-set trajectory. Terminate the script at any time to)

[//]: # (   bring the robot back to joint-damping position.)

## PhyDRL Runtime

PhyDRL adopts a model-based control approach with DDPG as its learning strategy. Through the utilization of the residual
control framework and a CLF(Control Lyapunov Function)-like reward design, it demonstrates great performance and
holds promise for addressing safety-critical control challenges.

### Training

To train the A1 using PhyDRL, refer to the following command:

   ```bash
   python -m examples.main_drl --gpu --mode=train --id={your phydrl name}
   ```

### Testing

To test the trained PhyDRL policy, refer to the following command:

   ```bash
   python -m examples.main_drl --gpu --mode=test --id={your phydrl name}
   ```

## Running Convex MPC Controller

We introduce two kinds of MPC controllers for the stance leg, one of which incorporates `state` as its objective
function while another uses `acceleration` for optimization. To test each variant, you need to modify the value of
*objective_function* in the config file.

### In Simulation

The config parameters are loaded from `config/a1_sim_params.py`. You can also change the running speed in
the *a1_mpc_controller_example.py* file after spawning an instance for the parameters. To
run it in simulation:

   ```bash
   python -m examples.a1_mpc_controller_example --show_gui=True --max_time_secs=10 --use_real_robot=False --world=plane
   ```

### In Real A1 Robot

The config parameters are loaded from `config/a1_real_params.py`. You can also change the running speed in
the *a1_mpc_controller_example.py* file after spawning an instance for the parameters. To
run it in real A1 robot:

1. Start the robot. After it stands up, get it enter joint-damping mode by pressing `L2+B` on the remote controller.
   Then the robot should lay down on the ground.

2. Establish a connection to the A1 hotspot (UnitreeRoboticsA1-xxxx) on your local machine and verify its communication
   with the low-level controller onboard.

3. Run the following command:

   ```bash
   python -m examples.a1_mpc_controller_example --show_gui=False --max_time_secs=15 --use_real_robot=True 
   ```

[//]: # (### Convex MPC Controller)

[//]: # ()

[//]: # (The `src/convex_mpc_controller` folder contains a Python implementation of)

[//]: # (MIT's [Convex MPC Controller]&#40;https://ieeexplore.ieee.org/document/8594448&#41;. Some notable files include:)

[//]: # ()

[//]: # (* `torque_stance_leg_controller_mpc.py` sets up and solves the MPC problem for stance legs.)

[//]: # (* `mpc_osqp.cc` actually sets up the QP and calls a QP library to solve it.)

[//]: # (* `raibert_swing_leg_controller.py` controlls swing legs.)

## Trouble-shootings

In case your onboard motor damaged due to unknown problems, refer to
the [instruction manual](docs/A1_Motor_Replacement.pdf) for its replacement