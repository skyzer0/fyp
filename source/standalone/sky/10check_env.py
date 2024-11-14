##./isaaclab.sh -p source/standalone/sky/pick_with_camera.py --headless --save

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with a pick and lift state machine.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.


"""

"""Launch Omniverse Toolkit first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for lift environments.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
parser.add_argument(
    "--draw",
    action="store_true",
    default=False,
    help="Draw the pointcloud from camera at index specified by ``--camera_id``.",
)

parser.add_argument(
    "--camera_id",
    type=int,
    choices={0, 1},
    default=0,
    help=(
        "The camera ID to use for displaying points or saving the camera data. Default is 0."
        " The viewport will always initialize with the perspective of camera 0."
    ),
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import omni.isaac.lab.sim as sim_utils
import torch
import omni.replicator.core as rep
from collections.abc import Sequence
import warp as wp
from omni.isaac.lab.assets.rigid_object.rigid_object_data import RigidObjectData
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg
from omni.isaac.lab.sensors import CameraData
import os
from omni.isaac.lab.utils.math import transform_points, unproject_depth
from omni.isaac.lab.utils import convert_dict_to_backend
from omni.isaac.lab.markers.config import RAY_CASTER_MARKER_CFG
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_depth



def main():
    # sim: sim_utils.SimulationContext
    # parse configuration
    env_cfg: LiftEnvCfg = parse_env_cfg(
        # "Isaac-Lift-Cube-ur5e-v3-IK",
        "Isaac-Reach-UR5E-v0",
        use_gpu=not args_cli.cpu,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # create environment
    # env = gym.make("Isaac-Lift-Cube-ur5e-v3-IK", cfg=env_cfg)
    env = gym.make("Isaac-Reach-UR5E-v0", cfg=env_cfg)
    # reset environment at start
    env.reset()
    # create action buffers (position + quaternion)
    actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
    actions[:, 3] = 1.0
    while simulation_app.is_running():
        with torch.inference_mode():
            # step environment
            env.step(actions)[-2]
    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
