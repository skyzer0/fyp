# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.lab_assets.UR5E_gripper import UR5E_CFG_v2_HIGH_PD_CFG  # isort: skip
from omni.isaac.lab_assets import UR5E_CFG_v2
from omni.isaac.lab.sensors import CameraCfg, RayCasterCameraCfg, TiledCameraCfg
import omni.isaac.lab.sim as sim_utils
import numpy as np


@configclass
class UR5eCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # Post-initialize parent class
        super().__post_init__()

        # Set UR5e as robot
        self.scene.robot = UR5E_CFG_v2.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (UR5e with Robotiq gripper)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint',
            'robotiq_85_left_knuckle_joint',
            'robotiq_85_right_knuckle_joint'
        ], scale=0.5, use_default_offset=True
        )

        # Gripper action configuration for Robotiq 85
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["robotiq_85_left_knuckle_joint", "robotiq_85_right_knuckle_joint"],
            open_command_expr={
                "robotiq_85_left_knuckle_joint": 0.00,
                "robotiq_85_right_knuckle_joint": 0.00,
            },
            close_command_expr={
                "robotiq_85_left_knuckle_joint": 1.0,
                "robotiq_85_right_knuckle_joint": 1.0,
            },
        )

        # Set the body name for the end effector
        self.commands.object_pose.body_name = "tool0"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Frame Transformer configuration for the end effector
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/tool0",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )


@configclass
class UR5eCubeLiftEnvCfg_PLAY(UR5eCubeLiftEnvCfg):
    def __post_init__(self):
        # Post-initialize parent class
        super().__post_init__()

        # Configure a smaller environment for play mode
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Disable randomization for play mode
        self.observations.policy.enable_corruption = False
