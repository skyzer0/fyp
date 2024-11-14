# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

##Camera: 简单易用，适合单一视角的应用。
##TiledCamera: 多视角高效，适合需要同时获取多个视角数据的应用，但配置和资源消耗较高。
##RayCasterCamera: 高精度，适合需要精确深度和法线数据的复杂场景，但计算开销大，配置复za

from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG

from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp
from omni.isaac.lab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
from omni.isaac.lab_assets import UR5E_CFG_v2
from omni.isaac.lab.sensors import CameraCfg, RayCasterCameraCfg, TiledCameraCfg
import omni.isaac.lab.sim as sim_utils
import numpy as np

@configclass
class v3ur5eCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = UR5E_CFG_v2.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.body_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[
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

        self.actions.finger_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["robotiq_85_left_knuckle_joint","robotiq_85_right_knuckle_joint"],
            open_command_expr={"robotiq_85_left_knuckle_joint":0.0,"robotiq_85_right_knuckle_joint":0.0},
            close_command_expr={"robotiq_85_left_knuckle_joint":0.45,"robotiq_85_right_knuckle_joint":0.45},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "tool0"

        
        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[0,0,0,0]),
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

        self.scene.camera2= CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/world/world_camera",
            update_period=0.1,
            height=480,
            width=640,
            data_types=[
                "rgb",
                # "distance_to_image_plane",
                # "normals",
                # "semantic_segmentation",
                # "instance_segmentation_fast",
                # "instance_id_segmentation_fast",
                ],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            ##default facing down the z-axis
            ##increase the z value to move the camera down
            offset=CameraCfg.OffsetCfg(
                pos=(1.0, 0.9, 1.5), 
                rot=(0.0, 0.437, 0.846, -0.306),
                # rot=(0,0,1,0),
                convention="ros"
            ),
        )
        
        
        # Listens to the required transforms
        self.scene.camera= CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/tool0/camera",
            update_period=0.1,
            height=480,
            width=640,
            data_types=[
                "rgb",
                "distance_to_image_plane",
                # "normals",
                # "semantic_segmentation",
                # "instance_segmentation_fast",
                # "instance_id_segmentation_fast",
                ],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            ##default facing down the z-axis
            ##increase the z value to move the camera down
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.02), 
                # rot=(0.0, 0.7071068, 0.0, 0.7071068),
                rot=(1,0,0,0),
                convention="ros"
            ),
        )

        for i in range(1, 29):  # Adjust range to start from 1
            rand_x=np.random.uniform(0.4, 0.9)
            rand_y=np.random.uniform(-0.2, 0.2)
            rand_z=np.random.uniform(0, 0.5)
            marker_name = f"object{i}"
            path=f"/World/Origin/object{i}"
            usd_path = f"/home/shi/Downloads/Collected_ur5e/objects/usd/{marker_name}.usd"
            cfg = sim_utils.UsdFileCfg(
                usd_path=usd_path,
                scale=(0.004, 0.004, 0.004),
                ##red,green,blue
                visual_material=sim_utils.GlassMdlCfg(glass_color=(0.5, 0.1, 0.1)),
            )
            cfg.func(path, cfg, translation=(rand_x, rand_y, rand_z))

        


        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.0, 0.0, 0.0)
        # frame_marker_cfg = FRAME_MARKER_CFG.copy()
        
        self.scene.ee_frame = FrameTransformerCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base_link", 
                debug_vis=False,
                visualizer_cfg=frame_marker_cfg,
                target_frames=[
                    FrameTransformerCfg.FrameCfg(
                        prim_path="{ENV_REGEX_NS}/Robot/tool0",  # Update this to the end effector link of UR5e
                        name="end_effector",
                        offset=OffsetCfg(
                            pos=[0.0, 0.0, 0.12],  # Adjust the offset if necessary
                        ),
                    ),
                ],
            )
    


@configclass
class v3ur5e_v3_CubeLiftEnvCfg_PLAY(v3ur5eCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
