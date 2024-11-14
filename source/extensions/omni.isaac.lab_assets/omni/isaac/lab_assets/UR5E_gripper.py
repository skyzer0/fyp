# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots.

The following configuration parameters are available:

Reference: https://github.com/ros-industrial/universal_robot
"""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##
UR5E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        ##change this to your usd path
        usd_path=f"/home/shi/Downloads/Collected_ur5e/ur5e.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            ## include all the revolute joints that is been using in robot
            ## you can open usd in isaac sim/ check your urdf file if you are not sure what are the joints
            ## the numbers beside indicate the initial angle of your robot when it spawn in the simulation
            ## they are in radius (negativa indicate anti-clock rotation and position indicate clock rotation)
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.712,
            "elbow_joint": 1.712,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
        },
        ##you can change the position/oritation of the robot spawn in the map (pos(0,0,0,0), rot(1,0,0,0,0))
        pos=(0.0,0.0,0.0)
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=180.0,
            effort_limit=150.0,
            stiffness=38000,
            damping=970,
        ),
    },
)
"""Configuration of UR5e arm using implicit actuator models."""


UR5E_cylinder_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        ##change this to your usd path
        usd_path=f"/home/shi/Downloads/Collected_ur5e/ur5e_cylinder.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            ## include all the revolute joints that is been using in robot
            ## you can open usd in isaac sim/ check your urdf file if you are not sure what are the joints
            ## the numbers beside indicate the initial angle of your robot when it spawn in the simulation
            ## they are in radius (negativa indicate anti-clock rotation and position indicate clock rotation)
            "shoulder_pan_joint":  0.0,
            "shoulder_lift_joint": -2.0,
            "elbow_joint": 2.0,
            "wrist_1_joint": -2.0,
            "wrist_2_joint": -2.0,
            "wrist_3_joint":0.0,
        },
        ##you can change the position/oritation of the robot spawn in the map (pos(0,0,0,0), rot(1,0,0,0,0))
        pos=(0.0,0.0,0.0)
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)
"""Configuration of UR5e arm using implicit actuator models."""

UR5E_hole_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/shi/Downloads/Collected_ur5e/ur5e_hole_camera.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            #180 degree
            "shoulder_pan_joint":  3.1415926536,
            #-110 degree
            "shoulder_lift_joint": -1.9198621772,
            #110 degree
            "elbow_joint": 1.9198621772,
            #-90 degree
            "wrist_1_joint": -1.5707963268,
            #90 degree
            "wrist_2_joint": 1.5707963268,
            "wrist_3_joint":0.0,
        },
        pos=(1,0,0)
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=800.0,
            damping=40.0,
        ),
    },
)

# UR5E_CFG_v1 = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"/home/shi/Downloads/Collected_ur5e/ur5e_v2.usd",
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             max_depenetration_velocity=5.0,
#         ),
#         activate_contact_sensors=False,
#     ),
#     prim_path="/World/robot",
#     init_state=ArticulationCfg.InitialStateCfg(
#         joint_pos={
#             "shoulder_pan_joint":  0.0,
#             "shoulder_lift_joint": -1.919,
#             "elbow_joint": 1.047,
#             "wrist_1_joint": -0.698,
#             "wrist_2_joint": -1.570,
#             "wrist_3_joint":0.0,
#             "body_f1_l":0.0,
#             "body_f1_r":0.0,
#             "f1_f2_l":0.0,
#             "f1_f2_r":0.0,
#             "f2_f4_l":0.0,
#             "f2_f4_r":0.0,
#             "f4_f3_l":0.0,
#             "f4_f3_r":0.0

#         },
#         pos=(0,0,00)
#     ),
#     actuators={
#         "ur5e_shoulder": ImplicitActuatorCfg(
#             joint_names_expr=["shoulder_pan_joint","shoulder_lift_joint","elbow_joint"],
#             effort_limit=87.0,
#             velocity_limit=2.175,
#             stiffness=80.0,
#             damping=4.0,
#         ),
#         "ur5e_forearm": ImplicitActuatorCfg(
#             joint_names_expr=["wrist_1_joint","wrist_2_joint","wrist_3_joint"],
#             effort_limit=12.0,
#             velocity_limit=2.61,
#             stiffness=80.0,
#             damping=4.0,
#         ),
#         "ur5e_hand": ImplicitActuatorCfg(
#             joint_names_expr=["body_f1_l"],
#             effort_limit=200.0,
#             velocity_limit=0.2,
#             stiffness=2e3,
#             damping=1e2,
#         ),
#     },
#     soft_joint_pos_limit_factor=1.0,
# )
# UR5E_CFG_v1_HIGH_PD_CFG = UR5E_CFG_v1.copy()
# UR5E_CFG_v1_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
# UR5E_CFG_v1_HIGH_PD_CFG.actuators["ur5e_shoulder"].stiffness = 400.0
# UR5E_CFG_v1_HIGH_PD_CFG.actuators["ur5e_shoulder"].damping = 80.0
# UR5E_CFG_v1_HIGH_PD_CFG.actuators["ur5e_forearm"].stiffness = 400.0
# UR5E_CFG_v1_HIGH_PD_CFG.actuators["ur5e_forearm"].damping = 80.0



UR5E_CFG_v2 = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/shi/Downloads/ur5e_2f85/robots/urdf/ur5e_with_gripper/ur5e_with_gripper.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        activate_contact_sensors=False,
    ),
    prim_path="/World/robot",
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint":  0.0,
            "shoulder_lift_joint": -1.745,
            "elbow_joint": 1.396,
            "wrist_1_joint": -1.221,
            "wrist_2_joint": -1.58,
            "wrist_3_joint":0.0,
            "robotiq_85_left_knuckle_joint":0.0,
            "robotiq_85_right_knuckle_joint":0.0


        },
        pos=(0,0,00)
    ),
    actuators={
        "ur5e_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint","shoulder_lift_joint","elbow_joint"],
            velocity_limit=200.0,
            effort_limit=160.0,
            stiffness=9000,
            damping=1000,
        ),
        "ur5e_forearm": ImplicitActuatorCfg(
            joint_names_expr=["wrist_1_joint","wrist_2_joint","wrist_3_joint"],
            velocity_limit=200.0,
            effort_limit=160.0,
            stiffness=9000,
            damping=1000,
        ),
        "ur5e_hand": ImplicitActuatorCfg(
            joint_names_expr=["robotiq_85_left_knuckle_joint","robotiq_85_right_knuckle_joint"],
            effort_limit=60.0,
            velocity_limit=0.5,
            stiffness=90000,
            damping=10,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
UR5E_CFG_v2_HIGH_PD_CFG = UR5E_CFG_v2.copy()
UR5E_CFG_v2_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
# UR5E_CFG_v2_HIGH_PD_CFG.actuators["ur5e_shoulder"].stiffness = 1000
# UR5E_CFG_v2_HIGH_PD_CFG.actuators["ur5e_shoulder"].damping = 70
# UR5E_CFG_v2_HIGH_PD_CFG.actuators["ur5e_forearm"].stiffness = 1100
# UR5E_CFG_v2_HIGH_PD_CFG.actuators["ur5e_forearm"].damping = 70
UR5E_CFG_v2_HIGH_PD_CFG.actuators["ur5e_hand"].stiffness = 30000
UR5E_CFG_v2_HIGH_PD_CFG.actuators["ur5e_hand"].damping = 50

