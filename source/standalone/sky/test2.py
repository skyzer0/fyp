

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""
import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument("--cpu", action="store_true", default='cpu', help="Use CPU device for camera output.")
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

parser.add_argument(
    "--save",
    action="store_true",
    default=False,
    help="Save the data from camera at index specified by ``--camera_id``.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余部分代码在此之后编写。"""

from omni.isaac.lab.sim import SimulationCfg, SimulationContext
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, AssetBaseCfg
from omni.isaac.lab_assets import UR5E_CFG
import torch
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms
from omni.isaac.lab.assets import RigidObject, RigidObjectCfg

#TESTING
from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG
from omni.isaac.lab_assets import UR10_CFG
from omni.isaac.lab_assets import UR5E_CFG
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.utils import convert_dict_to_backend
from omni.isaac.lab.sensors import FrameTransformerCfg
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab_assets import UR5E_CFG_v2
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG  # isort: skip
from omni.isaac.core.utils.stage import add_reference_to_stage
import numpy as np
import os
import omni.replicator.core as rep
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg,GroundPlaneCfg
# 配置类

def euler_to_quaternion(roll, pitch, yaw):
    # Convert degrees to radians
    roll, pitch, yaw = np.radians([roll, pitch, yaw])
    
    # Compute quaternion components
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return qw, qx, qy, qz
@configclass
class ur5e_SceneCfg(InteractiveSceneCfg):
    """设计场景，通过 USD 文件生成地面、光源、对象和网格。"""

    # sky_light = AssetBaseCfg(
    #     prim_path="/World/skyLight",
    #     spawn=sim_utils.DomeLightCfg(
    #         intensity=1000.0,
    #         texture_file=
    #         f"/home/simtech/Downloads/Collected_ur5e/cobblestone_street_night_8k.hdr",
    #         ),
    #         )

    # 地面
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    robot: AssetBaseCfg= (UR5E_CFG_v2.replace(prim_path="{ENV_REGEX_NS}/Robot"))
    
    # 灯光
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    table1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table1",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )
    table2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table2",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[1, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )
    
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robotiq_85_base_link/front_cam",
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
            focal_length=22.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        ##default facing down the z-axis
        ##increase the z value to move the camera down
        offset=CameraCfg.OffsetCfg(
            pos=(0, 0.0, 0), 
            rot=euler_to_quaternion(0,90,0), 
            convention="ros"
        ),

    )

##
def object():
    for i in range(1, 48):
        rand_x = np.random.uniform(0.3, 0.7)
        rand_y = np.random.uniform(-0.2, 0.2)
        rand_z = np.random.uniform(0, 0.3)
        marker_name = f"object{i}"
        path = f"/World/Origin/object{i}"
        usd_path = f"/home/simtech/Downloads/Collected_ur5e/objects/usd/{marker_name}.usd"
        cfg = sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(0.003, 0.003, 0.003),
            visual_material=sim_utils.GlassMdlCfg(glass_color=(5.0, 1.0, 1.0)),
        )
        cfg.func(path, cfg, translation=(rand_x, rand_y, rand_z))

# def random_quaternion():
#     while True:
#         q = torch.tensor([np.random.uniform(-1, 1) for _ in range(4)])
#         if torch.norm(q) < 1:
#             q = q / torch.norm(q)
#             if q[2] < 0:  # Ensure the end effector is facing downward
#                 return q


def generate_random_goals():
    center_x = (0.5 + 0.7) / 2
    center_y = (-0.4 + 0.4) / 2
    max_offset_x = max(abs(0.5 - center_x), abs(0.7 - center_x))
    max_offset_y = max(abs(-0.4 - center_y), abs(0.4 - center_y))
    

    x = np.random.uniform(0.3, 0.8)
    y = np.random.uniform(-0.4, 0.4)
    z = np.random.uniform(0.3, 0.5)

    roll=np.random.uniform(-90, 90)
    offset_x = x - center_x
    offset_y = y - center_y
    
    yaw = -15 * (offset_x / max_offset_x)
    pitch = -15 * (offset_y / max_offset_y)
    ##(rotate, negative is point away from robot, left/right)
    quaternion = euler_to_quaternion(roll, 90-yaw , pitch)
    goal = [[x, y, z] + list(quaternion)]
    return torch.tensor(goal, device="cpu")

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    camera = scene["camera"]
    # print("camera_data", camera.data)
    camera_index = args_cli.camera_id
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output", "camera")
    rep_writer = rep.BasicWriter(
        output_dir=output_dir,
        frame_padding=0,
        colorize_instance_id_segmentation=camera.cfg.colorize_instance_id_segmentation,
        colorize_instance_segmentation=camera.cfg.colorize_instance_segmentation,
        colorize_semantic_segmentation=camera.cfg.colorize_semantic_segmentation,
    )

    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["robotiq_85_base_link"])
    robot_entity_cfg.resolve(scene)
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    sim_dt = sim.get_physics_dt()
    count = 0
    current_goal_idx = 0
    goal_reached = False

    while simulation_app.is_running():
        

        single_cam_data = convert_dict_to_backend(camera.data.output[camera_index], backend="numpy")
        single_cam_info = camera.data.info[camera_index]
        rep_output = {"annotators": {}}

        if count % 150 == 0:
            ee_goals = generate_random_goals() 
            ee_goals = torch.tensor(ee_goals, device=sim.device)
            ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
            ik_commands[:] = ee_goals[current_goal_idx]
            count = 0
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            ik_commands[:] = ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
            goal_reached = False
        else:
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
            ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = robot.data.root_state_w[:, 0:7]
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

            # Check if goal is reached (this is a placeholder, you may need a more specific condition)
            goal_error = torch.norm(ee_pos_b - ee_goals[current_goal_idx, 0:3])
            if goal_error < 0.08:  # Assuming a threshold of 0.01 for goal reached
                goal_reached = True

        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)
        
        if goal_reached and args_cli.save:
            # Save image if goal is reached
            rep_output = {"annotators": {}}
            for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
                if info is not None:
                    rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                else:
                    rep_output["annotators"][key] = {"render_product": {"data": data}}
            rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
            rep_writer.write(rep_output)
            goal_reached = False  # Reset flag after saving

        

def main():
    """主函数。"""
    # 初始化仿真上下文
    sim_cfg = SimulationCfg(dt=0.01, substeps=1,device="cpu" if args_cli.cpu else "cuda:1")
    sim = SimulationContext(sim_cfg)


    # 设置主摄像头视角
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    object()
    # 创建场景
    scene_cfg = ur5e_SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # 重置仿真
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # 运行主函数
    main()
    # 关闭仿真应用
    simulation_app.close()
