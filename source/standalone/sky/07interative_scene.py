

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""
import argparse
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# 启动 Omniverse 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""其余部分代码在此之后编写。"""

from omni.isaac.lab.sim import SimulationCfg, SimulationContext
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, AssetBaseCfg
from omni.isaac.lab_assets import UR5E_CFG,UR5E_hole_CFG
import torch
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
import random

# 配置类
@configclass
class ur5e_SceneCfg(InteractiveSceneCfg):
    """设计场景，通过 USD 文件生成地面、光源、对象和网格。"""

    # 地面
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # 灯光
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # 设置 UR5e 机械臂
    robot: UR5E_CFG= (UR5E_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"))
    static_robot= UR5E_hole_CFG.replace(prim_path="{ENV_REGEX_NS}/static_robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """运行仿真循环。"""
    # 提取场景实体
    robot1 = scene["robot"]
    robot2 = scene["static_robot"]
    # 定义仿真步长
    sim_dt = sim.get_physics_dt()
    count = 0

    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 100 == 0:
            # reset counter
            count = 0
            # print("[INFO]: Resetting robot state...")
            # # Apply random action
            # -- generate random joint efforts
            random_num=random.uniform(-90, 90)
            efforts = torch.tensor([random_num,random_num,random_num,random_num,random_num,random_num], device='cuda:0')
            # -- apply action to the robot
            robot1.set_joint_effort_target(efforts)
            robot2.set_joint_effort_target(efforts)
             # -- write data to sim
            robot1.write_data_to_sim()
            robot2.write_data_to_sim()
            print("print(robot.data.joint_pos",robot1.data.joint_pos)
            print("print(robot.data.joint_pos",robot2.data.joint_pos)
            robot1.reset()
            robot2.reset()
 
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        robot1.update(sim_dt)
        robot2.update(sim_dt)

def main():
    """主函数。"""
    # 初始化仿真上下文
    sim_cfg = SimulationCfg(dt=0.01, substeps=1)
    sim = SimulationContext(sim_cfg)

    # 设置主摄像头视角
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

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
