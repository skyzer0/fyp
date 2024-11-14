


from __future__ import annotations

"""Launch Isaac Sim Simulator first."""
import argparse

from omni.isaac.lab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

from omni.isaac.lab.sim import SimulationCfg, SimulationContext
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.assets import Articulation
# from omni.isaac.lab_assets import UR5E_GRIPPER_CFG
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg
import torch
import random

# Copyright (c) 2022-2024, The lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


def design_scene():

    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""

    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    UR5E_hole_CFG = ArticulationCfg(
	spawn=sim_utils.UsdFileCfg(
	usd_path=f"/home/simtech/Downloads/Collected_ur5e/ur5e.usd",
	rigid_props=sim_utils.RigidBodyPropertiesCfg(
		disable_gravity=True,
		max_depenetration_velocity=5.0,),
		activate_contact_sensors=False,),
	init_state=ArticulationCfg.InitialStateCfg(
		joint_pos={
				"shoulder_pan_joint": 0.0,
				"shoulder_lift_joint": -2.0,
				"elbow_joint": 2.0,
				"wrist_1_joint": -2.0,
				"wrist_2_joint": -2.0,
				"wrist_3_joint":0.0,},
		pos=(1,0,1)),
	actuators={
			"arm": ImplicitActuatorCfg(
			joint_names_expr=[".*"],
			velocity_limit=100.0,
			effort_limit=87.0,
			stiffness=800.0,
			damping=40.0,),},)

    UR5E_hole_CFG.prim_path="/World/Robot"
    robot=Articulation(cfg=UR5E_hole_CFG)

    #return scene infomation
    scene_entities ={"ur5e": robot}
    return scene_entities


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation]):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["ur5e"]
    print(robot.data)
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            root_state = robot.data.default_root_state.clone()
            robot.write_root_state_to_sim(root_state)


            print("[INFO]: Resetting robot state...")
            # Apply random action
            # -- generate random joint efforts
            random_num=random.uniform(-90, 90)
            efforts = torch.tensor([random_num,random_num,random_num,random_num,random_num,random_num], device='cuda:0')
            # -- apply action to the robot
            robot.set_joint_effort_target(efforts)
             # -- write data to sim
            robot.write_data_to_sim()
            print("print(robot.data.joint_po",robot.data.joint_pos)
            robot.reset()
 
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        robot.update(sim_dt)


   
def main():
    """Main function."""
    
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device="cpu", use_gpu_pipeline=False)

    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

   # Design scene
    scene_entities = design_scene()

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Run the simulator
    run_simulator(sim, scene_entities)



if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()




