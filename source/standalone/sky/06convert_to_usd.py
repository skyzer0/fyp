# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to convert multiple OBJ/STL/FBX files into USD format.

This script uses the asset converter extension from Isaac Sim (``omni.kit.asset_converter``) to convert a
range of OBJ/STL/FBX assets into USD format. It is designed as a convenience script for command-line use.

Launch Isaac Sim Simulator first.
"""

import argparse
import contextlib
import os
from omni.isaac.lab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert mesh files into USD format.")
parser.add_argument(
    "start_index",
    type=int,
    help="The starting index of the input mesh files.",
)
parser.add_argument(
    "end_index",
    type=int,
    help="The ending index of the input mesh files.",
)
parser.add_argument(
    "input_dir",
    type=str,
    help="The directory containing input mesh files.",
)
parser.add_argument(
    "output_dir",
    type=str,
    help="The directory to store USD files.",
)
parser.add_argument(
    "--make-instanceable",
    action="store_true",
    default=False,
    help="Make the asset instanceable for efficient cloning.",
)
parser.add_argument(
    "--collision-approximation",
    type=str,
    default="convexDecomposition",
    choices=["convexDecomposition", "convexHull", "none"],
    help=(
        'The method used for approximating collision mesh. Set to "none" '
        "to not add a collision mesh to the converted mesh."
    ),
)
parser.add_argument(
    "--mass",
    type=float,
    default=None,
    help="The mass (in kg) to assign to the converted asset. If not provided, then no mass is added.",
)
# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import carb
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.lab.sim.converters import MeshConverter, MeshConverterCfg
from omni.isaac.lab.sim.schemas import schemas_cfg
from omni.isaac.lab.utils.assets import check_file_path
from omni.isaac.lab.utils.dict import print_dict


def main():
    # Create destination path
    input_dir = args_cli.input_dir
    output_dir = args_cli.output_dir

    # Iterate through the range of STL files
    for i in range(args_cli.start_index, args_cli.end_index + 1):
        mesh_path = os.path.join(input_dir, f"object{i}.stl")
        dest_path = os.path.join(output_dir, f"object{i}.usd")

        # Check valid file path
        if not os.path.isabs(mesh_path):
            mesh_path = os.path.abspath(mesh_path)
        if not check_file_path(mesh_path):
            print(f"Invalid mesh file path: {mesh_path}")
            continue

        # Mass properties
        if args_cli.mass is not None:
            mass_props = schemas_cfg.MassPropertiesCfg(mass=args_cli.mass)
            rigid_props = schemas_cfg.RigidBodyPropertiesCfg()
        else:
            mass_props = None
            rigid_props = None

        # Collision properties
        collision_props = schemas_cfg.CollisionPropertiesCfg(collision_enabled=args_cli.collision_approximation != "none")

        # Create Mesh converter config
        mesh_converter_cfg = MeshConverterCfg(
            mass_props=mass_props,
            rigid_props=rigid_props,
            collision_props=collision_props,
            asset_path=mesh_path,
            force_usd_conversion=True,
            usd_dir=os.path.dirname(dest_path),
            usd_file_name=os.path.basename(dest_path),
            make_instanceable=args_cli.make_instanceable,
            collision_approximation=args_cli.collision_approximation,
        )

        # Print info
        print("-" * 80)
        print("-" * 80)
        print(f"Input Mesh file: {mesh_path}")
        print("Mesh importer config:")
        print_dict(mesh_converter_cfg.to_dict(), nesting=0)
        print("-" * 80)
        print("-" * 80)

        # Create Mesh converter and import the file
        mesh_converter = MeshConverter(mesh_converter_cfg)
        # Print output
        print("Mesh importer output:")
        print(f"Generated USD file: {mesh_converter.usd_path}")
        print("-" * 80)
        print("-" * 80)

        # Determine if there is a GUI to update:
        # Acquire settings interface
        carb_settings_iface = carb.settings.get_settings()
        # Read flag for whether a local GUI is enabled
        local_gui = carb_settings_iface.get("/app/window/enabled")
        # Read flag for whether livestreaming GUI is enabled
        livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

        # Simulate scene (if not headless)
        if local_gui or livestream_gui:
            # Open the stage with USD
            stage_utils.open_stage(mesh_converter.usd_path)
            # Reinitialize the simulation
            app = omni.kit.app.get_app_interface()
            # Run simulation
            with contextlib.suppress(KeyboardInterrupt):
                while app.is_running():
                    # Perform step
                    app.update()


if __name__ == "__main__":
    # Run the main function
    main()
    # Close sim app
    simulation_app.close()
