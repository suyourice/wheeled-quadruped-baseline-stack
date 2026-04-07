# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Utility to convert a URDF into USD format.

Unified Robot Description Format (URDF) is an XML file format used in ROS to describe all elements of
a robot. For more information, see: http://wiki.ros.org/urdf

This script uses the URDF importer extension from Isaac Sim (``isaacsim.asset.importer.urdf``) to convert a
URDF asset into USD format. It is designed as a convenience script for command-line use. For more
information on the URDF importer, see the documentation for the extension:
https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/ext_isaacsim_asset_importer_urdf.html


positional arguments:
  input               Path to the input URDF file. (default: assets/go2w/urdf/go2w.urdf)
  output              Path to the output USD file. (default: assets/go2w/urdf/go2w_description/go2w.usd)

optional arguments:
  -h, --help                Show this help message and exit
  --merge-joints            Consolidate links connected by fixed joints. (default: True)
  --fix-base                Fix the base link in place. (default: False)
  --cleanup-resolved-urdf   Delete the temporary URDF with absolute mesh paths after conversion. (default: False)
  --joint-stiffness         USD-level joint stiffness; overridden by ImplicitActuatorCfg. (default: 0.0)
  --joint-damping           USD-level joint damping; overridden by ImplicitActuatorCfg. (default: 0.0)
  --joint-target-type       USD-level drive type {position, velocity, none}. (default: none)

Default usage (go2w):
    ./isaaclab.sh -p scripts/tools/convert_urdf.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Utility to convert a URDF into USD format.")
parser.add_argument(
    "input",
    type=str,
    nargs="?",
    default="assets/go2w/urdf/go2w.urdf",
    help="Path to the input URDF file.",
)
parser.add_argument(
    "output",
    type=str,
    nargs="?",
    default="assets/go2w/urdf/go2w_description/go2w.usd",
    help="Path to the output USD file.",
)
parser.add_argument(
    "--merge-joints",
    action="store_true",
    default=True,
    # Merge links connected by fixed joints into one rigid body.
    # Reduces simulation cost but may affect contact dynamics accuracy.
    # Trade-off: performance vs. fidelity. Evaluate carefully before use.
    help="Consolidate links that are connected by fixed joints.",
)
parser.add_argument(
    "--fix-base",
    action="store_true",
    default=False,
    # Pin the base link in place. Use for fixed-base arms, NOT for legged/wheeled robots.
    help="Fix the base to where it is imported.",
)
parser.add_argument(
    "--joint-stiffness",
    type=float,
    default=0.0,
    # go2w URDF has no <dynamics> tags, so there is no URDF-level stiffness to follow.
    # Default 0: all joint drive gains are handled by ImplicitActuatorCfg at runtime.
    help="The stiffness of the joint drive.",
)
parser.add_argument(
    "--joint-damping",
    type=float,
    default=0.0,
    # Same as stiffness: no URDF dynamics defined; ImplicitActuatorCfg controls damping.
    help="The damping of the joint drive.",
)
parser.add_argument(
    "--joint-target-type",
    type=str,
    default="none",
    choices=["position", "velocity", "none"],
    # "none": no USD-level drive; all actuation delegated to ImplicitActuatorCfg.
    # go2w uses mixed control (velocity for wheels, position for legs), handled in code.
    help="The type of control to use for the joint drive.",
)
parser.add_argument(
    "--cleanup-resolved-urdf",
    action="store_true",
    default=False,
    help="Delete the temporary URDF with absolute mesh paths after conversion.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import os
import sys
from pathlib import Path

import carb
import omni.kit.app

# Add scripts/tools/ to path so urdf_mesh_paths can be imported from any cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import isaaclab.sim as sim_utils
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict
from urdf_mesh_paths import prepare_urdf_for_import


def main():
    # check valid file path
    urdf_path = args_cli.input
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.abspath(urdf_path)
    if not check_file_path(urdf_path):
        raise ValueError(f"Invalid file path: {urdf_path}")
    # create destination path
    dest_path = args_cli.output
    if not os.path.isabs(dest_path):
        dest_path = os.path.abspath(dest_path)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    prepared_urdf_path, mesh_path_rewrites = prepare_urdf_for_import(urdf_path, usd_dir=os.path.dirname(dest_path))
    created_temp_urdf = os.path.abspath(prepared_urdf_path) != urdf_path

    try:
        # Create Urdf converter config
        urdf_converter_cfg = UrdfConverterCfg(
            asset_path=prepared_urdf_path,
            usd_dir=os.path.dirname(dest_path),
            usd_file_name=os.path.basename(dest_path),
            fix_base=args_cli.fix_base,
            merge_fixed_joints=args_cli.merge_joints,
            force_usd_conversion=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=args_cli.joint_stiffness,
                    damping=args_cli.joint_damping,
                ),
                target_type=args_cli.joint_target_type,
            ),
        )

        # Print info
        print("-" * 80)
        print("-" * 80)
        print(f"Input URDF file: {urdf_path}")
        if mesh_path_rewrites:
            print(f"Resolved mesh paths in: {prepared_urdf_path}")
            print(f"Resolved {len(mesh_path_rewrites)} mesh filename(s) to absolute paths before import.")
            preview_limit = 8
            for original, resolved in mesh_path_rewrites[:preview_limit]:
                print(f"  {original} -> {resolved}")
            if len(mesh_path_rewrites) > preview_limit:
                print(f"  ... {len(mesh_path_rewrites) - preview_limit} more")
        print("URDF importer config:")
        print_dict(urdf_converter_cfg.to_dict(), nesting=0)
        print("-" * 80)
        print("-" * 80)

        # Create Urdf converter and import the file
        urdf_converter = UrdfConverter(urdf_converter_cfg)
        # print output
        print("URDF importer output:")
        print(f"Generated USD file: {urdf_converter.usd_path}")
        print("-" * 80)
        print("-" * 80)

        # Determine if there is a GUI to update:
        # acquire settings interface
        carb_settings_iface = carb.settings.get_settings()
        # read flag for whether a local GUI is enabled
        local_gui = carb_settings_iface.get("/app/window/enabled")
        # read flag for whether livestreaming GUI is enabled
        livestream_gui = carb_settings_iface.get("/app/livestream/enabled")

        # Simulate scene (if not headless)
        if local_gui or livestream_gui:
            # Open the stage with USD
            sim_utils.open_stage(urdf_converter.usd_path)
            # Reinitialize the simulation
            app = omni.kit.app.get_app_interface()
            # Run simulation
            with contextlib.suppress(KeyboardInterrupt):
                while app.is_running():
                    # perform step
                    app.update()
    finally:
        if args_cli.cleanup_resolved_urdf and created_temp_urdf:
            _cleanup_resolved_urdf(prepared_urdf_path)


def _cleanup_resolved_urdf(prepared_urdf_path: str):
    resolved_path = Path(prepared_urdf_path)
    if resolved_path.is_file():
        resolved_path.unlink()

    with contextlib.suppress(OSError):
        resolved_path.parent.rmdir()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
