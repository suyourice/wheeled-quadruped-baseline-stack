# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import random
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from checkpoint_utils import configure_frozen_llc_action  # isort: skip
from play_common import (  # isort: skip
    TeeStream,
    build_teacher_policy,
    format_eval_metrics,
    override_play_obstacle_count,
    override_play_command_path_spawn,
    preflight_check_obstacle_slots,
    resolve_play_seed,
    select_episode_progress,
)

DEFAULT_DYNAMIC_OBSTACLE_SPEED_RANGE = (0.25, 0.70)
DEFAULT_DYNAMIC_OBSTACLE_LATERAL_SPEED = 0.12
DEFAULT_DYNAMIC_OBSTACLE_LONGITUDINAL_EXTENT = 2.0
DEFAULT_DYNAMIC_OBSTACLE_LATERAL_EXTENT = 0.30
DEFAULT_DYNAMIC_OBSTACLE_SPEED_CHANGE_INTERVAL = (0.8, 2.5)
DEFAULT_DYNAMIC_OBSTACLE_WANDER_FRACTION = 0.35
DEFAULT_RANDOM_OBSTACLE_FOOTPRINT_RANGE = (0.12, 0.60)
DEPTH_VIDEO_ENV = 0
DEPTH_VIDEO_SCALE = 4

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--episode_length_s",
    "--episode-length-s",
    dest="episode_length_s",
    type=float,
    default=None,
    help="Override play episode length in seconds. Structured play defaults to at least 60 s.",
)
parser.add_argument(
    "--num_obstacles",
    "--num-obstacles",
    dest="num_obstacles",
    type=int,
    default=None,
    help="Obstacle play tasks only: force this many active obstacles in the scene.",
)
parser.add_argument(
    "--dynamic_obstacles",
    "--dynamic-obstacles",
    dest="dynamic_obstacles",
    action="store_true",
    default=False,
    help="Navigation play tasks only: move active obstacles like pedestrians during inference.",
)
parser.add_argument(
    "--dynamic_obstacle_speed_range",
    "--dynamic-obstacle-speed-range",
    dest="dynamic_obstacle_speed_range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="Longitudinal pedestrian speed range [m/s]. Default: 0.25 0.70",
)
parser.add_argument(
    "--dynamic_obstacle_lateral_speed",
    "--dynamic-obstacle-lateral-speed",
    dest="dynamic_obstacle_lateral_speed",
    type=float,
    default=None,
    help="Maximum lateral drift speed [m/s]. Default: 0.12",
)
parser.add_argument(
    "--dynamic_obstacle_longitudinal_extent",
    "--dynamic-obstacle-longitudinal-extent",
    dest="dynamic_obstacle_longitudinal_extent",
    type=float,
    default=None,
    help="Maximum longitudinal excursion from each spawn point [m]. Default: 2.0",
)
parser.add_argument(
    "--dynamic_obstacle_lateral_extent",
    "--dynamic-obstacle-lateral-extent",
    dest="dynamic_obstacle_lateral_extent",
    type=float,
    default=None,
    help="Maximum lateral excursion from each spawn point [m]. Default: 0.30",
)
parser.add_argument(
    "--dynamic_obstacle_min_separation",
    "--dynamic-obstacle-min-separation",
    dest="dynamic_obstacle_min_separation",
    type=float,
    default=None,
    help="Minimum center-to-center distance maintained between moving obstacles [m]. Default: reset spacing.",
)
parser.add_argument(
    "--dynamic_obstacle_mixed_motion",
    "--dynamic-obstacle-mixed-motion",
    dest="dynamic_obstacle_mixed_motion",
    action="store_true",
    default=False,
    help="Dynamic play only: vary per-obstacle speeds over time and let a subset wander in random directions.",
)
parser.add_argument(
    "--dynamic_obstacle_speed_change_interval",
    "--dynamic-obstacle-speed-change-interval",
    dest="dynamic_obstacle_speed_change_interval",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="Mixed motion only: seconds between per-obstacle velocity updates. Default: 0.8 2.5",
)
parser.add_argument(
    "--dynamic_obstacle_wander_fraction",
    "--dynamic-obstacle-wander-fraction",
    dest="dynamic_obstacle_wander_fraction",
    type=float,
    default=None,
    help="Mixed motion only: fraction of active obstacles that follow random trajectories. Default: 0.35",
)
parser.add_argument(
    "--random_obstacle_shapes",
    "--random-obstacle-shapes",
    dest="random_obstacle_shapes",
    action="store_true",
    default=False,
    help="Navigation play tasks only: randomize each obstacle as a box, cylinder, or cone with a fixed height.",
)
parser.add_argument(
    "--random_obstacle_footprint_range",
    "--random-obstacle-footprint-range",
    dest="random_obstacle_footprint_range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="Random obstacle width/diameter range [m]. Default: 0.12 0.60",
)
parser.add_argument(
    "--structured_env",
    "--structured-env",
    dest="structured_env",
    choices=["none", "l_corridor", "serpentine_corridor", "t_corridor", "hospital_ward"],
    default="none",
    help="Navigation play tasks only: replace random reset with a known structured scene.",
)
parser.add_argument(
    "--corridor_width",
    "--corridor-width",
    dest="corridor_width",
    type=float,
    default=1.8,
    help="Structured l_corridor: free corridor width [m]. Default: 1.8",
)
parser.add_argument(
    "--corridor_leg_length",
    "--corridor-leg-length",
    dest="corridor_leg_length",
    type=float,
    default=6.0,
    help="Structured corridor: main horizontal leg length [m]. Default: 6.0",
)
parser.add_argument(
    "--corridor_turn_length",
    "--corridor-turn-length",
    dest="corridor_turn_length",
    type=float,
    default=None,
    help="Structured serpentine_corridor: vertical spacing between horizontal legs [m].",
)
parser.add_argument(
    "--corridor_wall_thickness",
    "--corridor-wall-thickness",
    dest="corridor_wall_thickness",
    type=float,
    default=0.20,
    help="Structured corridor: wall thickness [m]. Default: 0.20",
)
parser.add_argument(
    "--astar_grid_resolution",
    "--astar-grid-resolution",
    dest="astar_grid_resolution",
    type=float,
    default=0.20,
    help="Structured corridor: A* occupancy grid resolution [m]. Default: 0.20",
)
parser.add_argument(
    "--astar_lookahead_distance",
    "--astar-lookahead-distance",
    dest="astar_lookahead_distance",
    type=float,
    default=1.25,
    help="Structured corridor: path lookahead distance fed as local goal [m]. Default: 1.25",
)
parser.add_argument(
    "--astar_waypoint_reach_radius",
    "--astar-waypoint-reach-radius",
    dest="astar_waypoint_reach_radius",
    type=float,
    default=1.00,
    help="Structured corridor: radius for advancing held path waypoints [m]. Default: 1.00",
)
parser.add_argument(
    "--astar_clearance_cost_weight",
    "--astar-clearance-cost-weight",
    dest="astar_clearance_cost_weight",
    type=float,
    default=2.0,
    help="A* wall-proximity penalty weight (alpha). Higher = path stays further from walls. Default: 2.0",
)
parser.add_argument(
    "--astar_clearance_cost_sigma",
    "--astar-clearance-cost-sigma",
    dest="astar_clearance_cost_sigma",
    type=float,
    default=0.4,
    help="A* wall-proximity penalty decay length in metres (sigma). Default: 0.4",
)
parser.add_argument(
    "--no_adaptive_lookahead",
    "--no-adaptive-lookahead",
    dest="no_adaptive_lookahead",
    action="store_true",
    default=False,
    help="Disable adaptive lookahead; use fixed astar_lookahead_distance everywhere.",
)
parser.add_argument(
    "--lookahead_min",
    "--lookahead-min",
    dest="lookahead_min",
    type=float,
    default=0.6,
    help="Minimum lookahead distance near sharp turns [m]. Default: 0.6",
)
parser.add_argument(
    "--curvature_scan_horizon",
    "--curvature-scan-horizon",
    dest="curvature_scan_horizon",
    type=float,
    default=2.5,
    help="Arc-length ahead to scan for turns when computing adaptive lookahead [m]. Default: 2.5",
)
parser.add_argument(
    "--curvature_threshold",
    "--curvature-threshold",
    dest="curvature_threshold",
    type=float,
    default=0.3,
    help="Turn-angle threshold (rad) above which lookahead starts shrinking. Default: 0.3",
)
parser.add_argument(
    "--corner_rounding",
    "--corner-rounding",
    dest="corner_rounding",
    action="store_true",
    default=False,
    help="Enable Bezier arc corner rounding on the sparse A* path. Default: off",
)
parser.add_argument(
    "--corner_radius",
    "--corner-radius",
    dest="corner_radius",
    type=float,
    default=0.5,
    help="Corner rounding arc radius in metres. Default: 0.5",
)
parser.add_argument(
    "--structured_goal_done_radius",
    "--structured-goal-done-radius",
    dest="structured_goal_done_radius",
    type=float,
    default=0.70,
    help="Structured corridor: terminate episode within this distance of the final path goal [m]. Default: 0.70",
)
parser.add_argument(
    "--command_path_obstacles",
    "--command-path-obstacles",
    dest="command_path_obstacles",
    type=int,
    default=None,
    help="Obstacle play tasks only: force this many active slots to spawn in front of the current command direction.",
)
parser.add_argument(
    "--command_path_forward_range",
    "--command-path-forward-range",
    dest="command_path_forward_range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="Forward spawn range [m] for command-path obstacles. Default: 1.6 2.4",
)
parser.add_argument(
    "--command_path_lateral_range",
    "--command-path-lateral-range",
    dest="command_path_lateral_range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="Lateral spawn range [m] for command-path obstacles. Default: -0.35 0.35",
)
parser.add_argument(
    "--command_path_min_speed",
    "--command-path-min-speed",
    dest="command_path_min_speed",
    type=float,
    default=None,
    help="Below this command speed, command-path spawn falls back to random placement. Default: 0.2",
)
parser.add_argument(
    "--nav_case",
    "--nav-case",
    dest="nav_case",
    choices=[
        "random",
        "empty",
        "head_on",
        "left_edge",
        "right_edge",
        "diag_left",
        "diag_right",
        "off_left",
        "off_right",
        "narrow_gap",
    ],
    default="random",
    help="Navigation play tasks only: force the sampled obstacle template (legacy; use --scenario).",
)
parser.add_argument(
    "--scenario",
    dest="scenario",
    choices=[
        "random",
        "empty",
        "head_on",
        "left_edge",
        "right_edge",
        "diag_left",
        "diag_right",
        "off_left",
        "off_right",
        "narrow_gap",
        "narrow_gap_wide",
        "narrow_gap_barely",
        "partial_blockage_left_open",
        "partial_blockage_right_open",
        "cluttered",
    ],
    default=None,
    help=(
        "Navigation play tasks only: force this scenario template on every reset. "
        "Takes precedence over --nav_case. "
        "Choices include all Phase-0/1/2 templates."
    ),
)
parser.add_argument(
    "--fixed_layout",
    "--fixed-layout",
    dest="fixed_layout",
    action="store_true",
    default=False,
    help=(
        "Navigation play tasks only: use the same obstacle/goal layout on every "
        "episode reset (seeded from --seed). Does NOT affect training."
    ),
)
parser.add_argument(
    "--nav_fixed_start",
    "--nav-fixed-start",
    dest="nav_fixed_start",
    action="store_true",
    default=False,
    help="Navigation play tasks only: reset every episode from a fixed local start pose.",
)
parser.add_argument("--nav_start_x", "--nav-start-x", dest="nav_start_x", type=float, default=None)
parser.add_argument("--nav_start_y", "--nav-start-y", dest="nav_start_y", type=float, default=None)
parser.add_argument("--nav_start_yaw", "--nav-start-yaw", dest="nav_start_yaw", type=float, default=None)
parser.add_argument("--nav_goal_forward", "--nav-goal-forward", dest="nav_goal_forward", type=float, default=None)
parser.add_argument("--nav_goal_lateral", "--nav-goal-lateral", dest="nav_goal_lateral", type=float, default=None)
parser.add_argument(
    "--nav_goal_heading_jitter",
    "--nav-goal-heading-jitter",
    dest="nav_goal_heading_jitter",
    type=float,
    default=None,
)
parser.add_argument(
    "--nav_min_inter_obstacle_dist",
    "--nav-min-inter-obstacle-dist",
    dest="nav_min_inter_obstacle_dist",
    type=float,
    default=None,
    help="Navigation play tasks only: minimum distance between obstacles in meters.",
)
parser.add_argument(
    "--nav_start_exclusion_radius",
    "--nav-start-exclusion-radius",
    dest="nav_start_exclusion_radius",
    type=float,
    default=None,
    help="Navigation play tasks only: minimum obstacle distance from the episode start in meters.",
)
parser.add_argument(
    "--nav_goal_exclusion_radius",
    "--nav-goal-exclusion-radius",
    dest="nav_goal_exclusion_radius",
    type=float,
    default=None,
    help="Navigation play tasks only: minimum obstacle distance from the goal in meters.",
)
parser.add_argument(
    "--nav_head_on_progress_range",
    "--nav-head-on-progress-range",
    dest="nav_head_on_progress_range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="Navigation play tasks only: head_on obstacle progress along start-goal path, as fractions.",
)
parser.add_argument(
    "--no_astar",
    "--no-astar",
    dest="no_astar",
    action="store_true",
    default=False,
    help=(
        "Ablation: bypass A* path following and navigate directly to the final corridor "
        "goal without rolling waypoints. Requires --structured_env. Use to quantify the "
        "contribution of global planning (compare SR/SPL against the default A* mode)."
    ),
)
parser.add_argument(
    "--nav_log_interval",
    "--nav-log-interval",
    dest="nav_log_interval",
    type=int,
    default=120,
    help="Navigation play tasks only: print env debug every N sim steps. Use 0 to disable.",
)
parser.add_argument("--nav_log_env", "--nav-log-env", dest="nav_log_env", type=int, default=0)
parser.add_argument(
    "--nav_live_obstacle_labels",
    "--nav-live-obstacle-labels",
    dest="nav_live_obstacle_labels",
    action="store_true",
    default=False,
    help="Navigation play: draw live viewport-only labels above active obstacles.",
)
parser.add_argument(
    "--nav_contact_debug",
    "--nav-contact-debug",
    dest="nav_contact_debug",
    action="store_true",
    default=False,
    help="Navigation play tasks only: print obstacle-mask, clearance, and contact diagnostics each step.",
)
parser.add_argument(
    "--nav_contact_debug_steps",
    "--nav-contact-debug-steps",
    dest="nav_contact_debug_steps",
    type=int,
    default=12,
    help="Number of steps to print before exiting when --nav_contact_debug is enabled. Default: 12.",
)
parser.add_argument(
    "--nav_eval_episodes",
    "--nav-eval-episodes",
    dest="nav_eval_episodes",
    type=int,
    default=0,
    help=(
        "Navigation play tasks only: evaluate exactly this many trajectories, then exit. "
        "The evaluator admits only N trajectories and waits for each to terminate, so parallel "
        "auto-resets cannot truncate long-running samples. Use 0 for endless play."
    ),
)
parser.add_argument(
    "--hospital_maze_route_steps",
    "--hospital-maze-route-steps",
    dest="hospital_maze_route_steps",
    type=int,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="Hospital maze eval only: sample final routes with this inclusive junction-step range.",
)
parser.add_argument(
    "--terminate_on_final_goal",
    "--terminate-on-final-goal",
    dest="terminate_on_final_goal",
    action="store_true",
    help="Hospital maze eval only: end the episode on the first route completion instead of "
    "resampling a new route (goal_reached_and_resample keeps running by default, matching "
    "training, so path_progress_mean reflects sustained multi-route exposure). Use this only "
    "for a single-route success-rate/SPL protocol (e.g. maze_success) — enabling it for the "
    "long-horizon maze_train/static/dynamic scenarios changes what they measure.",
)
parser.add_argument(
    "--stuck_timeout_steps",
    "--stuck-timeout-steps",
    dest="stuck_timeout_steps",
    type=int,
    default=0,
    help="Force-reset envs that haven't moved --stuck_threshold m in this many steps. 0=disabled.",
)
parser.add_argument(
    "--stuck_threshold",
    "--stuck-threshold",
    dest="stuck_threshold",
    type=float,
    default=0.3,
    help="Displacement threshold (m) for stuck detection. Default: 0.3.",
)
parser.add_argument(
    "--seed_per_episode",
    "--seed-per-episode",
    dest="seed_per_episode",
    action="store_true",
    default=False,
    help="Increment fixed_layout_seed by 1 per completed episode (requires --seed). "
    "Ensures all ablation runs see the same sequence of layouts for fair comparison.",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument(
    "--play_name",
    "--play-name",
    dest="play_name",
    type=str,
    default=None,
    help=(
        "Name for this play run. Creates logs/nav_play/<name>/ and saves "
        "nav_debug.log, depth_camera.mp4 (if --depth_video), and nav_debug_env0.png automatically."
    ),
)
parser.add_argument(
    "--depth_video",
    "--depth-video",
    dest="depth_video",
    action="store_true",
    default=False,
    help="Save the student depth camera view as an MP4 video alongside the log.",
)
parser.add_argument(
    "--depth_video_steps",
    "--depth-video-steps",
    dest="depth_video_steps",
    type=int,
    default=0,
    help="Max steps to record for depth video. 0 = record entire run (default). "
    "Use e.g. 2000 to capture only the first ~2 episodes.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--disable_export",
    "--disable-export",
    dest="disable_export",
    action="store_true",
    default=False,
    help="Skip JIT/ONNX policy export before play/eval.",
)
parser.add_argument(
    "--teacher_steering",
    "--teacher-steering",
    dest="teacher_steering",
    action="store_true",
    default=False,
    help="Run the rule-based navigation teacher directly instead of loading a student checkpoint.",
)
parser.add_argument(
    "--locomotion_checkpoint",
    type=str,
    default=None,
    help="Flat locomotion checkpoint used to initialize the teacher frozen LLC in --teacher_steering mode.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

from collections import defaultdict
from datetime import datetime, timezone
import os
import subprocess
import time

import gymnasium as gym
import numpy as np
import torch
from matplotlib import colormaps as _cmaps
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import go2w.tasks  # noqa: F401
from go2w.tasks.manager_based.go2w import mdp as go2w_mdp
from go2w.tasks.manager_based.go2w.cfg.navigation.env import OBSTACLE_SIZE, make_play_obstacle_cfg
from go2w.tasks.manager_based.go2w.mdp.navigation.hospital.specs import (
    NAV_GOAL_SUCCESS_POSITION_THRESHOLD,
    NAV_WAYPOINT_COMMAND_MIN_FORWARD_PLAY,
    NAV_WAYPOINT_COMMAND_MAX_LATERAL_PLAY,
    NAV_WAYPOINT_COMMAND_MAX_HEADING_PLAY,
)
from go2w.tasks.manager_based.go2w.mdp.navigation.local_planning.obstacle_geometry import (
    OBSTACLE_SHAPE_CONE,
    OBSTACLE_SHAPE_CUBOID,
    OBSTACLE_SHAPE_CYLINDER,
)

from play_nav_debug import (  # isort: skip
    NAV_LIVE_LABEL_INTERVAL, NAV_LIVE_LABEL_SCALE, NAV_LIVE_LABEL_MAX,
    _LiveObstacleLabelDrawer,
    _nav_debug_corridor_plot_args,
    _ablation_apply_direct_goal,
    _get_nav_path_line_markers,
    _print_navigation_play_log,
    _print_nav_obstacle_label_log,
    _print_nav_contact_debug,
    _print_nav_episode_log,
)




def _override_dynamic_navigation_play(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    args_cli: argparse.Namespace,
):
    """Add pedestrian-style obstacle motion only for opt-in navigation play."""
    has_dynamic_params = any(
        value is not None
        for value in (
            args_cli.dynamic_obstacle_speed_range,
            args_cli.dynamic_obstacle_lateral_speed,
            args_cli.dynamic_obstacle_longitudinal_extent,
            args_cli.dynamic_obstacle_lateral_extent,
            args_cli.dynamic_obstacle_min_separation,
            args_cli.dynamic_obstacle_speed_change_interval,
            args_cli.dynamic_obstacle_wander_fraction,
        )
    )
    if not args_cli.dynamic_obstacles:
        if has_dynamic_params or args_cli.dynamic_obstacle_mixed_motion:
            raise ValueError("Dynamic obstacle tuning flags require --dynamic_obstacles.")
        return
    if not args_cli.dynamic_obstacle_mixed_motion and (
        args_cli.dynamic_obstacle_speed_change_interval is not None
        or args_cli.dynamic_obstacle_wander_fraction is not None
    ):
        raise ValueError(
            "--dynamic_obstacle_speed_change_interval and --dynamic_obstacle_wander_fraction "
            "require --dynamic_obstacle_mixed_motion."
        )

    events_cfg = getattr(env_cfg, "events", None)
    reset_obstacles = getattr(events_cfg, "reset_obstacles", None) if events_cfg is not None else None
    reset_params = getattr(reset_obstacles, "params", None)
    if reset_params is None or "fixed_scenario_template" not in reset_params:
        raise ValueError("--dynamic_obstacles requires a navigation play task.")

    speed_range = (
        tuple(args_cli.dynamic_obstacle_speed_range)
        if args_cli.dynamic_obstacle_speed_range is not None
        else DEFAULT_DYNAMIC_OBSTACLE_SPEED_RANGE
    )
    if speed_range[0] < 0.0 or speed_range[0] > speed_range[1]:
        raise ValueError("--dynamic_obstacle_speed_range requires 0 <= MIN <= MAX.")
    lateral_speed = (
        args_cli.dynamic_obstacle_lateral_speed
        if args_cli.dynamic_obstacle_lateral_speed is not None
        else DEFAULT_DYNAMIC_OBSTACLE_LATERAL_SPEED
    )
    longitudinal_extent = (
        args_cli.dynamic_obstacle_longitudinal_extent
        if args_cli.dynamic_obstacle_longitudinal_extent is not None
        else DEFAULT_DYNAMIC_OBSTACLE_LONGITUDINAL_EXTENT
    )
    lateral_extent = (
        args_cli.dynamic_obstacle_lateral_extent
        if args_cli.dynamic_obstacle_lateral_extent is not None
        else DEFAULT_DYNAMIC_OBSTACLE_LATERAL_EXTENT
    )
    min_separation = (
        args_cli.dynamic_obstacle_min_separation
        if args_cli.dynamic_obstacle_min_separation is not None
        else float(reset_params.get("min_inter_obstacle_dist", 0.7))
    )
    if min(lateral_speed, longitudinal_extent, lateral_extent, min_separation) < 0.0:
        raise ValueError("Dynamic obstacle speed, extents, and separation must be non-negative.")
    velocity_resample_interval_range = None
    random_trajectory_fraction = 0.0
    if args_cli.dynamic_obstacle_mixed_motion:
        velocity_resample_interval_range = (
            tuple(args_cli.dynamic_obstacle_speed_change_interval)
            if args_cli.dynamic_obstacle_speed_change_interval is not None
            else DEFAULT_DYNAMIC_OBSTACLE_SPEED_CHANGE_INTERVAL
        )
        if (
            velocity_resample_interval_range[0] <= 0.0
            or velocity_resample_interval_range[0] > velocity_resample_interval_range[1]
        ):
            raise ValueError("--dynamic_obstacle_speed_change_interval requires 0 < MIN <= MAX.")
        random_trajectory_fraction = (
            args_cli.dynamic_obstacle_wander_fraction
            if args_cli.dynamic_obstacle_wander_fraction is not None
            else DEFAULT_DYNAMIC_OBSTACLE_WANDER_FRACTION
        )
        if not 0.0 <= random_trajectory_fraction <= 1.0:
            raise ValueError("--dynamic_obstacle_wander_fraction requires 0 <= FRACTION <= 1.")

    dynamic_obstacle_names = reset_params.get("dynamic_obstacle_names", reset_params.get("obstacle_names", []))
    dynamic_obstacle_indices = reset_params.get("dynamic_obstacle_indices", None)
    events_cfg.dynamic_play_obstacles = EventTerm(
        func=go2w_mdp.move_dynamic_play_obstacles,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "obstacle_names": dynamic_obstacle_names,
            "obstacle_z": reset_params.get("obstacle_z", 0.30),
            "obstacle_indices": dynamic_obstacle_indices,
            "longitudinal_speed_range": speed_range,
            "lateral_speed_max": lateral_speed,
            "longitudinal_extent": longitudinal_extent,
            "lateral_extent": lateral_extent,
            "min_inter_obstacle_dist": min_separation,
            "velocity_resample_interval_range": velocity_resample_interval_range,
            "random_trajectory_fraction": random_trajectory_fraction,
            "goal_exclusion_radius": float(reset_params.get("goal_exclusion_radius", 0.9)),
            "robot_keepout_radius": float(reset_params.get("dynamic_robot_keepout_radius", 1.25)),
        },
    )
    print(
        "[INFO] Dynamic navigation play obstacles: "
        f"slots={len(dynamic_obstacle_names)}, speed={speed_range}, lateral_speed=+/-{lateral_speed:.2f}, "
        f"extent=({longitudinal_extent:.2f}, {lateral_extent:.2f}), "
        f"min_separation={min_separation:.2f}"
    )
    if velocity_resample_interval_range is not None:
        print(
            "[INFO] Mixed dynamic motion: "
            f"speed_change_interval={velocity_resample_interval_range}, "
            f"wander_fraction={random_trajectory_fraction:.2f}"
        )


def _override_random_navigation_play_obstacle_shapes(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    args_cli: argparse.Namespace,
    env_seed: int,
):
    """Replace play obstacle assets with seeded random fixed-height shapes."""
    if not args_cli.random_obstacle_shapes:
        if args_cli.random_obstacle_footprint_range is not None:
            raise ValueError("--random_obstacle_footprint_range requires --random_obstacle_shapes.")
        return

    events_cfg = getattr(env_cfg, "events", None)
    reset_obstacles = getattr(events_cfg, "reset_obstacles", None) if events_cfg is not None else None
    reset_params = getattr(reset_obstacles, "params", None)
    if reset_params is None or "fixed_scenario_template" not in reset_params:
        raise ValueError("--random_obstacle_shapes requires a navigation play task.")

    footprint_range = (
        tuple(args_cli.random_obstacle_footprint_range)
        if args_cli.random_obstacle_footprint_range is not None
        else DEFAULT_RANDOM_OBSTACLE_FOOTPRINT_RANGE
    )
    if footprint_range[0] <= 0.0 or footprint_range[0] > footprint_range[1]:
        raise ValueError("--random_obstacle_footprint_range requires 0 < MIN <= MAX.")

    required_separation = (2.0**0.5) * footprint_range[1] + 0.02
    if (
        args_cli.dynamic_obstacles
        and args_cli.dynamic_obstacle_min_separation is not None
        and args_cli.dynamic_obstacle_min_separation < required_separation
    ):
        raise ValueError(
            "--dynamic_obstacle_min_separation is too small for the random obstacle footprint range. "
            f"Use at least {required_separation:.2f} m or reduce the footprint MAX."
        )
    min_separation = float(reset_params.get("min_inter_obstacle_dist", 0.7))
    if min_separation < required_separation:
        reset_params["min_inter_obstacle_dist"] = required_separation
        print(
            "[INFO] Raised navigation obstacle spacing for random shapes: "
            f"{min_separation:.2f} -> {required_separation:.2f} m"
        )

    obstacle_names = reset_params.get("obstacle_names", [])
    if not obstacle_names:
        raise ValueError("--random_obstacle_shapes requires configured obstacle slots.")

    rng = random.Random(env_seed ^ 0x4F425354)
    shape_counts = {"cuboid": 0, "cylinder": 0, "cone": 0}
    shape_ids = {
        "cuboid": OBSTACLE_SHAPE_CUBOID,
        "cylinder": OBSTACLE_SHAPE_CYLINDER,
        "cone": OBSTACLE_SHAPE_CONE,
    }
    fixed_shape_ids: list[int] = []
    fixed_widths: list[float] = []
    fixed_depths: list[float] = []
    for idx, obstacle_name in enumerate(obstacle_names):
        shape_kind = rng.choice(tuple(shape_counts))
        width = rng.uniform(*footprint_range)
        depth = width if shape_kind in ("cylinder", "cone") else rng.uniform(*footprint_range)
        setattr(
            env_cfg.scene,
            obstacle_name,
            make_play_obstacle_cfg(obstacle_name, idx, shape_kind, (width, depth)),
        )
        shape_counts[shape_kind] += 1
        fixed_shape_ids.append(shape_ids[shape_kind])
        fixed_widths.append(width)
        fixed_depths.append(depth)

    reset_params["fixed_obstacle_shape_ids"] = tuple(fixed_shape_ids)
    reset_params["fixed_obstacle_widths"] = tuple(fixed_widths)
    reset_params["fixed_obstacle_depths"] = tuple(fixed_depths)

    print(
        "[INFO] Random navigation play obstacle shapes: "
        f"footprint={footprint_range}, height={OBSTACLE_SIZE[2]:.2f}, "
        f"cuboids={shape_counts['cuboid']}, cylinders={shape_counts['cylinder']}, cones={shape_counts['cone']}"
    )


def _override_structured_navigation_play(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    args_cli: argparse.Namespace,
):
    """Configure known structured navigation play scenes."""
    if args_cli.structured_env == "none":
        return
    if args_cli.structured_env not in ("l_corridor", "serpentine_corridor", "t_corridor", "hospital_ward"):
        raise ValueError(f"Unsupported --structured_env={args_cli.structured_env!r}.")

    if args_cli.corridor_width <= 0.8:
        raise ValueError("--corridor_width must be > 0.8 m for the Go2-W footprint.")
    if args_cli.corridor_leg_length <= args_cli.corridor_width + 1.0:
        raise ValueError("--corridor_leg_length must be at least corridor_width + 1.0 m.")
    if args_cli.corridor_turn_length is not None and args_cli.corridor_turn_length <= args_cli.corridor_width:
        raise ValueError("--corridor_turn_length must be larger than --corridor_width.")
    if args_cli.corridor_wall_thickness <= 0.0:
        raise ValueError("--corridor_wall_thickness must be positive.")
    if args_cli.astar_grid_resolution <= 0.0:
        raise ValueError("--astar_grid_resolution must be positive.")
    if args_cli.structured_goal_done_radius <= 0.0:
        raise ValueError("--structured_goal_done_radius must be positive.")

    events_cfg = getattr(env_cfg, "events", None)
    reset_obstacles = getattr(events_cfg, "reset_obstacles", None) if events_cfg is not None else None
    reset_base = getattr(events_cfg, "reset_base", None) if events_cfg is not None else None
    reset_params = getattr(reset_obstacles, "params", None)
    if reset_params is None or "obstacle_names" not in reset_params:
        raise ValueError("--structured_env requires a navigation play task with obstacle slots.")
    if reset_base is None:
        raise ValueError("--structured_env requires a reset_base event.")

    obstacle_names = list(reset_params.get("obstacle_names", []))

    # Prefer env-cfg corridor dimensions so task-specific envs (e.g. hospital_ward)
    # use their own configured values.  CLI args act as fallback for generic envs.
    _eff_width = reset_params.get("corridor_width", args_cli.corridor_width)
    _eff_leg = reset_params.get("leg_length", args_cli.corridor_leg_length)
    _eff_thickness = reset_params.get("wall_thickness", args_cli.corridor_wall_thickness)
    _eff_turn = reset_params.get("corridor_turn_length", args_cli.corridor_turn_length)

    wall_specs = go2w_mdp.structured_corridor_wall_specs(
        args_cli.structured_env,
        _eff_leg,
        _eff_width,
        _eff_thickness,
        _eff_turn,
    )
    wall_count = len(wall_specs)
    if len(obstacle_names) < wall_count:
        raise ValueError(
            f"{args_cli.structured_env} requires {wall_count} wall slots, "
            f"but the play scene only has {len(obstacle_names)} obstacle slots."
        )
    max_dynamic = len(obstacle_names) - wall_count
    requested_dynamic = args_cli.num_obstacles if args_cli.num_obstacles is not None else min(12, max_dynamic)
    dynamic_count = max(0, min(requested_dynamic, max_dynamic))
    if requested_dynamic > max_dynamic:
        print(
            "[WARN] Requested dynamic obstacle count exceeds structured scene capacity: "
            f"{requested_dynamic} -> {dynamic_count}."
        )

    shape_ids = list(reset_params.get("fixed_obstacle_shape_ids", (OBSTACLE_SHAPE_CUBOID,) * len(obstacle_names)))
    widths = list(reset_params.get("fixed_obstacle_widths", (OBSTACLE_SIZE[0],) * len(obstacle_names)))
    depths = list(reset_params.get("fixed_obstacle_depths", (OBSTACLE_SIZE[1],) * len(obstacle_names)))
    if len(shape_ids) != len(obstacle_names) or len(widths) != len(obstacle_names) or len(depths) != len(obstacle_names):
        shape_ids = [OBSTACLE_SHAPE_CUBOID] * len(obstacle_names)
        widths = [OBSTACLE_SIZE[0]] * len(obstacle_names)
        depths = [OBSTACLE_SIZE[1]] * len(obstacle_names)

    for slot_idx, obstacle_name in enumerate(obstacle_names[:wall_count]):
        _, _, _, wall_length, wall_thickness = wall_specs[slot_idx]
        setattr(
            env_cfg.scene,
            obstacle_name,
            make_play_obstacle_cfg(obstacle_name, slot_idx, "cuboid", (wall_length, wall_thickness)),
        )
        shape_ids[slot_idx] = OBSTACLE_SHAPE_CUBOID
        widths[slot_idx] = wall_length
        depths[slot_idx] = wall_thickness

    pose_range = reset_base.params.setdefault("pose_range", {})
    velocity_range = reset_base.params.setdefault("velocity_range", {})
    pose_range["x"] = (0.0, 0.0)
    pose_range["y"] = (0.0, 0.0)
    pose_range["yaw"] = (0.0, 0.0)
    for key in ("x", "y", "z", "roll", "pitch", "yaw"):
        velocity_range[key] = (0.0, 0.0)

    reset_obstacles.func = go2w_mdp.reset_structured_astar_corridor
    reset_obstacles.params = {
        "fixed_scenario_template": args_cli.structured_env,
        "corridor_kind": args_cli.structured_env,
        "obstacle_names": obstacle_names,
        "dynamic_obstacle_names": obstacle_names[wall_count: wall_count + dynamic_count],
        "dynamic_obstacle_indices": list(range(wall_count, wall_count + dynamic_count)),
        "dynamic_obstacle_count": dynamic_count,
        "corridor_width": _eff_width,
        "leg_length": _eff_leg,
        "corridor_turn_length": _eff_turn,
        "wall_thickness": _eff_thickness,
        "grid_resolution": args_cli.astar_grid_resolution,
        "robot_inflation": 0.50,
        "lookahead_distance": args_cli.astar_lookahead_distance,
        "waypoint_reach_radius": args_cli.astar_waypoint_reach_radius,
        "obstacle_z": reset_params.get("obstacle_z", 0.30),
        "park_distance": reset_params.get("park_distance", 1000.0),
        "min_inter_obstacle_dist": reset_params.get("min_inter_obstacle_dist", 0.75),
        "dynamic_start_exclusion_radius": 1.8,
        "goal_exclusion_radius": reset_params.get("goal_exclusion_radius", 0.9),
        "dynamic_robot_keepout_radius": 1.25,
        "obstacle_radius_margin": reset_params.get("obstacle_radius_margin", 0.0),
        "fixed_obstacle_shape_ids": tuple(shape_ids),
        "fixed_obstacle_widths": tuple(widths),
        "fixed_obstacle_depths": tuple(depths),
        "randomize_obstacle_yaw": reset_params.get("randomize_obstacle_yaw", True),
        "obstacle_yaw_range": reset_params.get("obstacle_yaw_range", (-3.141592653589793, 3.141592653589793)),
        "clearance_cost_weight": args_cli.astar_clearance_cost_weight,
        "clearance_cost_sigma": args_cli.astar_clearance_cost_sigma,
        "corner_rounding": args_cli.corner_rounding,
        "corner_radius": args_cli.corner_radius,
        "adaptive_lookahead": not args_cli.no_adaptive_lookahead,
        "lookahead_min": args_cli.lookahead_min,
        "curvature_scan_horizon": args_cli.curvature_scan_horizon,
        "curvature_threshold": args_cli.curvature_threshold,
    }
    terminations_cfg = getattr(env_cfg, "terminations", None)
    if terminations_cfg is not None:
        terminations_cfg.structured_goal_reached = DoneTerm(
            func=go2w_mdp.navigation_path_final_goal_reached,
            params={"position_threshold": args_cli.structured_goal_done_radius},
        )
    # Write effective values back so callers (logging, plot script) see the real dims.
    args_cli.corridor_width = _eff_width
    args_cli.corridor_leg_length = _eff_leg
    args_cli.corridor_wall_thickness = _eff_thickness
    if _eff_turn is not None:
        args_cli.corridor_turn_length = _eff_turn

    print(
        "[INFO] Structured navigation play: "
        f"env={args_cli.structured_env}, wall_slots={wall_count}, dynamic_slots={dynamic_count}, "
        f"width={_eff_width:.2f}, leg_length={_eff_leg:.2f}, "
        f"turn_length={_eff_turn}, "
        f"A*_resolution={args_cli.astar_grid_resolution:.2f}, lookahead={args_cli.astar_lookahead_distance:.2f}, "
        f"reach_radius={args_cli.astar_waypoint_reach_radius:.2f}, "
        f"done_radius={args_cli.structured_goal_done_radius:.2f}"
    )


def _override_navigation_play_goal_command_params(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    args_cli: argparse.Namespace,
) -> dict[str, float]:
    """Apply play-time local goal command caps before the observation manager is built."""
    overrides = {
        "command_min_forward": NAV_WAYPOINT_COMMAND_MIN_FORWARD_PLAY,
        "command_max_lateral": NAV_WAYPOINT_COMMAND_MAX_LATERAL_PLAY,
        "command_max_heading": NAV_WAYPOINT_COMMAND_MAX_HEADING_PLAY,
        "command_turn_slowdown_heading": 0.45,
        "command_turn_slowdown_min_forward": 0.25,
    }
    obs_cfg = getattr(env_cfg, "observations", None)
    if obs_cfg is None:
        return overrides

    for obs_group in vars(obs_cfg).values():
        if not hasattr(obs_group, "__dict__"):
            continue
        for term in vars(obs_group).values():
            if hasattr(term, "func") and getattr(term.func, "__name__", "") == "local_goal_command_b":
                term.params.update(overrides)
    return overrides


def _override_play_episode_length(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    args_cli: argparse.Namespace,
):
    """Override play timeout without changing training configs."""
    requested = args_cli.episode_length_s
    if requested is None and args_cli.structured_env != "none":
        requested = max(float(getattr(env_cfg, "episode_length_s", 20.0)), 60.0)
    if requested is None:
        return
    if requested <= 0.0:
        raise ValueError("--episode_length_s must be positive.")
    env_cfg.episode_length_s = float(requested)
    print(f"[INFO] Play episode length: {env_cfg.episode_length_s:.1f} s")


def _override_hospital_maze_route_steps(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    args_cli: argparse.Namespace,
) -> None:
    """Set an eval-only hospital-maze route range without touching training cfgs."""
    if args_cli.hospital_maze_route_steps is None:
        return
    min_steps, max_steps = args_cli.hospital_maze_route_steps
    if min_steps < 1 or max_steps < min_steps:
        raise ValueError("--hospital_maze_route_steps requires 1 <= MIN <= MAX.")
    reset_obstacles = getattr(getattr(env_cfg, "events", None), "reset_obstacles", None)
    params = getattr(reset_obstacles, "params", None)
    if params is None or getattr(reset_obstacles.func, "__name__", "") != "reset_hospital_maze_training":
        raise ValueError("--hospital_maze_route_steps requires a hospital maze task.")
    params.update({
        "min_path_steps_override": min_steps,
        "max_path_steps_override": max_steps,
        "long_path_max_steps_override": max_steps,
        "long_path_probability_override": 0.0,
    })
    print(f"[INFO] Hospital maze eval route: {min_steps}-{max_steps} junction steps")


def _override_terminate_on_final_goal(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    args_cli: argparse.Namespace,
) -> None:
    """Add a single-route-completion termination for the success/SPL protocol.

    Off by default: maze_train/static/dynamic must keep training's
    uninterrupted-on-goal semantics (goal_reached_and_resample keeps sampling
    new routes for the full episode) so path_progress_mean measures sustained
    multi-route exposure. Use --terminate_on_final_goal only for a scenario
    that is meant to measure single-route success, e.g. maze_success.
    """
    if not args_cli.terminate_on_final_goal:
        return
    terminations_cfg = getattr(env_cfg, "terminations", None)
    if terminations_cfg is None:
        raise ValueError("--terminate_on_final_goal requires a task with a terminations manager.")
    terminations_cfg.structured_goal_reached = DoneTerm(
        func=go2w_mdp.navigation_path_final_goal_reached,
        params={"position_threshold": NAV_GOAL_SUCCESS_POSITION_THRESHOLD},
    )
    print(
        "[INFO] Hospital maze eval: episode terminates on first route completion "
        f"(position_threshold={NAV_GOAL_SUCCESS_POSITION_THRESHOLD:.2f} m)."
    )


def _override_navigation_play_case(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    args_cli: argparse.Namespace,
    env_seed: int,
):
    """Override start/goal/template sampling and layout seed for navigation play tasks."""
    events_cfg = getattr(env_cfg, "events", None)
    reset_obstacles = getattr(events_cfg, "reset_obstacles", None) if events_cfg is not None else None
    reset_base = getattr(events_cfg, "reset_base", None) if events_cfg is not None else None
    reset_params = getattr(reset_obstacles, "params", None)

    # --scenario takes precedence over legacy --nav_case.
    effective_scenario = args_cli.scenario if args_cli.scenario is not None else None
    if effective_scenario is None and args_cli.nav_case != "random":
        effective_scenario = args_cli.nav_case

    has_nav_override = (
        effective_scenario is not None
        or args_cli.fixed_layout
        or args_cli.nav_goal_forward is not None
        or args_cli.nav_goal_lateral is not None
        or args_cli.nav_goal_heading_jitter is not None
        or args_cli.nav_min_inter_obstacle_dist is not None
        or args_cli.nav_start_exclusion_radius is not None
        or args_cli.nav_goal_exclusion_radius is not None
        or args_cli.nav_head_on_progress_range is not None
        or args_cli.nav_fixed_start
        or args_cli.nav_start_x is not None
        or args_cli.nav_start_y is not None
        or args_cli.nav_start_yaw is not None
    )
    if not has_nav_override:
        return
    if reset_params is None or "fixed_scenario_template" not in reset_params:
        raise ValueError(
            "Navigation play overrides require a navigation play task "
            "(Nav-ObstacleFlat-Teacher-Go2w-Play-v0 or similar)."
        )

    if effective_scenario is not None:
        reset_params["fixed_scenario_template"] = effective_scenario
    if args_cli.fixed_layout:
        reset_params["fixed_layout_seed"] = env_seed
        print(f"[INFO] Fixed layout enabled: every episode reset uses seed={env_seed}")
    if args_cli.nav_goal_forward is not None:
        reset_params["fixed_goal_forward"] = args_cli.nav_goal_forward
        # Goal resampling after success uses the range params captured in
        # _nav_resample_on_goal, so pin the range too for repeated fixed tests.
        reset_params["goal_forward_range"] = (args_cli.nav_goal_forward, args_cli.nav_goal_forward)
    if args_cli.nav_goal_lateral is not None:
        reset_params["fixed_goal_lateral"] = args_cli.nav_goal_lateral
        reset_params["goal_lateral_range"] = (args_cli.nav_goal_lateral, args_cli.nav_goal_lateral)
    if args_cli.nav_goal_heading_jitter is not None:
        reset_params["fixed_goal_heading_jitter"] = args_cli.nav_goal_heading_jitter
        reset_params["goal_heading_jitter_range"] = (
            args_cli.nav_goal_heading_jitter,
            args_cli.nav_goal_heading_jitter,
        )
    if args_cli.nav_min_inter_obstacle_dist is not None:
        reset_params["min_inter_obstacle_dist"] = args_cli.nav_min_inter_obstacle_dist
    if args_cli.nav_start_exclusion_radius is not None:
        reset_params["start_exclusion_radius"] = args_cli.nav_start_exclusion_radius
    if args_cli.nav_goal_exclusion_radius is not None:
        if args_cli.nav_goal_exclusion_radius < 0.0:
            raise ValueError("--nav_goal_exclusion_radius must be non-negative.")
        reset_params["goal_exclusion_radius"] = args_cli.nav_goal_exclusion_radius
    if args_cli.nav_head_on_progress_range is not None:
        progress_min, progress_max = args_cli.nav_head_on_progress_range
        if not 0.0 <= progress_min <= progress_max <= 1.0:
            raise ValueError("--nav_head_on_progress_range requires 0.0 <= MIN <= MAX <= 1.0.")
        reset_params["head_on_progress_range"] = (progress_min, progress_max)

    if (
        args_cli.nav_fixed_start
        or args_cli.nav_start_x is not None
        or args_cli.nav_start_y is not None
        or args_cli.nav_start_yaw is not None
    ):
        if reset_base is None:
            raise ValueError("Navigation fixed-start play requires a reset_base event.")
        pose_range = reset_base.params.setdefault("pose_range", {})
        velocity_range = reset_base.params.setdefault("velocity_range", {})
        start_x = 0.0 if args_cli.nav_start_x is None else args_cli.nav_start_x
        start_y = 0.0 if args_cli.nav_start_y is None else args_cli.nav_start_y
        start_yaw = 0.0 if args_cli.nav_start_yaw is None else args_cli.nav_start_yaw
        pose_range["x"] = (start_x, start_x)
        pose_range["y"] = (start_y, start_y)
        pose_range["yaw"] = (start_yaw, start_yaw)
        for key in ("x", "y", "z", "roll", "pitch", "yaw"):
            velocity_range[key] = (0.0, 0.0)
        print(f"[INFO] Navigation fixed start: x={start_x:.2f}, y={start_y:.2f}, yaw={start_yaw:.2f}")

    print(
        "[INFO] Navigation play overrides: "
        f"scenario={reset_params.get('fixed_scenario_template') or 'random'}, "
        f"fixed_layout={args_cli.fixed_layout}, "
        f"goal_forward={reset_params.get('fixed_goal_forward')}, "
        f"goal_lateral={reset_params.get('fixed_goal_lateral')}, "
        f"goal_heading_jitter={reset_params.get('fixed_goal_heading_jitter')}, "
        f"min_inter_obstacle_dist={reset_params.get('min_inter_obstacle_dist')}, "
        f"start_exclusion_radius={reset_params.get('start_exclusion_radius')}, "
        f"goal_exclusion_radius={reset_params.get('goal_exclusion_radius')}"
    )


def _init_spl_tracking(
    base_env, termination_names: list[str]
) -> tuple[bool, torch.Tensor, torch.Tensor, int | None]:
    """Initialize SPL tracking state for structured navigation validation."""
    spl_enabled = hasattr(base_env, "_go2w_navigation_path_s") and hasattr(
        base_env, "_go2w_navigation_path_count"
    )
    nav_optimal_len = torch.zeros(base_env.num_envs)
    nav_actual_len = torch.zeros(base_env.num_envs)
    goal_reached_term_idx: int | None = (
        termination_names.index("structured_goal_reached")
        if "structured_goal_reached" in termination_names
        else None
    )
    if spl_enabled:
        for ei in range(base_env.num_envs):
            pc = int(base_env._go2w_navigation_path_count[ei].item())
            if pc > 0:
                nav_optimal_len[ei] = float(base_env._go2w_navigation_path_s[ei, pc - 1].item())
    return spl_enabled, nav_optimal_len, nav_actual_len, goal_reached_term_idx


def _init_nav_markers(
    base_env, args_cli: argparse.Namespace
) -> tuple:
    """Create navigation visualization markers; returns None markers for non-nav tasks."""
    nav_goal_marker = None
    nav_final_goal_marker = None
    nav_start_marker = None
    nav_path_marker = None
    has_goal_markers = hasattr(base_env, "_go2w_goal_pos_w") and hasattr(base_env, "_go2w_start_pos_w")
    has_astar_markers = (
        not args_cli.no_astar
        and hasattr(base_env, "_go2w_navigation_path_w")
        and hasattr(base_env, "_go2w_navigation_path_count")
    )
    if has_goal_markers:
        nav_goal_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/NavGoalMarker",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.30,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.9, 0.1)),
                    ),
                },
            )
        )
        nav_start_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/NavStartMarker",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.20,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.5, 1.0)),
                    ),
                },
            )
        )
        if has_astar_markers:
            nav_final_goal_marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/NavFinalGoalMarker",
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=0.22,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.05, 0.02)),
                        ),
                    },
                )
            )
            nav_path_marker = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/NavPathMarker",
                    markers={
                        "line": sim_utils.CylinderCfg(
                            radius=0.025,
                            height=1.0,
                            visual_material=sim_utils.PreviewSurfaceCfg(
                                diffuse_color=(1.0, 0.9, 0.0),
                                roughness=1.0,
                            ),
                        ),
                    },
                )
            )
    return nav_goal_marker, nav_final_goal_marker, nav_start_marker, nav_path_marker, has_goal_markers


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    # Resize terrain generator grid so tiles match num_envs exactly (same logic as train.py).
    _scene_terrain = getattr(env_cfg.scene, "terrain", None)
    if _scene_terrain is not None and getattr(_scene_terrain, "use_terrain_origins", False):
        _tg = getattr(_scene_terrain, "terrain_generator", None)
        if _tg is not None:
            import math as _m
            _n = env_cfg.scene.num_envs
            _r = _m.isqrt(_n)
            while _r > 1 and _n % _r != 0:
                _r -= 1
            _tg.num_rows = max(1, _r)
            _tg.num_cols = max(1, _m.ceil(_n / _r))
            del _m, _n, _r, _tg
    del _scene_terrain
    env_cfg.sim.use_fabric = not args_cli.disable_fabric
    _override_play_episode_length(env_cfg, args_cli)
    _override_hospital_maze_route_steps(env_cfg, args_cli)
    _override_terminate_on_final_goal(env_cfg, args_cli)
    override_play_obstacle_count(env_cfg, args_cli.num_obstacles)
    override_play_command_path_spawn(
        env_cfg,
        args_cli.command_path_obstacles,
        args_cli.command_path_forward_range,
        args_cli.command_path_lateral_range,
        args_cli.command_path_min_speed,
    )

    # Resolve seed early so _override_navigation_play_case can inject fixed_layout_seed.
    # handle_deprecated_rsl_rl_cfg does not affect the seed path, so order is safe.
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_seed = resolve_play_seed(args_cli, agent_cfg.seed)
    agent_cfg.seed = env_seed
    env_cfg.seed = env_seed
    print(f"[INFO] Play seed: {env_seed}")

    _override_navigation_play_case(env_cfg, args_cli, env_seed)
    _override_random_navigation_play_obstacle_shapes(env_cfg, args_cli, env_seed)
    _override_structured_navigation_play(env_cfg, args_cli)
    _override_dynamic_navigation_play(env_cfg, args_cli)
    _nav_play_overrides = _override_navigation_play_goal_command_params(env_cfg, args_cli)
    if args_cli.no_astar and args_cli.structured_env == "none":
        raise ValueError("--no_astar requires --structured_env to be set.")
    if args_cli.no_astar:
        print("[INFO] Ablation mode: A* path following disabled — navigating directly to final corridor goal.")
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = None
    if args_cli.teacher_steering:
        log_dir = os.path.join(log_root_path, "teacher_play")
    elif args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
        log_dir = os.path.dirname(resume_path)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        log_dir = os.path.dirname(resume_path)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir
    configure_frozen_llc_action(env_cfg, args_cli.locomotion_checkpoint, args_cli.task)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    teacher = None
    policy_nn = None
    if args_cli.teacher_steering:
        teacher = build_teacher_policy(
            env, obs, agent_cfg, env.unwrapped.device, args_cli.locomotion_checkpoint
        )
        print("[INFO] Running direct teacher steering: geometric navigation teacher + frozen LLC")
    else:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        if agent_cfg.class_name == "DistillationRunner":
            # Inference only uses the student policy.  Some play/eval envs expose a
            # teacher obs shape that differs from the distillation checkpoint.
            runner.load(
                resume_path,
                load_cfg={"student": True, "teacher": False, "optimizer": False, "iteration": True},
            )
        else:
            runner.load(resume_path)

        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # export the trained policy to JIT and ONNX formats
        if args_cli.disable_export or agent_cfg.class_name == "DistillationRunner":
            print("[INFO] Skipping policy export before play/eval.")
        else:
            export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

            if version.parse(installed_version) >= version.parse("4.0.0"):
                # use the new export functions for rsl-rl >= 4.0.0
                runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
                runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
            else:
                # extract the neural network for rsl-rl < 4.0.0
                if version.parse(installed_version) >= version.parse("2.3.0"):
                    policy_nn = runner.alg.policy
                else:
                    policy_nn = runner.alg.actor_critic

                # extract the normalizer
                if hasattr(policy_nn, "actor_obs_normalizer"):
                    normalizer = policy_nn.actor_obs_normalizer
                elif hasattr(policy_nn, "student_obs_normalizer"):
                    normalizer = policy_nn.student_obs_normalizer
                else:
                    normalizer = None

                # export to JIT and ONNX
                export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
                export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    timestep = 0
    step_count = 0
    last_hlc_cmd: torch.Tensor | None = None
    episode_collision_counts: dict[int, int] = defaultdict(int)
    episode_lengths = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device, dtype=torch.long)
    completed_episodes = 0
    total_episode_length = 0.0
    termination_manager = getattr(env.unwrapped, "termination_manager", None)
    termination_names = list(termination_manager.active_terms) if termination_manager is not None else []
    termination_counts: dict[str, int] = defaultdict(int)
    _obstacle_contact_term_idx: int | None = (
        termination_names.index("obstacle_contact") if "obstacle_contact" in termination_names else None
    )
    multi_term_episodes = 0
    _stuck_termination_count = 0

    # Navigation visualization markers — only created for nav tasks.
    _base_env = env.unwrapped
    preflight_check_obstacle_slots(_base_env)
    try:
        _reset_obstacles_cfg = _base_env.event_manager.get_term_cfg("reset_obstacles")
        _fixed_layout_pairing_supported = "fixed_layout_seed" in (_reset_obstacles_cfg.params or {})
    except (ValueError, KeyError, AttributeError):
        _fixed_layout_pairing_supported = False

    # SPL = (1/N) Σ sᵢ × lᵢ / max(pᵢ, lᵢ) — initialized from first A* path lengths.
    _spl_enabled, _nav_optimal_len, _nav_actual_len, _goal_reached_term_idx = (
        _init_spl_tracking(_base_env, termination_names)
    )
    _nav_prev_robot_pos: torch.Tensor | None = None
    # Pre-allocated (not lazily assigned inside the loop) so this tensor's
    # identity is fixed before inference_mode is ever entered — see the
    # in-place-mutation note above _update_contact_events for why a rebind
    # here would poison it as an "inference tensor" and crash the first
    # in-place write to it from the done-envs reset block.
    _stuck_last_pos = torch.zeros(env.unwrapped.num_envs, 2)
    _stuck_last_pos_initialized = False
    _stuck_counter = torch.zeros(env.unwrapped.num_envs, dtype=torch.long)
    # A stuck reset becomes a time_out on the following step. Keep the
    # attribution per environment until that termination is observed.
    _stuck_forced_pending = torch.zeros(env.unwrapped.num_envs, dtype=torch.bool)
    _nav_spl_history: list[float] = []
    _goals_per_episode_history: list[float] = []
    _path_progress_history: list[float] = []
    # Per-episode artifact for analysis beyond aggregate manifests.
    _episode_metric_records: list[dict[str, object]] = []

    # --- Onset-based collision tracking (eval only, requires ContactSensor in scene) ---
    # obstacle_contacts sensor: robot↔obstacle contacts (obstacles float 5cm above ground,
    # so net_forces only reflects robot hits). Low slots use the reset-time metadata flag,
    # not a second center-height heuristic.
    # contact_forces sensor on Robot/base: catches wall hits (base never touches ground normally).
    _OBSTACLE_CONTACT_ON      = 1.0   # N — event starts above this (training reward threshold)
    _OBSTACLE_CONTACT_OFF     = 0.5   # N — event ends below this (hysteresis suppresses chatter)
    _WALL_CONTACT_ON          = 0.5   # N — lower than termination (1.0 N) to catch soft grazes
    _WALL_CONTACT_OFF         = 0.25  # N
    _CONTACT_REFRACTORY_STEPS = max(1, round(0.5 / dt))  # merge re-contacts within 0.5 s
    _obs_sensor     = _base_env.scene.sensors.get("obstacle_contacts")
    # Prefer full robot-body sensor (eval scenes); fall back to base-only contact_forces.
    _base_sensor    = (
        _base_env.scene.sensors.get("robot_body_contacts")
        or _base_env.scene.sensors.get("contact_forces")
    )
    _num_envs_val   = _base_env.num_envs
    _dev             = _base_env.device

    # Read the reset-time top-z classification used by the reward code.  It is
    # shaped [env, slot], so per-env layouts remain correct too.
    _low_obs_mask: torch.Tensor | None = None
    if _obs_sensor is not None:
        _num_obs_slots = _obs_sensor.data.net_forces_w_history.shape[2]
        _low_flags = getattr(_base_env, "_go2w_obstacle_low_flag", None)
        if _low_flags is not None and _low_flags.shape[-1] >= _num_obs_slots:
            _low_obs_mask = _low_flags[:, :_num_obs_slots].to(device=_dev, dtype=torch.bool)
        else:
            _low_obs_mask = torch.zeros(_num_envs_val, _num_obs_slots, dtype=torch.bool, device=_dev)

    # Mask for wall sensor: exclude wheel bodies (*_foot) to avoid ground contact false positives.
    _wall_body_non_wheel_mask: torch.Tensor | None = None
    if _base_sensor is not None:
        _wall_body_non_wheel_mask = torch.tensor(
            [not n.endswith("_foot") for n in _base_sensor.body_names],
            dtype=torch.bool,
            device=_dev,
        )

    _obs_slot_count    = _obs_sensor.data.net_forces_w_history.shape[2] if _obs_sensor is not None else 1
    _obs_contact_prev  = torch.zeros(_num_envs_val, _obs_slot_count, dtype=torch.bool, device=_dev) if _obs_sensor is not None else None
    _base_contact_prev = torch.zeros(_num_envs_val, dtype=torch.bool, device=_dev)
    _obs_coll_ep       = torch.zeros(_num_envs_val, dtype=torch.long, device=_dev)
    _low_obs_coll_ep   = torch.zeros(_num_envs_val, dtype=torch.long, device=_dev)
    _wall_coll_ep      = torch.zeros(_num_envs_val, dtype=torch.long, device=_dev)
    # Per-event severity tracking: refractory merges micro-bounces into one event,
    # per-event peak force enables post-hoc graze/collision classification
    # (all events are dumped to contact_events.csv next to the manifest).
    _obs_refract       = torch.zeros(_num_envs_val, _obs_slot_count, dtype=torch.long, device=_dev)
    _obs_event_peak    = torch.zeros(_num_envs_val, _obs_slot_count, device=_dev)
    _obs_event_is_low  = torch.zeros(_num_envs_val, _obs_slot_count, dtype=torch.bool, device=_dev)
    _wall_refract      = torch.zeros(_num_envs_val, dtype=torch.long, device=_dev)
    _wall_event_peak   = torch.zeros(_num_envs_val, device=_dev)
    _episode_index = torch.zeros(_num_envs_val, dtype=torch.long, device=_dev)

    # Evaluation cohort scheduler.  ManagerBasedRLEnv auto-resets an env as
    # soon as it terminates, so stopping on the *global* first N completions
    # silently dropped the still-running (usually longer-lived) trajectories.
    # Instead, admit exactly N episode instances, ignore any automatic reset
    # beyond that cohort, and stop only after every admitted instance has a
    # terminal row.  This keeps the normal per-episode timeout semantics while
    # making the collected sample independent of completion order.
    _eval_requested = max(int(args_cli.nav_eval_episodes), 0)
    _eval_cohort_enabled = _eval_requested > 0
    _eval_active = torch.zeros(_num_envs_val, dtype=torch.bool, device=_dev)
    _eval_target_episode_index = torch.full(
        (_num_envs_val,), -1, dtype=torch.long, device=_dev
    )
    _eval_started = min(_eval_requested, _num_envs_val)
    if _eval_cohort_enabled and _eval_started > 0:
        _eval_active[:_eval_started] = True
        _eval_target_episode_index[:_eval_started] = 0
        print(
            "[PLAY-EVAL] Cohort protocol: "
            f"{_eval_requested} trajectories; {_eval_started} admitted initially; "
            "each admitted trajectory is recorded at its own terminal event."
        )

    def _is_eval_sample(env_id: int) -> bool:
        """Whether the current episode of ``env_id`` belongs to the eval cohort."""
        if not _eval_cohort_enabled:
            return True
        return bool(
            _eval_active[env_id].item()
            and _episode_index[env_id].item() == _eval_target_episode_index[env_id].item()
        )

    _contact_event_records: list[tuple[str, int, int, int, float, int]] = []
    _obs_coll_history:      list[float] = []
    _low_obs_coll_history:  list[float] = []
    _wall_coll_history:     list[float] = []
    # Cumulative navigation distance per env: sum of completed path lengths within the episode.
    _cumul_path_progress = torch.zeros(_num_envs_val, device=_dev)

    def _update_contact_events() -> None:
        """Consume the current (pre-step) contact-sensor sample.

        ManagerBasedRLEnv resets completed environments inside ``env.step``.
        Sampling here prevents a terminal reset from being attributed to the
        newly started episode.  A terminal obstacle termination remains in the
        separate termination metric even if it occurs between sensor samples.
        """
        nonlocal _low_obs_mask, _obs_contact_prev, _base_contact_prev
        nonlocal _obs_refract, _obs_event_peak, _obs_event_is_low
        nonlocal _wall_refract, _wall_event_peak, _obs_coll_ep, _low_obs_coll_ep, _wall_coll_ep
        if _obs_sensor is None:
            return
        _low_flags = getattr(_base_env, "_go2w_obstacle_low_flag", None)
        if _low_flags is not None and _low_flags.shape[-1] >= _num_obs_slots:
            _low_obs_mask = _low_flags[:, :_num_obs_slots].to(device=_dev, dtype=torch.bool)
        # All cross-step state tensors below (_obs_refract, _obs_event_peak,
        # _obs_contact_prev, _wall_refract, _wall_event_peak, _base_contact_prev)
        # must be mutated in place (.copy_/.sub_/.clamp_/indexed write) rather
        # than rebound with `=` to a freshly computed tensor. A rebind inside
        # this inference_mode step would replace the pre-loop tensor object
        # with a new one tagged as an "inference tensor"; a later in-place
        # write to it from the done-envs reset block below then raises
        # "Inplace update to inference tensor outside InferenceMode".
        _obs_forces = _obs_sensor.data.net_forces_w_history[:, 0]
        _obs_force_mag = _obs_forces.norm(dim=-1)
        _obs_refract.sub_(1).clamp_(min=0)
        _obs_contact_now = (
            (_obs_contact_prev & (_obs_force_mag > _OBSTACLE_CONTACT_OFF))
            | (_obs_force_mag > _OBSTACLE_CONTACT_ON)
        )
        _obs_onset = _obs_contact_now & ~_obs_contact_prev & (_obs_refract == 0)
        _obs_coll_ep += _obs_onset.long().sum(dim=-1)
        if _low_obs_mask is not None:
            _low_obs_coll_ep += (_obs_onset & _low_obs_mask).long().sum(dim=-1)
            _obs_event_is_low[_obs_onset] = _low_obs_mask[_obs_onset]
        _obs_offset = ~_obs_contact_now & _obs_contact_prev
        _obs_refract[_obs_offset] = _CONTACT_REFRACTORY_STEPS
        _obs_tracking = _obs_contact_now | (_obs_refract > 0)
        _obs_event_peak[_obs_tracking] = torch.maximum(_obs_event_peak, _obs_force_mag)[_obs_tracking]
        _obs_flush = ~_obs_tracking & (_obs_event_peak > 0.0)
        if _obs_flush.any():
            for _e, _s in _obs_flush.nonzero(as_tuple=False).tolist():
                if _is_eval_sample(_e):
                    _contact_event_records.append((
                        "obstacle", _e, int(_episode_index[_e].item()), step_count,
                        float(_obs_event_peak[_e, _s].item()), int(_obs_event_is_low[_e, _s].item()),
                    ))
            _obs_event_peak[_obs_flush] = 0.0
            _obs_event_is_low[_obs_flush] = False
        _obs_contact_prev.copy_(_obs_contact_now)

        if _base_sensor is None:
            return
        _base_forces = _base_sensor.data.net_forces_w_history[:, 0]
        if _wall_body_non_wheel_mask is not None:
            _base_forces = _base_forces[:, _wall_body_non_wheel_mask, :]
        _base_force_mag = _base_forces.norm(dim=-1).max(dim=-1).values
        _base_contact_now = (
            (_base_contact_prev & (_base_force_mag > _WALL_CONTACT_OFF))
            | (_base_force_mag > _WALL_CONTACT_ON)
        )
        _wall_refract.sub_(1).clamp_(min=0)
        _wall_onset = _base_contact_now & ~_base_contact_prev & (_wall_refract == 0) & ~_obs_contact_now.any(dim=-1)
        _wall_coll_ep += _wall_onset.long()
        _wall_offset = ~_base_contact_now & _base_contact_prev
        _wall_refract[_wall_offset] = _CONTACT_REFRACTORY_STEPS
        _wall_tracking = _base_contact_now | (_wall_refract > 0)
        _wall_event_peak[_wall_tracking] = torch.maximum(_wall_event_peak, _base_force_mag)[_wall_tracking]
        _wall_flush = ~_wall_tracking & (_wall_event_peak > 0.0)
        if _wall_flush.any():
            for _e in _wall_flush.nonzero(as_tuple=False).squeeze(-1).tolist():
                if _is_eval_sample(_e):
                    _contact_event_records.append((
                        "wall", _e, int(_episode_index[_e].item()), step_count,
                        float(_wall_event_peak[_e].item()), 0,
                    ))
            _wall_event_peak[_wall_flush] = 0.0
        _base_contact_prev.copy_(_base_contact_now)

    (
        _nav_goal_marker, _nav_final_goal_marker, _nav_start_marker, _nav_path_marker,
        _nav_has_goal_markers,
    ) = _init_nav_markers(_base_env, args_cli)

    # play output directory setup
    _out_dir: str | None = None
    _out_log_file = None
    if args_cli.play_name:
        _out_dir = os.path.abspath(os.path.join("logs", "nav_play", args_cli.play_name))
        os.makedirs(_out_dir, exist_ok=True)
        _out_log_file = open(os.path.join(_out_dir, "nav_debug.log"), "w", buffering=1)
        sys.stdout = TeeStream(sys.__stdout__, _out_log_file)
        os.environ.setdefault("GO2W_NAV_DEBUG", "1")
        print(f"[INFO] Play output dir: {_out_dir}")

    print(f"[INFO] Nav command play values: {_nav_play_overrides}")

    _obstacle_label_logging = (
        args_cli.nav_live_obstacle_labels
        or "Hospital" in str(args_cli.task)
    )
    if _obstacle_label_logging:
        print("[INFO] Obstacle label logging enabled for top-down debug plots.")

    _live_label_drawer = None
    if args_cli.nav_live_obstacle_labels:
        if getattr(args_cli, "headless", False):
            print("[WARN] --nav_live_obstacle_labels requested in headless mode; viewport labels may not be visible.")
        _live_label_drawer = _LiveObstacleLabelDrawer(
            scale=NAV_LIVE_LABEL_SCALE,
            max_labels=NAV_LIVE_LABEL_MAX,
        )
        if _live_label_drawer.enabled:
            print(
                "[INFO] Live obstacle labels enabled: "
                f"interval={NAV_LIVE_LABEL_INTERVAL}, "
                f"scale={NAV_LIVE_LABEL_SCALE:.3f}, max={NAV_LIVE_LABEL_MAX}"
            )

    # Auto-enable depth video when --play_name is set and the env has a depth camera.
    _has_depth_camera = (
        hasattr(_base_env.scene, "sensors") and "depth_camera" in _base_env.scene.sensors
    )
    if args_cli.play_name and _has_depth_camera and not args_cli.depth_video:
        args_cli.depth_video = True
        print("[INFO] --play_name: depth camera found — auto-enabling depth video recording.")

    # depth video writer setup
    _depth_video_frames: list = []
    if args_cli.depth_video:
        if not _has_depth_camera:
            print("[WARN] --depth_video: no 'depth_camera' sensor found; depth video disabled.")
            args_cli.depth_video = False
        else:
            print(
                f"[INFO] Depth video: recording env={DEPTH_VIDEO_ENV}, "
                f"scale={DEPTH_VIDEO_SCALE}x -> "
                f"{128 * DEPTH_VIDEO_SCALE}x{72 * DEPTH_VIDEO_SCALE} MP4"
            )

    # simulate environment
    if args_cli.nav_contact_debug and args_cli.nav_contact_debug_steps <= 0:
        raise ValueError("--nav_contact_debug_steps must be positive when --nav_contact_debug is enabled.")
    import signal as _signal
    _play_interrupted = False

    def _handle_sigint(sig, frame):
        nonlocal _play_interrupted
        _play_interrupted = True
        print("\n[INFO] Interrupt received — finishing current step and saving outputs ...")

    _signal.signal(_signal.SIGINT, _handle_sigint)

    _run_start_time = time.time()
    _run_started_at = datetime.now(timezone.utc).isoformat()
    while simulation_app.is_running() and not _play_interrupted:
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # Read the previous physics step before env.step() can reset a
            # just-finished environment and contaminate the next episode.
            _update_contact_events()
            if args_cli.structured_env != "none":
                if not args_cli.no_astar:
                    go2w_mdp.update_navigation_path_waypoint(
                        _base_env,
                        lookahead_distance=args_cli.astar_lookahead_distance,
                        waypoint_reach_radius=args_cli.astar_waypoint_reach_radius,
                        adaptive_lookahead=not args_cli.no_adaptive_lookahead,
                        lookahead_min=args_cli.lookahead_min,
                        curvature_scan_horizon=args_cli.curvature_scan_horizon,
                        curvature_threshold=args_cli.curvature_threshold,
                    )
                else:
                    _ablation_apply_direct_goal(_base_env)
                obs = env.get_observations()
            # agent stepping
            actions = teacher(obs) if teacher is not None else policy(obs)
            # Capture state before env.step() so episode-end and goal-reached events can be detected.
            _pre_step_path_progress = (
                _base_env._go2w_navigation_path_progress_s.clone()
                if hasattr(_base_env, "_go2w_navigation_path_progress_s") else None
            )
            _pre_step_goals_reached = (
                _base_env._go2w_goals_reached_episode.clone()
                if hasattr(_base_env, "_go2w_goals_reached_episode") else None
            )
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # Accumulate completed path length when a goal is reached mid-episode (non-done envs only).
            if _pre_step_goals_reached is not None and _pre_step_path_progress is not None:
                _post_goals = _base_env._go2w_goals_reached_episode
                _goal_hit = (_post_goals > _pre_step_goals_reached) & ~dones.to(_dev).bool()
                _cumul_path_progress[_goal_hit] += _pre_step_path_progress[_goal_hit]
            # reset recurrent states for episodes that have terminated
            if teacher is not None:
                teacher.reset(dones.bool())
            elif version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

        # Collect depth camera frame for video recording.
        _depth_video_limit = args_cli.depth_video_steps
        if args_cli.depth_video and (_depth_video_limit == 0 or len(_depth_video_frames) < _depth_video_limit):
            _depth_sensor = _base_env.scene.sensors["depth_camera"]
            _depth_raw = _depth_sensor.data.output["distance_to_image_plane"]
            if _depth_raw.ndim == 4:
                _depth_raw = _depth_raw.squeeze(-1)
            _depth_frame = _depth_raw[DEPTH_VIDEO_ENV].float().cpu().numpy()
            _closeness = (1.0 - _depth_frame / 6.0).clip(0.0, 1.0)
            _viridis = _cmaps["viridis"]
            _colored = (_viridis(_closeness)[:, :, :3] * 255).astype(np.uint8)
            _scale = DEPTH_VIDEO_SCALE
            _scaled = np.repeat(np.repeat(_colored, _scale, axis=0), _scale, axis=1)
            _depth_video_frames.append(_scaled)

        # SPL: accumulate per-env robot displacement, skipping the reset step for
        # done envs (env.step teleports the robot; that jump must not be counted).
        if _spl_enabled:
            cur_robot_pos = _base_env.scene["robot"].data.root_pos_w[:, :2].cpu()
            if _nav_prev_robot_pos is not None:
                step_disp = (cur_robot_pos - _nav_prev_robot_pos).norm(dim=-1)
                done_cpu = dones.cpu().bool()
                _nav_actual_len += torch.where(done_cpu, torch.zeros_like(step_disp), step_disp)
            _nav_prev_robot_pos = cur_robot_pos

        # Stuck detection: force-reset envs that haven't moved enough in N steps.
        if args_cli.stuck_timeout_steps > 0:
            _cur_stuck_pos = _base_env.scene["robot"].data.root_pos_w[:, :2].cpu()
            if not _stuck_last_pos_initialized:
                _stuck_last_pos.copy_(_cur_stuck_pos)
                _stuck_last_pos_initialized = True
            else:
                _disp = (_cur_stuck_pos - _stuck_last_pos).norm(dim=-1)
                _moved = _disp >= args_cli.stuck_threshold
                _stuck_last_pos[_moved] = _cur_stuck_pos[_moved]
                _stuck_counter[_moved] = 0
                _stuck_counter[~_moved & ~dones.cpu().bool()] += 1
                _stuck_mask = _stuck_counter >= args_cli.stuck_timeout_steps
                if _stuck_mask.any():
                    _stuck_ids = _stuck_mask.nonzero(as_tuple=False).squeeze(-1).to(_base_env.device)
                    _base_env.episode_length_buf[_stuck_ids] = _base_env.max_episode_length - 1
                    _stuck_forced_pending[_stuck_mask.cpu()] = True
                    _stuck_counter[_stuck_mask] = 0
                    _stuck_last_pos[_stuck_mask] = _cur_stuck_pos[_stuck_mask]

        # Track last HLC command (policy output = 3D velocity) for logging.
        last_hlc_cmd = actions.detach()

        # Update goal/start markers for nav tasks.
        if _nav_goal_marker is not None:
            goal_pos = _base_env._go2w_goal_pos_w.clone()
            goal_pos[:, 2] += 0.35
            _nav_goal_marker.visualize(translations=goal_pos)
            if _nav_final_goal_marker is not None:
                final_idx = (_base_env._go2w_navigation_path_count - 1).clamp(min=0)
                row_idx = torch.arange(_base_env.num_envs, device=_base_env._go2w_navigation_path_w.device)
                final_goal_pos = _base_env._go2w_navigation_path_w[row_idx, final_idx, :3].clone()
                final_goal_pos[:, 2] += 0.45
                _nav_final_goal_marker.visualize(translations=final_goal_pos)
            start_pos = _base_env._go2w_start_pos_w.clone()
            start_pos[:, 2] += 0.35
            if _nav_start_marker is not None:
                _nav_start_marker.visualize(translations=start_pos)
            if _nav_path_marker is not None:
                path_markers = _get_nav_path_line_markers(_base_env, args_cli.nav_log_env)
                if path_markers is not None:
                    positions, orientations, scales = path_markers
                    _nav_path_marker.visualize(
                        translations=positions,
                        orientations=orientations,
                        scales=scales,
                    )
        if _live_label_drawer is not None and step_count % NAV_LIVE_LABEL_INTERVAL == 0:
            _live_label_drawer.update(_base_env, args_cli.nav_log_env)

        episode_lengths += 1
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        num_done = int(done_ids.numel())
        if num_done > 0:
            _tracked_done = [_is_eval_sample(int(env_id.item())) for env_id in done_ids]
            _tracked_done_count = sum(_tracked_done)
            if args_cli.stuck_timeout_steps > 0 and _stuck_last_pos_initialized:
                _done_cpu = done_ids.cpu()
                _stuck_counter[_done_cpu] = 0
                _stuck_last_pos[_done_cpu] = _base_env.scene["robot"].data.root_pos_w[:, :2].cpu()[_done_cpu]
            if _tracked_done_count:
                total_episode_length += sum(
                    float(episode_lengths[env_id].item())
                    for env_id, tracked in zip(done_ids, _tracked_done, strict=True) if tracked
                )
            done_terms = (
                termination_manager._last_episode_dones[done_ids] if termination_manager is not None else None
            )
            if done_terms is not None and done_terms.numel() > 0:
                for i, tracked in enumerate(_tracked_done):
                    if not tracked:
                        continue
                    multi_term_episodes += int(done_terms[i].sum().item() > 1)
                    for idx, term_name in enumerate(termination_names):
                        termination_counts[term_name] += int(done_terms[i, idx].item())
                # Track per-env obstacle collision terminations for episode log.
                if _obstacle_contact_term_idx is not None:
                    for i, env_id in enumerate(done_ids.tolist()):
                        if done_terms[i, _obstacle_contact_term_idx].item():
                            episode_collision_counts[env_id] += 1

            # Persist one pre-reset row per terminated episode.  The environment
            # has already reset its scene, so use the snapshots and per-episode
            # counters maintained above rather than reading reset buffers.
            _episode_progress_for_history: dict[int, float] = {}
            for i, env_id_tensor in enumerate(done_ids):
                _env_i = int(env_id_tensor.item())
                _stuck_episode = bool(_stuck_forced_pending[_env_i].item())
                if not _tracked_done[i]:
                    _stuck_forced_pending[_env_i] = False
                    continue
                _term_list: list[str] = []
                if done_terms is not None:
                    for _term_i, _term_name in enumerate(termination_names):
                        if bool(done_terms[i, _term_i].item()):
                            _term_list.append(
                                "stuck_timeout"
                                if _term_name == "time_out" and _stuck_episode
                                else _term_name
                            )
                _success = bool(
                    done_terms is not None
                    and _goal_reached_term_idx is not None
                    and bool(done_terms[i, _goal_reached_term_idx].item())
                )
                _goals = (
                    float(_pre_step_goals_reached[env_id_tensor].item()) + float(_success)
                    if _pre_step_goals_reached is not None else float(_success)
                )
                _progress = (
                    float((_cumul_path_progress[env_id_tensor] + _pre_step_path_progress[env_id_tensor]).item())
                    if _pre_step_path_progress is not None else None
                )
                _optimal = float(_nav_optimal_len[_env_i].item()) if _spl_enabled else None
                _final_progress = select_episode_progress(_progress, _optimal, _success)
                if _final_progress is not None:
                    _episode_progress_for_history[_env_i] = _final_progress
                _spl_value = None
                if _spl_enabled and _optimal is not None:
                    _actual = float(_nav_actual_len[_env_i].item())
                    _spl_value = float(_success) * _optimal / max(_actual, _optimal) if _actual > 0.0 and _optimal > 0.0 else 0.0
                _obs_events = int(_obs_coll_ep[_env_i].item()) if _obs_sensor is not None else None
                _low_obs_events = int(_low_obs_coll_ep[_env_i].item()) if _obs_sensor is not None else None
                _wall_events = int(_wall_coll_ep[_env_i].item()) if _base_sensor is not None else None
                _episode_metric_records.append({
                    "seed": args_cli.seed,
                    "env": _env_i,
                    "episode_index": int(_episode_index[_env_i].item()),
                    "termination": "|".join(_term_list) if _term_list else "unknown",
                    "success": int(_success),
                    "steps": int(episode_lengths[env_id_tensor].item()),
                    "duration_seconds": float(episode_lengths[env_id_tensor].item() * dt),
                    "goals_reached": _goals,
                    "path_progress_m": _progress,
                    "optimal_path_m": _optimal,
                    "spl": _spl_value,
                    "obstacle_contact_events": _obs_events,
                    "low_obstacle_contact_events": _low_obs_events,
                    "wall_contact_events": _wall_events,
                    "obstacle_contacts_per_path_progress_m": (
                        _obs_events / _progress if _obs_events is not None and _progress is not None and _progress > 0.0 else None
                    ),
                })
                if _stuck_episode:
                    _stuck_termination_count += 1
                _stuck_forced_pending[_env_i] = False

            # Print episode summary for the watched nav env before resetting counts.
            _print_nav_episode_log(
                _base_env, done_ids, args_cli.nav_log_env, last_hlc_cmd, episode_collision_counts
            )
            # Reset collision count for episodes that just ended.
            for env_id in done_ids.tolist():
                episode_collision_counts[env_id] = 0
            # Record goals reached per episode. The in-step reset has already
            # zeroed the env counter by the time env.step() returns, so read the
            # snapshot captured before the step instead.
            if _pre_step_goals_reached is not None:
                for i, env_id in enumerate(done_ids):
                    if not _tracked_done[i]:
                        continue
                    reached_final_goal = (
                        done_terms is not None
                        and _goal_reached_term_idx is not None
                        and bool(done_terms[i, _goal_reached_term_idx].item())
                    )
                    _goals_per_episode_history.append(
                        float(_pre_step_goals_reached[env_id].item()) + float(reached_final_goal)
                    )
            # Record contact-event counts (obstacle / low-obstacle / wall) per episode.
            if _obs_sensor is not None:
                for env_id, tracked in zip(done_ids, _tracked_done, strict=True):
                    if not tracked:
                        continue
                    _obs_coll_history.append(float(_obs_coll_ep[env_id].item()))
                    _low_obs_coll_history.append(float(_low_obs_coll_ep[env_id].item()))
                    _wall_coll_history.append(float(_wall_coll_ep[env_id].item()))
                # Flush events still live at episode end, then reset event state.
                for _e in done_ids.tolist():
                    if _is_eval_sample(_e):
                        for _s in (_obs_event_peak[_e] > 0.0).nonzero(as_tuple=False).squeeze(-1).tolist():
                            _contact_event_records.append((
                                "obstacle", _e, int(_episode_index[_e].item()), step_count,
                                float(_obs_event_peak[_e, _s].item()), int(_obs_event_is_low[_e, _s].item()),
                            ))
                        if float(_wall_event_peak[_e].item()) > 0.0:
                            _contact_event_records.append((
                                "wall", _e, int(_episode_index[_e].item()), step_count,
                                float(_wall_event_peak[_e].item()), 0,
                            ))
                _obs_coll_ep[done_ids]     = 0
                _low_obs_coll_ep[done_ids] = 0
                _wall_coll_ep[done_ids]    = 0
                _obs_contact_prev[done_ids]  = False
                _base_contact_prev[done_ids] = False
                _obs_refract[done_ids]     = 0
                _wall_refract[done_ids]    = 0
                _obs_event_peak[done_ids]  = 0.0
                _obs_event_is_low[done_ids] = False
                _wall_event_peak[done_ids] = 0.0
            # Record cumulative navigation distance: sum of all completed paths + final partial path.
            if _pre_step_path_progress is not None:
                for env_id, tracked in zip(done_ids, _tracked_done, strict=True):
                    if not tracked:
                        continue
                    _env_i = int(env_id.item())
                    _raw_progress = float(
                        (_cumul_path_progress[env_id] + _pre_step_path_progress[env_id]).item()
                    )
                    _path_progress_history.append(
                        _episode_progress_for_history.get(_env_i, _raw_progress)
                    )
                _cumul_path_progress[done_ids] = 0.0
            # Refresh the reward's reset-time low-obstacle flags after the reset.
            if _obs_sensor is not None:
                _low_flags = getattr(_base_env, "_go2w_obstacle_low_flag", None)
                if _low_flags is not None and _low_flags.shape[-1] >= _num_obs_slots:
                    _low_obs_mask = _low_flags[:, :_num_obs_slots].to(device=_dev, dtype=torch.bool)
            episode_lengths[done_ids] = 0
            _episode_index[done_ids] += 1
            completed_episodes += _tracked_done_count

            # Admit a replacement only for an episode that was in the cohort.
            # Envs reset automatically; episodes beyond the requested cohort
            # remain live in the simulator but are deliberately ignored.
            if _eval_cohort_enabled:
                for env_id, tracked in zip(done_ids.tolist(), _tracked_done, strict=True):
                    if not tracked:
                        continue
                    if _eval_started < _eval_requested:
                        _eval_active[env_id] = True
                        _eval_target_episode_index[env_id] = _episode_index[env_id]
                        _eval_started += 1
                    else:
                        _eval_active[env_id] = False
                        _eval_target_episode_index[env_id] = -1

            # Advance a fixed layout seed only for reset functions that expose
            # that contract. Hospital maze reset does not currently do so, thus
            # its policies share evaluation seed distributions but are not
            # episode-by-episode layout paired.
            if args_cli.seed_per_episode and args_cli.seed is not None:
                try:
                    _obs_term_cfg = _base_env.event_manager.get_term_cfg("reset_obstacles")
                    if "fixed_layout_seed" in (_obs_term_cfg.params or {}):
                        _obs_term_cfg.params["fixed_layout_seed"] = args_cli.seed + completed_episodes
                except (ValueError, KeyError, AttributeError):
                    pass

            # SPL: record contribution for each completed episode then reset trackers.
            if _spl_enabled and done_terms is not None and _goal_reached_term_idx is not None:
                for i, env_id_tensor in enumerate(done_ids):
                    if not _tracked_done[i]:
                        continue
                    ei = int(env_id_tensor.item())
                    success = bool(done_terms[i, _goal_reached_term_idx].item())
                    actual = float(_nav_actual_len[ei].item())
                    optimal = float(_nav_optimal_len[ei].item())
                    if actual > 0.0 and optimal > 0.0:
                        spl_val = float(success) * optimal / max(actual, optimal)
                    else:
                        spl_val = 0.0
                    _nav_spl_history.append(spl_val)
            if _spl_enabled and done_ids.numel() > 0:
                _nav_actual_len[done_ids.cpu()] = 0.0
                # Read optimal path length for newly started episodes (env already reset).
                for env_id_tensor in done_ids:
                    ei = int(env_id_tensor.item())
                    pc = int(_base_env._go2w_navigation_path_count[ei].item())
                    _nav_optimal_len[ei] = (
                        float(_base_env._go2w_navigation_path_s[ei, pc - 1].item()) if pc > 0 else 0.0
                    )

        if args_cli.nav_log_interval > 0 and step_count % args_cli.nav_log_interval == 0:
            _print_navigation_play_log(
                obs, dones, step_count, args_cli.nav_log_env, _base_env, last_hlc_cmd
            )
            if _obstacle_label_logging:
                _print_nav_obstacle_label_log(_base_env, step_count, args_cli.nav_log_env)
        if args_cli.nav_contact_debug:
            _print_nav_contact_debug(_base_env, step_count, args_cli.nav_log_env, last_hlc_cmd)
            if step_count + 1 >= args_cli.nav_contact_debug_steps:
                break
        if args_cli.nav_eval_episodes > 0 and (
            completed_episodes >= args_cli.nav_eval_episodes
            or (num_done > 0 and completed_episodes % max(args_cli.nav_eval_episodes // 4, 1) < num_done)
        ):
            averaged = {
                f"{term_name}_rate": termination_counts[term_name] / max(completed_episodes, 1)
                for term_name in termination_names
            }
            averaged["multi_term_fraction"] = multi_term_episodes / max(completed_episodes, 1)
            if _nav_spl_history:
                averaged["spl"] = sum(_nav_spl_history) / len(_nav_spl_history)
            avg_episode_length = total_episode_length / max(completed_episodes, 1)
            print("[PLAY-EVAL] " + format_eval_metrics(averaged, completed_episodes, avg_episode_length))
            if completed_episodes >= args_cli.nav_eval_episodes:
                break

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        step_count += 1

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Save depth video.
    if args_cli.depth_video and _depth_video_frames:
        import imageio as _imageio
        _save_dir = _out_dir if _out_dir else log_dir
        _depth_video_path = os.path.join(_save_dir, "depth_camera.mp4")
        _imageio.mimwrite(_depth_video_path, _depth_video_frames, fps=int(1.0 / dt), quality=8)
        print(f"[INFO] Depth video saved: {_depth_video_path}  ({len(_depth_video_frames)} frames)")

    # Write session manifest when --play_name is set.
    if _out_dir is not None:
        import json as _json
        _tc = dict(termination_counts) if termination_counts else {}
        if _stuck_termination_count > 0:
            # Stuck-forced resets terminate via time_out; keep the buckets disjoint.
            _tc["stuck_timeout"] = _stuck_termination_count
            _tc["time_out"] = max(_tc.get("time_out", 0) - _stuck_termination_count, 0)
        _spl_mean = float(sum(_nav_spl_history) / len(_nav_spl_history)) if _nav_spl_history else None
        _avg_ep_len = total_episode_length / max(completed_episodes, 1)
        _success_rate = (
            _tc.get("goal_reached", 0) + _tc.get("structured_goal_reached", 0)
        ) / max(completed_episodes, 1)
        _goals_per_ep_mean = (
            float(sum(_goals_per_episode_history) / len(_goals_per_episode_history))
            if _goals_per_episode_history else None
        )
        _path_progress_mean = (
            float(sum(_path_progress_history) / len(_path_progress_history))
            if _path_progress_history else None
        )
        _avg_obs_contacts = (
            float(sum(_obs_coll_history) / len(_obs_coll_history))
            if _obs_coll_history else None
        )
        _avg_low_obs_contacts = (
            float(sum(_low_obs_coll_history) / len(_low_obs_coll_history))
            if _low_obs_coll_history else None
        )
        _avg_wall_contacts = (
            float(sum(_wall_coll_history) / len(_wall_coll_history))
            if _wall_coll_history else None
        )
        _total_path_progress = float(sum(_path_progress_history))
        # This normalizes by route progress, not physical odometry.  It is an
        # exposure-to-task-progress metric and must not be presented as true
        # contacts per travelled meter.
        _obstacle_contacts_per_path_progress_meter = (
            float(sum(_obs_coll_history) / _total_path_progress)
            if _obs_coll_history and _total_path_progress > 0.0 else None
        )
        _run_finished_at = datetime.now(timezone.utc).isoformat()
        _manifest = {
            "task": args_cli.task,
            "checkpoint": getattr(args_cli, "checkpoint", None),
            "num_envs": args_cli.num_envs,
            "fixed_layout_pairing": _fixed_layout_pairing_supported,
            "completed_episodes": completed_episodes,
            "evaluation_protocol": (
                "completion_order_independent_cohort" if _eval_cohort_enabled else "continuous_play"
            ),
            "requested_episodes": _eval_requested if _eval_cohort_enabled else None,
            "started_episodes": _eval_started if _eval_cohort_enabled else None,
            "incomplete_cohort_episodes": (
                max(_eval_started - completed_episodes, 0) if _eval_cohort_enabled else None
            ),
            "steps": step_count,
            "started_at": _run_started_at,
            "finished_at": _run_finished_at,
            "wall_time_seconds": time.time() - _run_start_time,
            # Termination counts are intentionally multi-label (e.g. contact +
            # time_out can happen in the same final step); the completed count
            # is the only episode denominator.
            "total_episodes": completed_episodes,
            "termination_counts": _tc,
            "success_rate": _success_rate,
            "spl": _spl_mean,
            "avg_episode_length": _avg_ep_len,
            "goals_per_episode": _goals_per_ep_mean,
            "path_progress_mean": _path_progress_mean,
            "contact_event_definition": {
                "obstacle_on_force_n": _OBSTACLE_CONTACT_ON,
                "obstacle_off_force_n": _OBSTACLE_CONTACT_OFF,
                "wall_on_force_n": _WALL_CONTACT_ON,
                "wall_off_force_n": _WALL_CONTACT_OFF,
                "refractory_seconds": _CONTACT_REFRACTORY_STEPS * dt,
            },
            "avg_obstacle_contacts_per_ep": _avg_obs_contacts,
            "avg_low_obstacle_contacts_per_ep": _avg_low_obs_contacts,
            "avg_wall_contacts_per_ep": _avg_wall_contacts,
            "obstacle_contacts_per_path_progress_meter": _obstacle_contacts_per_path_progress_meter,
        }
        _manifest_path = os.path.join(_out_dir, "session_manifest.json")
        with open(_manifest_path, "w") as _mf:
            _json.dump(_manifest, _mf, indent=2)
        print(f"[INFO] Session manifest saved: {_manifest_path}")
        if _episode_metric_records:
            import csv as _csv
            _episodes_path = os.path.join(_out_dir, "episode_metrics.csv")
            with open(_episodes_path, "w", newline="") as _epf:
                _fieldnames = list(_episode_metric_records[0])
                _epw = _csv.DictWriter(_epf, fieldnames=_fieldnames)
                _epw.writeheader()
                _epw.writerows(_episode_metric_records)
            print(f"[INFO] Episode metrics saved: {_episodes_path}  ({len(_episode_metric_records)} episodes)")
        if _contact_event_records:
            import csv as _csv
            _events_path = os.path.join(_out_dir, "contact_events.csv")
            with open(_events_path, "w", newline="") as _ef:
                _ew = _csv.writer(_ef)
                _ew.writerow(["kind", "env", "episode_index", "step", "peak_force_n", "is_low_obstacle"])
                _ew.writerows(_contact_event_records)
            print(f"[INFO] Contact events saved: {_events_path}  ({len(_contact_event_records)} events)")

    # Auto-generate nav debug plot when --play_name is set.
    if _out_dir is not None and _nav_has_goal_markers:
        _nav_log = os.path.join(_out_dir, "nav_debug.log")
        if _out_log_file is not None:
            _out_log_file.flush()
        _plot_cmd = [
            sys.executable, "scripts/plot_nav_debug.py", _nav_log,
            "--output_dir", _out_dir,
        ]
        _plot_cmd.extend(_nav_debug_corridor_plot_args(_base_env, args_cli))
        print("[INFO] Generating nav debug plot ...")
        sys.stdout.flush()
        result = subprocess.run(_plot_cmd, capture_output=True, text=True, start_new_session=True)
        if result.stdout:
            print(result.stdout.strip())
        if result.returncode != 0 and result.stderr:
            print(f"[WARN] plot_nav_debug: {result.stderr.strip()}")

    if _out_log_file is not None:
        sys.stdout = sys.__stdout__
        _out_log_file.close()

    if _live_label_drawer is not None:
        _live_label_drawer.clear()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
