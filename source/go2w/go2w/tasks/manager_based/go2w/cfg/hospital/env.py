# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hospital play/eval environment configurations for Go2-W navigation policies.

These configs run existing Nav Teacher or depth-student checkpoints in structured
hospital corridors without requiring new training.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import MultiMeshRayCasterCfg
from isaaclab.utils import configclass

from ... import mdp
from ..navigation.env import (
    Go2wNavDepthRLDistillEnvCfg_PLAY,
    Go2wNavTeacherEnvCfg_PLAY,
    _configure_play_obstacle_obs,
    make_play_obstacle_cfg,
)
from ...mdp.navigation.hospital import events as _hospital_events
from ...mdp.navigation.hospital import specs as _hospital_specs
from ...mdp.navigation.hospital.specs import *  # noqa: F401, F403
from ...mdp.navigation.hospital.floor import (
    HOSPITAL_FLOOR_CORRIDOR_KIND,
    HOSPITAL_FLOOR_CORRIDOR_WIDTH,
    HOSPITAL_FLOOR_DYNAMIC_OBSTACLE_COUNT,
    HOSPITAL_FLOOR_EPISODE_LENGTH_S,
    HOSPITAL_FLOOR_ENV_SPACING,
    HOSPITAL_FLOOR_GOAL_DONE_RADIUS,
    HOSPITAL_FLOOR_LEG_LENGTH,
    HOSPITAL_FLOOR_RAMP_B_LOCAL_POSE,
    HOSPITAL_FLOOR_RAMP_LOCAL_POSE,
    HOSPITAL_FLOOR_ROBOT_START_LOCAL_XY,
    HOSPITAL_FLOOR_WALL_THICKNESS,
    HOSPITAL_RAMP_ASSET_NAME,
    HOSPITAL_RAMP_B_ASSET_NAME,
    hospital_floor_queue_groups as _hospital_floor_queue_groups,
    hospital_floor_seated_groups as _hospital_floor_seated_groups,
    hospital_floor_semantic_local_poses as _hospital_floor_semantic_local_poses,
)


# =============================================================================
# Hospital play environment (test existing Nav Teacher in hospital corridors)
# =============================================================================

# Default hospital corridor layout for play testing.
# Derived from HOSPITAL_LAYOUT_TEMPLATES["main_corridor"]:
#   corridor_kind="l_corridor", leg_length=12.0, corridor_width=2.6
HOSPITAL_PLAY_CORRIDOR_KIND = "l_corridor"
HOSPITAL_PLAY_LEG_LENGTH = 12.0
HOSPITAL_PLAY_CORRIDOR_WIDTH = 2.6
HOSPITAL_PLAY_WALL_THICKNESS = 0.20
HOSPITAL_PLAY_WALL_HEIGHT = 2.40
HOSPITAL_PLAY_EPISODE_LENGTH_S = 90.0
HOSPITAL_PLAY_ENV_SPACING = 30.0
HOSPITAL_PLAY_GOAL_DONE_RADIUS = 0.70
# Number of dynamic obstacles placed inside the corridor (excludes wall slots).
HOSPITAL_PLAY_DYNAMIC_OBSTACLE_COUNT = 12


def _shape_kind_from_id(shape_id: int) -> str:
    """Map obstacle metadata shape ids back to scene shape names."""
    if shape_id == OBSTACLE_SHAPE_CYLINDER:
        return "cylinder"
    if shape_id == OBSTACLE_SHAPE_CONE:
        return "cone"
    return "cuboid"


def _hospital_structured_slot_tables(
    corridor_kind: str,
    leg_length: float,
    corridor_width: float,
    wall_thickness: float,
    corridor_turn_length: float | None = None,
) -> dict[str, object]:
    """Return physical slot tables with walls followed by the hospital actor palette."""
    wall_specs = mdp.structured_corridor_wall_specs(
        corridor_kind,
        leg_length,
        corridor_width,
        wall_thickness,
        corridor_turn_length,
    )
    wall_count = len(wall_specs)
    slot_count = len(PLAY_OBSTACLE_NAMES)

    shape_ids = [OBSTACLE_SHAPE_CUBOID] * slot_count
    widths = [OBSTACLE_SIZE[0]] * slot_count
    depths = [OBSTACLE_SIZE[1]] * slot_count
    heights = [OBSTACLE_SIZE[2]] * slot_count
    labels = ["chair"] * slot_count
    colors = [HOSPITAL_LABEL_COLORS.get("chair", HOSPITAL_DEFAULT_COLOR)] * slot_count

    for slot_idx, (_, _, _, wall_length, wall_depth) in enumerate(wall_specs):
        shape_ids[slot_idx] = OBSTACLE_SHAPE_CUBOID
        widths[slot_idx] = wall_length
        depths[slot_idx] = wall_depth
        heights[slot_idx] = HOSPITAL_PLAY_WALL_HEIGHT
        labels[slot_idx] = "wall"
        colors[slot_idx] = HOSPITAL_LABEL_COLORS.get("wall", HOSPITAL_DEFAULT_COLOR)

    palette_slots = slot_count - wall_count
    for palette_idx in range(palette_slots):
        slot_idx = wall_count + palette_idx
        shape_ids[slot_idx] = HOSPITAL_PLAY_OBSTACLE_SHAPE_IDS[palette_idx]
        widths[slot_idx] = HOSPITAL_PLAY_OBSTACLE_WIDTHS[palette_idx]
        depths[slot_idx] = HOSPITAL_PLAY_OBSTACLE_DEPTHS[palette_idx]
        heights[slot_idx] = HOSPITAL_PLAY_OBSTACLE_HEIGHTS[palette_idx]
        labels[slot_idx] = HOSPITAL_PLAY_OBSTACLE_LABELS[palette_idx]
        colors[slot_idx] = HOSPITAL_LABEL_COLORS.get(labels[slot_idx], HOSPITAL_DEFAULT_COLOR)

    center_zs = tuple(height * 0.5 + OBSTACLE_GROUND_CLEARANCE for height in heights)
    return {
        "wall_count": wall_count,
        "shape_ids": tuple(shape_ids),
        "widths": tuple(widths),
        "depths": tuple(depths),
        "heights": tuple(heights),
        "center_zs": center_zs,
        "labels": tuple(labels),
        "colors": tuple(colors),
    }


def _apply_hospital_obstacle_asset_overrides(scene, slot_tables: dict[str, object]) -> None:
    """Make hospital physical assets match the slot metadata seen by depth sensors."""
    shape_ids = slot_tables["shape_ids"]
    widths = slot_tables["widths"]
    depths = slot_tables["depths"]
    heights = slot_tables["heights"]
    colors = slot_tables["colors"]
    for slot_idx, obstacle_name in enumerate(PLAY_OBSTACLE_NAMES):
        setattr(
            scene,
            obstacle_name,
            make_play_obstacle_cfg(
                obstacle_name,
                slot_idx,
                _shape_kind_from_id(shape_ids[slot_idx]),
                (widths[slot_idx], depths[slot_idx]),
                heights[slot_idx],
                colors[slot_idx],
            ),
        )


def _include_hospital_ramp_in_depth_camera(scene) -> None:
    """Expose both ramp halves to depth without adding them to obstacle contacts/rewards."""
    scene.depth_camera.mesh_prim_paths = [
        "/World/ground",
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/obstacle_.*",
            track_mesh_transforms=True,
            is_shared=True,
        ),
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/hospital_ramp",
            track_mesh_transforms=True,
            is_shared=True,
        ),
        MultiMeshRayCasterCfg.RaycastTargetCfg(
            prim_expr="{ENV_REGEX_NS}/hospital_ramp_b",
            track_mesh_transforms=True,
            is_shared=True,
        ),
    ]


def _hospital_group_registry(wall_count: int) -> list[dict]:
    """Build physical group relationships from dynamic-palette relation pairs."""
    return [
        {
            "relation_type": rel,
            "leader_name": f"obstacle_{wall_count + leader}",
            "follower_name": f"obstacle_{wall_count + follower}",
        }
        for leader, follower, rel in HOSPITAL_PLAY_GROUP_PAIRS
    ]


def _hospital_motion_slot_params(
    wall_count: int,
    slot_tables: dict,
    group_registry: list[dict],
    queue_groups: list[dict] | None = None,
    seated_groups: list[dict] | None = None,
) -> dict:
    """Return moving actors plus read-only static blockers for hospital motion."""
    labels = list(slot_tables["labels"])
    center_zs = tuple(slot_tables["center_zs"])
    keep_names: set[str] = set()

    for slot_idx in range(wall_count, len(PLAY_OBSTACLE_NAMES)):
        label = labels[slot_idx]
        label_spec = _hospital_specs.HOSPITAL_LABEL_SPECS.get(label)
        if label_spec is None:
            continue
        if label_spec.motion_profile != "static" or label_spec.category in {"furniture", "misc"}:
            keep_names.add(PLAY_OBSTACLE_NAMES[slot_idx])

    for group in group_registry:
        keep_names.add(group["leader_name"])
        keep_names.add(group["follower_name"])
    for queue_group in queue_groups or ():
        keep_names.update(queue_group.get("names", ()))
    for seated_group in seated_groups or ():
        name = seated_group.get("name")
        if name is not None:
            keep_names.add(name)

    indices = [
        slot_idx
        for slot_idx, name in enumerate(PLAY_OBSTACLE_NAMES)
        if slot_idx >= wall_count and name in keep_names
    ]
    return {
        "obstacle_names": [PLAY_OBSTACLE_NAMES[slot_idx] for slot_idx in indices],
        "obstacle_labels": [labels[slot_idx] for slot_idx in indices],
        "obstacle_indices": indices,
        "obstacle_center_zs": tuple(center_zs[slot_idx] for slot_idx in indices),
    }


def _structured_path_update_event() -> EventTerm:
    """Update the rolling local waypoint every step for structured routes."""
    return EventTerm(
        func=mdp.update_navigation_path_waypoint_event,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "lookahead_distance": 1.25,
            "waypoint_reach_radius": 0.45,
            "adaptive_lookahead": True,
            "lookahead_min": 0.6,
            "curvature_scan_horizon": 2.5,
            "curvature_threshold": 0.3,
        },
    )


def _set_structured_goal_termination(cfg, position_threshold: float) -> None:
    """Terminate structured play when the robot reaches the final path goal."""
    cfg.terminations.structured_goal_reached = DoneTerm(
        func=mdp.navigation_path_final_goal_reached,
        params={"position_threshold": position_threshold},
    )


def _configure_hospital_structured_depth_play(
    cfg,
    *,
    corridor_kind: str,
    leg_length: float,
    corridor_width: float,
    wall_thickness: float,
    dynamic_obstacle_count: int,
    speed_scale: float = 1.0,
    semantic_local_poses: tuple[tuple[int, float, float, float], ...] | None = None,
    queue_groups: list[dict] | None = None,
    seated_groups: list[dict] | None = None,
    ramp_local_pose: tuple[float, ...] | None = None,
    ramp_b_local_pose: tuple[float, ...] | None = None,
    robot_start_local_xy: tuple[float, float] = (0.0, 0.0),
    min_inter_obstacle_dist: float = 0.80,
) -> None:
    """Configure a depth play env as a structured hospital scene."""
    slot_tables = _hospital_structured_slot_tables(
        corridor_kind,
        leg_length,
        corridor_width,
        wall_thickness,
    )
    _apply_hospital_obstacle_asset_overrides(cfg.scene, slot_tables)
    wall_count = int(slot_tables["wall_count"])
    cfg.events.reset_obstacles.func = mdp.reset_structured_astar_corridor
    cfg.events.reset_obstacles.params = {
        "obstacle_names": PLAY_OBSTACLE_NAMES,
        "corridor_kind": corridor_kind,
        "leg_length": leg_length,
        "corridor_width": corridor_width,
        "wall_thickness": wall_thickness,
        "dynamic_obstacle_count": dynamic_obstacle_count,
        "obstacle_z": OBSTACLE_Z,
        "min_inter_obstacle_dist": min_inter_obstacle_dist,
        "obstacle_radius_margin": NAV_OBSTACLE_RADIUS_MARGIN,
        "fixed_obstacle_shape_ids": slot_tables["shape_ids"],
        "fixed_obstacle_widths": slot_tables["widths"],
        "fixed_obstacle_depths": slot_tables["depths"],
        "fixed_obstacle_center_zs": slot_tables["center_zs"],
        "obstacle_labels": slot_tables["labels"],
        "fixed_obstacle_local_poses": semantic_local_poses,
        "robot_start_local_xy": robot_start_local_xy,
        "ramp_asset_name": HOSPITAL_RAMP_ASSET_NAME,
        "ramp_local_pose": ramp_local_pose,
        "ramp_b_asset_name": HOSPITAL_RAMP_B_ASSET_NAME,
        "ramp_b_local_pose": ramp_b_local_pose,
        "randomize_obstacle_yaw": NAV_RANDOMIZE_OBSTACLE_YAW,
        "obstacle_yaw_range": NAV_OBSTACLE_YAW_RANGE,
        "robot_inflation": 0.50,
        "clearance_cost_weight": 3.0,
        "clearance_cost_sigma": 0.60,
        "corner_rounding": True,
        "corner_radius": 0.80,
        "goal_exclusion_radius": NAV_GOAL_EXCLUSION_RADIUS,
        "dynamic_start_exclusion_radius": 1.8,
        "dynamic_robot_keepout_radius": 1.25,
        "lookahead_distance": 1.25,
        "waypoint_reach_radius": 0.45,
        "adaptive_lookahead": True,
        "lookahead_min": 0.6,
        "curvature_scan_horizon": 2.5,
        "curvature_threshold": 0.3,
    }

    cfg.events.navigation_path_update = _structured_path_update_event()
    cfg.events.hospital_velocity_resample = None
    cfg.events.hospital_group_update = None
    group_registry = _hospital_group_registry(wall_count)
    motion_slot_params = _hospital_motion_slot_params(
        wall_count,
        slot_tables,
        group_registry,
        queue_groups=queue_groups,
        seated_groups=seated_groups,
    )
    motion_params = {
        **motion_slot_params,
        "group_registry": group_registry,
        "min_inter_obstacle_dist": 0.25,
        "active_distance": 24.0,
        "goal_exclusion_radius": NAV_GOAL_EXCLUSION_RADIUS,
        "robot_keepout_radius": 1.25,
    }
    if queue_groups is not None:
        motion_params["queue_groups"] = queue_groups
    if seated_groups is not None:
        motion_params["seated_groups"] = seated_groups
    if speed_scale != 1.0:
        motion_params["speed_scale"] = speed_scale
    cfg.events.hospital_dynamic_motion = EventTerm(
        func=_hospital_events.move_hospital_dynamic_obstacles,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params=motion_params,
    )


@configclass
class Go2wHospitalPlayEnvCfg(Go2wNavTeacherEnvCfg_PLAY):
    """Play/eval env that runs the Nav Teacher policy in a hospital-style corridor.

    Uses ``reset_structured_astar_corridor`` to lay out an L-shaped corridor
    (main_corridor template: leg_length=12 m, width=2.6 m) with hospital-scale
    obstacle footprints (patients, wheelchairs, carts, beds).  No new training
    is required — load any existing Nav Teacher checkpoint directly.
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = HOSPITAL_PLAY_EPISODE_LENGTH_S
        self.scene.env_spacing = HOSPITAL_PLAY_ENV_SPACING
        slot_tables = _hospital_structured_slot_tables(
            HOSPITAL_PLAY_CORRIDOR_KIND,
            HOSPITAL_PLAY_LEG_LENGTH,
            HOSPITAL_PLAY_CORRIDOR_WIDTH,
            HOSPITAL_PLAY_WALL_THICKNESS,
        )
        _apply_hospital_obstacle_asset_overrides(self.scene, slot_tables)
        _wall_count = int(slot_tables["wall_count"])
        # _configure_play_common already called by parent; redo obstacle event
        # to use the structured corridor reset instead of the flat scatter reset.
        self.events.reset_obstacles.func = mdp.reset_structured_astar_corridor
        self.events.reset_obstacles.params = {
            "obstacle_names": PLAY_OBSTACLE_NAMES,
            "corridor_kind": HOSPITAL_PLAY_CORRIDOR_KIND,
            "leg_length": HOSPITAL_PLAY_LEG_LENGTH,
            "corridor_width": HOSPITAL_PLAY_CORRIDOR_WIDTH,
            "wall_thickness": HOSPITAL_PLAY_WALL_THICKNESS,
            "dynamic_obstacle_count": HOSPITAL_PLAY_DYNAMIC_OBSTACLE_COUNT,
            "obstacle_z": OBSTACLE_Z,
            "min_inter_obstacle_dist": 0.80,
            "obstacle_radius_margin": NAV_OBSTACLE_RADIUS_MARGIN,
            "fixed_obstacle_shape_ids": slot_tables["shape_ids"],
            "fixed_obstacle_widths": slot_tables["widths"],
            "fixed_obstacle_depths": slot_tables["depths"],
            "fixed_obstacle_center_zs": slot_tables["center_zs"],
            "obstacle_labels": slot_tables["labels"],
            "randomize_obstacle_yaw": NAV_RANDOMIZE_OBSTACLE_YAW,
            "obstacle_yaw_range": NAV_OBSTACLE_YAW_RANGE,
            "goal_exclusion_radius": NAV_GOAL_EXCLUSION_RADIUS,
            "dynamic_start_exclusion_radius": 1.8,
            "dynamic_robot_keepout_radius": 1.25,
            "lookahead_distance": 1.25,
            "waypoint_reach_radius": 0.45,
            "adaptive_lookahead": True,
            "lookahead_min": 0.6,
            "curvature_scan_horizon": 2.5,
            "curvature_threshold": 0.3,
        }
        # Redirect privileged obs to the full play obstacle list.
        _configure_play_obstacle_obs(self.observations.policy)

        self.events.navigation_path_update = _structured_path_update_event()
        _set_structured_goal_termination(self, HOSPITAL_PLAY_GOAL_DONE_RADIUS)
        self.events.hospital_velocity_resample = None
        self.events.hospital_group_update = None
        group_registry = _hospital_group_registry(_wall_count)
        motion_slot_params = _hospital_motion_slot_params(_wall_count, slot_tables, group_registry)
        self.events.hospital_dynamic_motion = EventTerm(
            func=_hospital_events.move_hospital_dynamic_obstacles,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={
                **motion_slot_params,
                "group_registry": group_registry,
                "min_inter_obstacle_dist": 0.25,
                "active_distance": 24.0,
                "goal_exclusion_radius": NAV_GOAL_EXCLUSION_RADIUS,
                "robot_keepout_radius": 1.25,
            },
        )


# =============================================================================
# Hospital play environment — Nav Teacher, full hospital floor
# =============================================================================


@configclass
class Go2wHospitalFloorPlayEnvCfg(Go2wNavTeacherEnvCfg_PLAY):
    """Teacher play/eval env for the full hospital floor.

    Same obs as Go2wHospitalPlayEnvCfg (NavTeacherObsCfg, privileged obstacle
    positions) but uses the full-floor corridor layout instead of the L-corridor.
    Load any existing Nav Teacher checkpoint directly.
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = HOSPITAL_FLOOR_EPISODE_LENGTH_S
        self.scene.env_spacing = HOSPITAL_FLOOR_ENV_SPACING
        slot_tables = _hospital_structured_slot_tables(
            HOSPITAL_FLOOR_CORRIDOR_KIND,
            HOSPITAL_FLOOR_LEG_LENGTH,
            HOSPITAL_FLOOR_CORRIDOR_WIDTH,
            HOSPITAL_FLOOR_WALL_THICKNESS,
        )
        wall_count = int(slot_tables["wall_count"])
        _configure_hospital_structured_depth_play(
            self,
            corridor_kind=HOSPITAL_FLOOR_CORRIDOR_KIND,
            leg_length=HOSPITAL_FLOOR_LEG_LENGTH,
            corridor_width=HOSPITAL_FLOOR_CORRIDOR_WIDTH,
            wall_thickness=HOSPITAL_FLOOR_WALL_THICKNESS,
            dynamic_obstacle_count=HOSPITAL_FLOOR_DYNAMIC_OBSTACLE_COUNT,
            speed_scale=0.8,
            semantic_local_poses=_hospital_floor_semantic_local_poses(wall_count, HOSPITAL_FLOOR_LEG_LENGTH),
            queue_groups=_hospital_floor_queue_groups(wall_count),
            seated_groups=_hospital_floor_seated_groups(wall_count),
            ramp_local_pose=HOSPITAL_FLOOR_RAMP_LOCAL_POSE,
            ramp_b_local_pose=HOSPITAL_FLOOR_RAMP_B_LOCAL_POSE,
            robot_start_local_xy=HOSPITAL_FLOOR_ROBOT_START_LOCAL_XY,
            min_inter_obstacle_dist=1.05,
        )
        _set_structured_goal_termination(self, HOSPITAL_FLOOR_GOAL_DONE_RADIUS)
        _configure_play_obstacle_obs(self.observations.policy)


# =============================================================================
# Hospital play environment — Depth Camera Student
# =============================================================================


@configclass
class Go2wHospitalDepthPlayEnvCfg(Go2wNavDepthRLDistillEnvCfg_PLAY):
    """Play/eval env for the depth-camera student in a hospital-style corridor.

    Shares depth camera and student obs (state + depth stack) with the standard
    depth-distillation play env.  Replaces the flat scatter reset with
    ``reset_structured_astar_corridor`` using hospital-scale obstacle footprints.
    No new training required — load any existing depth-student checkpoint.
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = HOSPITAL_PLAY_EPISODE_LENGTH_S
        self.scene.env_spacing = HOSPITAL_PLAY_ENV_SPACING
        slot_tables = _hospital_structured_slot_tables(
            HOSPITAL_PLAY_CORRIDOR_KIND,
            HOSPITAL_PLAY_LEG_LENGTH,
            HOSPITAL_PLAY_CORRIDOR_WIDTH,
            HOSPITAL_PLAY_WALL_THICKNESS,
        )
        _apply_hospital_obstacle_asset_overrides(self.scene, slot_tables)
        _wall_count = int(slot_tables["wall_count"])
        self.events.reset_obstacles.func = mdp.reset_structured_astar_corridor
        self.events.reset_obstacles.params = {
            "obstacle_names": PLAY_OBSTACLE_NAMES,
            "corridor_kind": HOSPITAL_PLAY_CORRIDOR_KIND,
            "leg_length": HOSPITAL_PLAY_LEG_LENGTH,
            "corridor_width": HOSPITAL_PLAY_CORRIDOR_WIDTH,
            "wall_thickness": HOSPITAL_PLAY_WALL_THICKNESS,
            "dynamic_obstacle_count": HOSPITAL_PLAY_DYNAMIC_OBSTACLE_COUNT,
            "obstacle_z": OBSTACLE_Z,
            "min_inter_obstacle_dist": 0.80,
            "obstacle_radius_margin": NAV_OBSTACLE_RADIUS_MARGIN,
            "fixed_obstacle_shape_ids": slot_tables["shape_ids"],
            "fixed_obstacle_widths": slot_tables["widths"],
            "fixed_obstacle_depths": slot_tables["depths"],
            "fixed_obstacle_center_zs": slot_tables["center_zs"],
            "obstacle_labels": slot_tables["labels"],
            "randomize_obstacle_yaw": NAV_RANDOMIZE_OBSTACLE_YAW,
            "obstacle_yaw_range": NAV_OBSTACLE_YAW_RANGE,
            "goal_exclusion_radius": NAV_GOAL_EXCLUSION_RADIUS,
            "dynamic_start_exclusion_radius": 1.8,
            "dynamic_robot_keepout_radius": 1.25,
            "lookahead_distance": 1.25,
            "waypoint_reach_radius": 0.45,
            "adaptive_lookahead": True,
            "lookahead_min": 0.6,
            "curvature_scan_horizon": 2.5,
            "curvature_threshold": 0.3,
        }

        self.events.navigation_path_update = _structured_path_update_event()
        _set_structured_goal_termination(self, HOSPITAL_PLAY_GOAL_DONE_RADIUS)
        self.events.hospital_velocity_resample = None
        self.events.hospital_group_update = None
        group_registry = _hospital_group_registry(_wall_count)
        motion_slot_params = _hospital_motion_slot_params(_wall_count, slot_tables, group_registry)
        self.events.hospital_dynamic_motion = EventTerm(
            func=_hospital_events.move_hospital_dynamic_obstacles,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={
                **motion_slot_params,
                "group_registry": group_registry,
                "min_inter_obstacle_dist": 0.25,
                "active_distance": 24.0,
                "goal_exclusion_radius": NAV_GOAL_EXCLUSION_RADIUS,
                "robot_keepout_radius": 1.25,
            },
        )


# =============================================================================
# Hospital ward play environment — Depth Camera Student
# =============================================================================

# Hospital ward: long main corridor (3× leg_length) with two side branches.
# Branch 1 (at x=leg_length) is a dead-end arm populated with dynamic obstacles.
# Branch 2 (at x=2×leg_length) is the goal arm the robot navigates into.
# Wall slots are derived from the structured corridor specs; remaining slots use
# the hospital actor palette, so wall count changes do not discard key actors.
HOSPITAL_WARD_CORRIDOR_KIND = "hospital_ward"
HOSPITAL_WARD_LEG_LENGTH = 10.0    # total main = 30 m, each branch = 10 m
HOSPITAL_WARD_CORRIDOR_WIDTH = 3.0  # patient-care corridor with bed/cart passing room
HOSPITAL_WARD_WALL_THICKNESS = 0.25
HOSPITAL_WARD_DYNAMIC_OBSTACLE_COUNT = 16  # includes child/dog, pushed wheelchair/cart/gurney, self wheelchair, IV pole
_HOSPITAL_WARD_WALL_COUNT = 11             # 11 walls (open left entrance, no end cap at x=0)
HOSPITAL_WARD_EPISODE_LENGTH_S = 140.0
HOSPITAL_WARD_ENV_SPACING = 46.0
HOSPITAL_WARD_GOAL_DONE_RADIUS = 0.80


@configclass
class Go2wHospitalWardDepthPlayEnvCfg(Go2wNavDepthRLDistillEnvCfg_PLAY):
    """Depth-student play/eval env in a hospital ward floor layout.

    The ward has a 30 m main corridor with two perpendicular branches (each 10 m),
    forming two T-junctions the robot navigates through.  Branch 1 is a dead-end
    populated with dynamic obstacles that can spill into the main corridor.  Goal
    is the tip of branch 2.  Corridor width is 3.0 m, wall thickness is 0.25 m.
    Dynamic actors use the hospital label palette at 0.7× speed.
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = HOSPITAL_WARD_EPISODE_LENGTH_S
        self.scene.env_spacing = HOSPITAL_WARD_ENV_SPACING
        slot_tables = _hospital_structured_slot_tables(
            HOSPITAL_WARD_CORRIDOR_KIND,
            HOSPITAL_WARD_LEG_LENGTH,
            HOSPITAL_WARD_CORRIDOR_WIDTH,
            HOSPITAL_WARD_WALL_THICKNESS,
        )
        _apply_hospital_obstacle_asset_overrides(self.scene, slot_tables)
        _wall_count = int(slot_tables["wall_count"])
        self.events.reset_obstacles.func = mdp.reset_structured_astar_corridor
        self.events.reset_obstacles.params = {
            "obstacle_names": PLAY_OBSTACLE_NAMES,
            "corridor_kind": HOSPITAL_WARD_CORRIDOR_KIND,
            "leg_length": HOSPITAL_WARD_LEG_LENGTH,
            "corridor_width": HOSPITAL_WARD_CORRIDOR_WIDTH,
            "wall_thickness": HOSPITAL_WARD_WALL_THICKNESS,
            "dynamic_obstacle_count": HOSPITAL_WARD_DYNAMIC_OBSTACLE_COUNT,
            "obstacle_z": OBSTACLE_Z,
            "min_inter_obstacle_dist": 1.00,
            "obstacle_radius_margin": NAV_OBSTACLE_RADIUS_MARGIN,
            "fixed_obstacle_shape_ids": slot_tables["shape_ids"],
            "fixed_obstacle_widths": slot_tables["widths"],
            "fixed_obstacle_depths": slot_tables["depths"],
            "fixed_obstacle_center_zs": slot_tables["center_zs"],
            "obstacle_labels": slot_tables["labels"],
            "randomize_obstacle_yaw": NAV_RANDOMIZE_OBSTACLE_YAW,
            "obstacle_yaw_range": NAV_OBSTACLE_YAW_RANGE,
            "goal_exclusion_radius": NAV_GOAL_EXCLUSION_RADIUS,
            "dynamic_start_exclusion_radius": 1.8,
            "dynamic_robot_keepout_radius": 1.25,
            "lookahead_distance": 1.25,
            "waypoint_reach_radius": 0.45,
            "adaptive_lookahead": True,
            "lookahead_min": 0.6,
            "curvature_scan_horizon": 2.5,
            "curvature_threshold": 0.3,
        }

        self.events.navigation_path_update = _structured_path_update_event()
        _set_structured_goal_termination(self, HOSPITAL_WARD_GOAL_DONE_RADIUS)
        self.events.hospital_velocity_resample = None
        self.events.hospital_group_update = None
        group_registry = _hospital_group_registry(_wall_count)
        motion_slot_params = _hospital_motion_slot_params(_wall_count, slot_tables, group_registry)
        self.events.hospital_dynamic_motion = EventTerm(
            func=_hospital_events.move_hospital_dynamic_obstacles,
            mode="interval",
            interval_range_s=(0.0, 0.0),
            params={
                **motion_slot_params,
                "group_registry": group_registry,
                "speed_scale": 0.7,
                "min_inter_obstacle_dist": 0.30,
                "active_distance": 24.0,
                "goal_exclusion_radius": NAV_GOAL_EXCLUSION_RADIUS,
                "robot_keepout_radius": 1.25,
            },
        )


# =============================================================================
# Hospital full-floor play environment — Depth Camera Student
# (constants imported from mdp/hospital/floor.py)
# =============================================================================


@configclass
class Go2wHospitalFloorDepthPlayEnvCfg(Go2wNavDepthRLDistillEnvCfg_PLAY):
    """Depth-student play/eval env for a combined hospital floor.

    Includes reception queueing, waiting bench occupancy, doorway crossing,
    ward/service flow with a pushed patient gurney, and a mild ramp connector.
    """

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = HOSPITAL_FLOOR_EPISODE_LENGTH_S
        self.scene.env_spacing = HOSPITAL_FLOOR_ENV_SPACING
        slot_tables = _hospital_structured_slot_tables(
            HOSPITAL_FLOOR_CORRIDOR_KIND,
            HOSPITAL_FLOOR_LEG_LENGTH,
            HOSPITAL_FLOOR_CORRIDOR_WIDTH,
            HOSPITAL_FLOOR_WALL_THICKNESS,
        )
        wall_count = int(slot_tables["wall_count"])
        _configure_hospital_structured_depth_play(
            self,
            corridor_kind=HOSPITAL_FLOOR_CORRIDOR_KIND,
            leg_length=HOSPITAL_FLOOR_LEG_LENGTH,
            corridor_width=HOSPITAL_FLOOR_CORRIDOR_WIDTH,
            wall_thickness=HOSPITAL_FLOOR_WALL_THICKNESS,
            dynamic_obstacle_count=HOSPITAL_FLOOR_DYNAMIC_OBSTACLE_COUNT,
            speed_scale=0.8,
            semantic_local_poses=_hospital_floor_semantic_local_poses(wall_count, HOSPITAL_FLOOR_LEG_LENGTH),
            queue_groups=_hospital_floor_queue_groups(wall_count),
            seated_groups=_hospital_floor_seated_groups(wall_count),
            ramp_local_pose=HOSPITAL_FLOOR_RAMP_LOCAL_POSE,
            ramp_b_local_pose=HOSPITAL_FLOOR_RAMP_B_LOCAL_POSE,
            robot_start_local_xy=HOSPITAL_FLOOR_ROBOT_START_LOCAL_XY,
            min_inter_obstacle_dist=1.05,
        )
        _set_structured_goal_termination(self, HOSPITAL_FLOOR_GOAL_DONE_RADIUS)
        _include_hospital_ramp_in_depth_camera(self.scene)
