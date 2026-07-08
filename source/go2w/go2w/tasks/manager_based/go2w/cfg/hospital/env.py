# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hospital play/eval environment configurations for Go2-W navigation policies.

These configs run existing Nav Teacher or depth-student checkpoints in structured
hospital corridors without requiring new training.  All corridor walls are rendered
via a TerrainImporter trimesh (HospitalWallSubTerrainCfg) so every obstacle slot
is available for dynamic actors.
"""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import MultiMeshRayCasterCfg
from isaaclab.sensors import patterns as _sensor_patterns
from isaaclab.utils import configclass

from ... import mdp
from ..navigation.env import (
    Go2wHospitalTeacherEnvCfg,
    Go2wNavDepthRLDistillEnvCfg_PLAY,
    Go2wNavDepthLongHistRLDistillEnvCfg_PLAY,
    Go2wNavDepthSparseRLDistillEnvCfg_PLAY,
    Go2wNavDepthMultiCamRLDistillEnvCfg_PLAY,
    Go2wNavTeacherEnvCfg_PLAY,
    ObstaclePlayTerrainWallSceneCfg,
    DepthObstaclePlayTerrainWallSceneCfg,
    DepthObstacleMultiCamPlayTerrainWallSceneCfg,
    _configure_play_obstacle_obs,
    make_play_obstacle_cfg,
)
from ..navigation.observations import (
    NavHospitalTeacherObsCfg,
    NavDepthRLDistillLongHistObsCfg,
    NavDepthRLDistillMultiCamObsCfg,
)
from ...mdp.navigation.hospital.terrain import HospitalWallSubTerrainCfg
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
# Hospital play environment constants
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
# Number of dynamic obstacles placed inside the corridor (all slots are actors).
HOSPITAL_PLAY_DYNAMIC_OBSTACLE_COUNT = 12

# Hospital ward: long main corridor (3× leg_length) with two side branches.
HOSPITAL_WARD_CORRIDOR_KIND = "hospital_ward"
HOSPITAL_WARD_LEG_LENGTH = 10.0    # total main = 30 m, each branch = 10 m
HOSPITAL_WARD_CORRIDOR_WIDTH = 3.0  # patient-care corridor with bed/cart passing room
HOSPITAL_WARD_WALL_THICKNESS = 0.25
HOSPITAL_WARD_DYNAMIC_OBSTACLE_COUNT = 16
HOSPITAL_WARD_EPISODE_LENGTH_S = 140.0
HOSPITAL_WARD_ENV_SPACING = 46.0
HOSPITAL_WARD_GOAL_DONE_RADIUS = 0.80


# =============================================================================
# Shared helpers
# =============================================================================


def _shape_kind_from_id(shape_id: int) -> str:
    """Map obstacle metadata shape ids back to scene shape names."""
    if shape_id == OBSTACLE_SHAPE_CYLINDER:
        return "cylinder"
    if shape_id == OBSTACLE_SHAPE_CONE:
        return "cone"
    return "cuboid"


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
    """Append ramp prim targets to all depth cameras, preserving existing terrain wall entries."""
    ramp_targets = [
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
    for cam_attr in ("depth_camera", "depth_camera_left", "depth_camera_right", "depth_camera_rear"):
        cam = getattr(scene, cam_attr, None)
        if cam is not None:
            cam.mesh_prim_paths = list(cam.mesh_prim_paths) + ramp_targets


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
            "waypoint_reach_radius": NAV_GOAL_SUCCESS_POSITION_THRESHOLD,
            "adaptive_lookahead": True,
            "lookahead_min": 0.55,
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


# =============================================================================
# Terrain wall helpers
# =============================================================================


def _hospital_actor_only_slot_tables(slot_count: int = len(PLAY_OBSTACLE_NAMES)) -> dict[str, object]:
    """Return slot tables with all slots allocated to hospital actors (no wall slots).

    The hospital play palette is cycled to fill all ``slot_count`` slots.
    """
    palette_size = len(HOSPITAL_PLAY_OBSTACLE_SHAPE_IDS)
    shape_ids = [HOSPITAL_PLAY_OBSTACLE_SHAPE_IDS[i % palette_size] for i in range(slot_count)]
    widths    = [HOSPITAL_PLAY_OBSTACLE_WIDTHS[i % palette_size]     for i in range(slot_count)]
    depths    = [HOSPITAL_PLAY_OBSTACLE_DEPTHS[i % palette_size]     for i in range(slot_count)]
    heights   = [HOSPITAL_PLAY_OBSTACLE_HEIGHTS[i % palette_size]    for i in range(slot_count)]
    labels    = [HOSPITAL_PLAY_OBSTACLE_LABELS[i % palette_size]     for i in range(slot_count)]
    colors    = [HOSPITAL_LABEL_COLORS.get(labels[i], HOSPITAL_DEFAULT_COLOR) for i in range(slot_count)]
    center_zs = tuple(h * 0.5 + OBSTACLE_GROUND_CLEARANCE for h in heights)
    return {
        "wall_count": 0,
        "shape_ids":  tuple(shape_ids),
        "widths":     tuple(widths),
        "depths":     tuple(depths),
        "heights":    tuple(heights),
        "center_zs":  center_zs,
        "labels":     tuple(labels),
        "colors":     tuple(colors),
    }


def _apply_mesh_wall_overrides(
    cfg,
    *,
    corridor_kind: str,
    leg_length: float,
    corridor_width: float,
    wall_thickness: float,
    wall_height: float = 2.40,
    corridor_turn_length: float | None = None,
    dynamic_obstacle_count: int,
    speed_scale: float = 1.0,
    semantic_local_poses: tuple | None = None,
    queue_groups: list | None = None,
    seated_groups: list | None = None,
    ramp_asset_name: str | None = None,
    ramp_local_pose: tuple | None = None,
    ramp_b_asset_name: str | None = None,
    ramp_b_local_pose: tuple | None = None,
    robot_start_local_xy: tuple[float, float] = (0.0, 0.0),
    min_inter_obstacle_dist: float = 0.80,
    tile_size: float = 30.0,
) -> None:
    """Convert a hospital play env to use TerrainImporter corridor walls.

    Must be called AFTER ``reset_obstacles.func`` and ``reset_obstacles.params``
    have been initialised by the caller.  This function:

    1. Sets ``scene.terrain`` to a ``TerrainImporterCfg`` with a single-corridor
       sub-terrain so walls get proper global-mesh physics collision.
    2. Re-assigns all 64 obstacle slots to actors (no wall slots).
    3. Merges corridor geometry into ``reset_obstacles.params`` and adds
       ``skip_wall_placement=True`` / ``use_env_origin=True``.
    4. Rebuilds the dynamic-motion event with the actor-only slot mapping.
    5. Adds ``/World/terrain`` to the depth camera ray-cast targets if present.
    """
    import math as _math
    import isaaclab.sim as _sim_utils
    from isaaclab.terrains import TerrainImporterCfg as _TerrainImporterCfg
    from isaaclab.terrains import TerrainGeneratorCfg as _TerrainGeneratorCfg

    slot_tables = _hospital_actor_only_slot_tables()
    wall_count  = 0  # terrain walls → no obstacle slots used for walls

    # 1. Set terrain via TerrainImporter (proper global mesh, physics collision for all envs)
    num_envs = getattr(cfg.scene, "num_envs", 4)
    num_rows = max(1, _math.isqrt(num_envs))
    num_cols = max(1, (num_envs + num_rows - 1) // num_rows)
    cfg.scene.terrain = _TerrainImporterCfg(
        prim_path="/World/terrain",
        terrain_type="generator",
        terrain_generator=_TerrainGeneratorCfg(
            num_rows=num_rows,
            num_cols=num_cols,
            size=(tile_size, tile_size),
            curriculum=False,
            sub_terrains={
                "wall": HospitalWallSubTerrainCfg(
                    size=(tile_size, tile_size),
                    corridor_kind=corridor_kind,
                    leg_length=leg_length,
                    corridor_width=corridor_width,
                    wall_thickness=wall_thickness,
                    wall_height=wall_height,
                    corridor_turn_length=corridor_turn_length,
                )
            },
        ),
        use_terrain_origins=True,
        physics_material=_sim_utils.RigidBodyMaterialCfg(
            static_friction=0.9,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
        visual_material=_sim_utils.PreviewSurfaceCfg(diffuse_color=(0.58, 0.60, 0.64)),
    )

    # 2. Add /World/terrain to all depth camera ray-cast targets if present.
    # Also strip /World/ground — the wall terrain mesh has no floor geometry, so
    # keeping ground here would add baselevel depth hits absent from maze training.
    for _cam_attr in ("depth_camera", "depth_camera_left", "depth_camera_right", "depth_camera_rear"):
        _cam = getattr(cfg.scene, _cam_attr, None)
        if _cam is not None:
            _existing = [p for p in _cam.mesh_prim_paths if not (isinstance(p, str) and p == "/World/ground")]
            if "/World/terrain" not in _existing:
                _cam.mesh_prim_paths = ["/World/terrain"] + _existing
            else:
                _cam.mesh_prim_paths = _existing

    # 3. Re-assign obstacle asset shapes/colors (all actors now)
    _apply_hospital_obstacle_asset_overrides(cfg.scene, slot_tables)

    # 4. Merge corridor geometry into reset params
    base_params = cfg.events.reset_obstacles.params
    cfg.events.reset_obstacles.params = {
        **base_params,
        "corridor_kind":               corridor_kind,
        "leg_length":                  leg_length,
        "corridor_width":              corridor_width,
        "wall_thickness":              wall_thickness,
        "corridor_turn_length":        corridor_turn_length,
        "obstacle_names":              PLAY_OBSTACLE_NAMES,
        "fixed_obstacle_shape_ids":    slot_tables["shape_ids"],
        "fixed_obstacle_widths":       slot_tables["widths"],
        "fixed_obstacle_depths":       slot_tables["depths"],
        "fixed_obstacle_center_zs":    slot_tables["center_zs"],
        "obstacle_labels":             slot_tables["labels"],
        "dynamic_obstacle_count":      dynamic_obstacle_count,
        "robot_start_local_xy":        robot_start_local_xy,
        "min_inter_obstacle_dist":     min_inter_obstacle_dist,
        "fixed_obstacle_local_poses":  semantic_local_poses,
        "ramp_asset_name":             ramp_asset_name,
        "ramp_local_pose":             ramp_local_pose,
        "ramp_b_asset_name":           ramp_b_asset_name,
        "ramp_b_local_pose":           ramp_b_local_pose,
        "skip_wall_placement":         True,
        "use_env_origin":              True,
        "reset_robot_pose":            True,
    }

    # 5. Rebuild dynamic motion with actor-only slot mapping
    group_registry    = _hospital_group_registry(wall_count)
    motion_slot_params = _hospital_motion_slot_params(
        wall_count, slot_tables, group_registry,
        queue_groups=queue_groups, seated_groups=seated_groups,
    )
    motion_params = {
        **motion_slot_params,
        "group_registry":        group_registry,
        "min_inter_obstacle_dist": 0.25,
        "active_distance":       24.0,
        "goal_exclusion_radius": NAV_GOAL_EXCLUSION_RADIUS,
        "robot_keepout_radius":  1.25,
    }
    if queue_groups  is not None: motion_params["queue_groups"]  = queue_groups
    if seated_groups is not None: motion_params["seated_groups"] = seated_groups
    if speed_scale   != 1.0:     motion_params["speed_scale"]   = speed_scale
    cfg.events.hospital_dynamic_motion = EventTerm(
        func=_hospital_events.move_hospital_dynamic_obstacles,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params=motion_params,
    )


def _base_structured_reset_params() -> dict:
    """Return shared reset params common to all structured hospital play envs."""
    return {
        "obstacle_names":              PLAY_OBSTACLE_NAMES,
        "obstacle_z":                  OBSTACLE_Z,
        "obstacle_radius_margin":      NAV_OBSTACLE_RADIUS_MARGIN,
        "randomize_obstacle_yaw":      NAV_RANDOMIZE_OBSTACLE_YAW,
        "obstacle_yaw_range":          NAV_OBSTACLE_YAW_RANGE,
        "goal_exclusion_radius":       NAV_GOAL_EXCLUSION_RADIUS,
        "dynamic_start_exclusion_radius": 1.8,
        "dynamic_robot_keepout_radius":   1.25,
        "robot_inflation":             0.50,
        "clearance_cost_weight":       3.0,
        "clearance_cost_sigma":        0.60,
        "corner_rounding":             True,
        "corner_radius":               0.80,
        "lookahead_distance":          1.25,
        "waypoint_reach_radius":       NAV_GOAL_SUCCESS_POSITION_THRESHOLD,
        "adaptive_lookahead":          True,
        "lookahead_min":               0.55,
        "curvature_scan_horizon":      2.5,
        "curvature_threshold":         0.3,
    }


# =============================================================================
# L-corridor play environments
# =============================================================================


@configclass
class Go2wHospitalPlayEnvCfg(Go2wNavTeacherEnvCfg_PLAY):
    """Play/eval env that runs the Nav Teacher policy in a hospital-style corridor.

    L-shaped corridor (leg_length=12 m, width=2.6 m) with hospital-scale obstacle
    footprints.  Corridor walls are rendered via a TerrainImporter trimesh so all 64
    obstacle slots are available for dynamic actors.  No new training is required —
    load any existing Nav Teacher checkpoint directly.
    """

    scene: ObstaclePlayTerrainWallSceneCfg = ObstaclePlayTerrainWallSceneCfg(
        num_envs=4, env_spacing=HOSPITAL_PLAY_ENV_SPACING
    )

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = HOSPITAL_PLAY_EPISODE_LENGTH_S
        self.scene.env_spacing = HOSPITAL_PLAY_ENV_SPACING
        self.events.reset_obstacles.func = mdp.reset_structured_astar_corridor
        self.events.reset_obstacles.params = {
            **_base_structured_reset_params(),
            "min_inter_obstacle_dist": 0.80,
        }
        self.events.hospital_velocity_resample = None
        self.events.hospital_group_update = None
        _apply_mesh_wall_overrides(
            self,
            corridor_kind=HOSPITAL_PLAY_CORRIDOR_KIND,
            leg_length=HOSPITAL_PLAY_LEG_LENGTH,
            corridor_width=HOSPITAL_PLAY_CORRIDOR_WIDTH,
            wall_thickness=HOSPITAL_PLAY_WALL_THICKNESS,
            dynamic_obstacle_count=HOSPITAL_PLAY_DYNAMIC_OBSTACLE_COUNT,
            tile_size=HOSPITAL_PLAY_ENV_SPACING,
        )
        _configure_play_obstacle_obs(self.observations.policy)
        self.events.navigation_path_update = _structured_path_update_event()
        _set_structured_goal_termination(self, HOSPITAL_PLAY_GOAL_DONE_RADIUS)


@configclass
class Go2wHospitalDepthPlayEnvCfg(Go2wNavDepthRLDistillEnvCfg_PLAY):
    """Play/eval env for the depth-camera student in a hospital-style corridor.

    Same L-corridor layout as ``Go2wHospitalPlayEnvCfg``.  Corridor walls are a
    static TerrainImporter trimesh; all 64 obstacle slots are dynamic actors.
    Load any existing depth-student checkpoint directly.
    """

    scene: DepthObstaclePlayTerrainWallSceneCfg = DepthObstaclePlayTerrainWallSceneCfg(
        num_envs=4, env_spacing=HOSPITAL_PLAY_ENV_SPACING
    )

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = HOSPITAL_PLAY_EPISODE_LENGTH_S
        self.scene.env_spacing = HOSPITAL_PLAY_ENV_SPACING
        self.events.reset_obstacles.func = mdp.reset_structured_astar_corridor
        self.events.reset_obstacles.params = {
            **_base_structured_reset_params(),
            "min_inter_obstacle_dist": 0.80,
        }
        self.events.hospital_velocity_resample = None
        self.events.hospital_group_update = None
        _apply_mesh_wall_overrides(
            self,
            corridor_kind=HOSPITAL_PLAY_CORRIDOR_KIND,
            leg_length=HOSPITAL_PLAY_LEG_LENGTH,
            corridor_width=HOSPITAL_PLAY_CORRIDOR_WIDTH,
            wall_thickness=HOSPITAL_PLAY_WALL_THICKNESS,
            dynamic_obstacle_count=HOSPITAL_PLAY_DYNAMIC_OBSTACLE_COUNT,
            tile_size=HOSPITAL_PLAY_ENV_SPACING,
        )
        self.events.navigation_path_update = _structured_path_update_event()
        _set_structured_goal_termination(self, HOSPITAL_PLAY_GOAL_DONE_RADIUS)


# =============================================================================
# Ward play environments
# =============================================================================


@configclass
class Go2wHospitalWardDepthPlayEnvCfg(Go2wNavDepthRLDistillEnvCfg_PLAY):
    """Depth-student play/eval env in a hospital ward floor layout.

    The ward has a 30 m main corridor with two perpendicular branches (each 10 m),
    forming two T-junctions the robot navigates through.  Branch 1 is a dead-end
    populated with dynamic obstacles that can spill into the main corridor.  Goal
    is the tip of branch 2.  Corridor width is 3.0 m, wall thickness is 0.25 m.
    Dynamic actors use the hospital label palette at 0.7× speed.
    """

    scene: DepthObstaclePlayTerrainWallSceneCfg = DepthObstaclePlayTerrainWallSceneCfg(
        num_envs=4, env_spacing=HOSPITAL_WARD_ENV_SPACING
    )

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = HOSPITAL_WARD_EPISODE_LENGTH_S
        self.scene.env_spacing = HOSPITAL_WARD_ENV_SPACING
        self.events.reset_obstacles.func = mdp.reset_structured_astar_corridor
        self.events.reset_obstacles.params = {
            **_base_structured_reset_params(),
            "min_inter_obstacle_dist": 1.00,
        }
        self.events.hospital_velocity_resample = None
        self.events.hospital_group_update = None
        _apply_mesh_wall_overrides(
            self,
            corridor_kind=HOSPITAL_WARD_CORRIDOR_KIND,
            leg_length=HOSPITAL_WARD_LEG_LENGTH,
            corridor_width=HOSPITAL_WARD_CORRIDOR_WIDTH,
            wall_thickness=HOSPITAL_WARD_WALL_THICKNESS,
            dynamic_obstacle_count=HOSPITAL_WARD_DYNAMIC_OBSTACLE_COUNT,
            speed_scale=0.7,
            min_inter_obstacle_dist=1.00,
            tile_size=HOSPITAL_WARD_ENV_SPACING,
        )
        self.events.navigation_path_update = _structured_path_update_event()
        _set_structured_goal_termination(self, HOSPITAL_WARD_GOAL_DONE_RADIUS)


# =============================================================================
# Full floor play environments
# (constants imported from mdp/hospital/floor.py)
# =============================================================================


@configclass
class Go2wHospitalFloorPlayEnvCfg(Go2wNavTeacherEnvCfg_PLAY):
    """Teacher play/eval env for the full hospital floor.

    Same obs as Go2wHospitalPlayEnvCfg (NavTeacherObsCfg, privileged obstacle
    positions) but uses the full-floor corridor layout.  Walls are a TerrainImporter
    trimesh; all 64 obstacle slots are available for actors.
    """

    scene: ObstaclePlayTerrainWallSceneCfg = ObstaclePlayTerrainWallSceneCfg(
        num_envs=4, env_spacing=HOSPITAL_FLOOR_ENV_SPACING
    )

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = HOSPITAL_FLOOR_EPISODE_LENGTH_S
        self.scene.env_spacing = HOSPITAL_FLOOR_ENV_SPACING
        wall_count = 0  # terrain walls — no obstacle slots used for walls
        self.events.reset_obstacles.func = mdp.reset_structured_astar_corridor
        self.events.reset_obstacles.params = {
            **_base_structured_reset_params(),
            "min_inter_obstacle_dist": 1.05,
        }
        self.events.hospital_velocity_resample = None
        self.events.hospital_group_update = None
        _apply_mesh_wall_overrides(
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
            ramp_asset_name=HOSPITAL_RAMP_ASSET_NAME,
            ramp_local_pose=HOSPITAL_FLOOR_RAMP_LOCAL_POSE,
            ramp_b_asset_name=HOSPITAL_RAMP_B_ASSET_NAME,
            ramp_b_local_pose=HOSPITAL_FLOOR_RAMP_B_LOCAL_POSE,
            robot_start_local_xy=HOSPITAL_FLOOR_ROBOT_START_LOCAL_XY,
            min_inter_obstacle_dist=1.05,
            tile_size=HOSPITAL_FLOOR_ENV_SPACING,
        )
        _configure_play_obstacle_obs(self.observations.policy)
        self.events.navigation_path_update = _structured_path_update_event()
        _set_structured_goal_termination(self, HOSPITAL_FLOOR_GOAL_DONE_RADIUS)


@configclass
class Go2wHospitalFloorDepthPlayEnvCfg(Go2wNavDepthRLDistillEnvCfg_PLAY):
    """Depth-student play/eval env for a combined hospital floor.

    Includes reception queueing, waiting bench occupancy, doorway crossing,
    ward/service flow with a pushed patient gurney, and a mild ramp connector.
    Walls are a TerrainImporter trimesh; all 64 obstacle slots are actors.
    """

    scene: DepthObstaclePlayTerrainWallSceneCfg = DepthObstaclePlayTerrainWallSceneCfg(
        num_envs=4, env_spacing=HOSPITAL_FLOOR_ENV_SPACING
    )

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = HOSPITAL_FLOOR_EPISODE_LENGTH_S
        self.scene.env_spacing = HOSPITAL_FLOOR_ENV_SPACING
        wall_count = 0  # terrain walls — no obstacle slots used for walls
        self.events.reset_obstacles.func = mdp.reset_structured_astar_corridor
        self.events.reset_obstacles.params = {
            **_base_structured_reset_params(),
            "min_inter_obstacle_dist": 1.05,
        }
        self.events.hospital_velocity_resample = None
        self.events.hospital_group_update = None
        _apply_mesh_wall_overrides(
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
            ramp_asset_name=HOSPITAL_RAMP_ASSET_NAME,
            ramp_local_pose=HOSPITAL_FLOOR_RAMP_LOCAL_POSE,
            ramp_b_asset_name=HOSPITAL_RAMP_B_ASSET_NAME,
            ramp_b_local_pose=HOSPITAL_FLOOR_RAMP_B_LOCAL_POSE,
            robot_start_local_xy=HOSPITAL_FLOOR_ROBOT_START_LOCAL_XY,
            min_inter_obstacle_dist=1.05,
            tile_size=HOSPITAL_FLOOR_ENV_SPACING,
        )
        self.events.navigation_path_update = _structured_path_update_event()
        _set_structured_goal_termination(self, HOSPITAL_FLOOR_GOAL_DONE_RADIUS)


# =============================================================================
# Hospital maze teacher play environment
# =============================================================================


@configclass
class Go2wHospitalTeacherPlayEnvCfg(Go2wHospitalTeacherEnvCfg):
    """Play env for the hospital teacher: same scene, obs, and wall generation as training.

    Only play-mode overrides: smaller num_envs, fixed curriculum phase, no obs noise.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.episode_length_s = 120.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_obstacles.params["curriculum_iteration_offset"] = 1100


# =============================================================================
# Hospital Teacher structured corridor play envs (v0)
# =============================================================================


def _configure_hospital_teacher_play_obs(obs_group) -> None:
    """Use hospital teacher actor slots for privileged play observations."""
    obs_group.obstacle_nav_features.params["obstacle_names"] = HOSPITAL_TRAIN_OBSTACLE_NAMES
    obs_group.obstacle_full_geometry.params["obstacle_names"] = HOSPITAL_TRAIN_OBSTACLE_NAMES
    obs_group.enable_corruption = False


@configclass
class Go2wHospitalTeacherLCorridorPlayEnvCfg(Go2wHospitalPlayEnvCfg):
    """L-corridor play env with hospital-teacher privileged observations.

    Same terrain-wall layout and actors as ``Go2wHospitalPlayEnvCfg``.
    """

    observations: NavHospitalTeacherObsCfg = NavHospitalTeacherObsCfg()

    def __post_init__(self):
        super().__post_init__()
        _configure_hospital_teacher_play_obs(self.observations.policy)


@configclass
class Go2wHospitalTeacherWardPlayEnvCfg(Go2wHospitalPlayEnvCfg):
    """Hospital-ward play env with hospital-teacher privileged observations.

    Inherits the nav-teacher L-corridor base and re-applies ward geometry on top.
    """

    scene: ObstaclePlayTerrainWallSceneCfg = ObstaclePlayTerrainWallSceneCfg(
        num_envs=4, env_spacing=HOSPITAL_WARD_ENV_SPACING
    )
    observations: NavHospitalTeacherObsCfg = NavHospitalTeacherObsCfg()

    def __post_init__(self):
        super().__post_init__()
        _apply_mesh_wall_overrides(
            self,
            corridor_kind=HOSPITAL_WARD_CORRIDOR_KIND,
            leg_length=HOSPITAL_WARD_LEG_LENGTH,
            corridor_width=HOSPITAL_WARD_CORRIDOR_WIDTH,
            wall_thickness=HOSPITAL_WARD_WALL_THICKNESS,
            dynamic_obstacle_count=HOSPITAL_WARD_DYNAMIC_OBSTACLE_COUNT,
            speed_scale=0.7,
            min_inter_obstacle_dist=1.00,
            tile_size=HOSPITAL_WARD_ENV_SPACING,
        )
        self.episode_length_s = HOSPITAL_WARD_EPISODE_LENGTH_S
        self.scene.env_spacing = HOSPITAL_WARD_ENV_SPACING
        _set_structured_goal_termination(self, HOSPITAL_WARD_GOAL_DONE_RADIUS)
        _configure_hospital_teacher_play_obs(self.observations.policy)


@configclass
class Go2wHospitalTeacherFloorPlayEnvCfg(Go2wHospitalPlayEnvCfg):
    """Full-floor play env with hospital-teacher privileged observations.

    Inherits the nav-teacher L-corridor base and re-applies full-floor geometry,
    ramps, semantic labels, and actor overrides on top.
    """

    scene: ObstaclePlayTerrainWallSceneCfg = ObstaclePlayTerrainWallSceneCfg(
        num_envs=4, env_spacing=HOSPITAL_FLOOR_ENV_SPACING
    )
    observations: NavHospitalTeacherObsCfg = NavHospitalTeacherObsCfg()

    def __post_init__(self):
        super().__post_init__()
        wall_count = 0
        _apply_mesh_wall_overrides(
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
            ramp_asset_name=HOSPITAL_RAMP_ASSET_NAME,
            ramp_local_pose=HOSPITAL_FLOOR_RAMP_LOCAL_POSE,
            ramp_b_asset_name=HOSPITAL_RAMP_B_ASSET_NAME,
            ramp_b_local_pose=HOSPITAL_FLOOR_RAMP_B_LOCAL_POSE,
            robot_start_local_xy=HOSPITAL_FLOOR_ROBOT_START_LOCAL_XY,
            min_inter_obstacle_dist=1.05,
            tile_size=HOSPITAL_FLOOR_ENV_SPACING,
        )
        self.episode_length_s = HOSPITAL_FLOOR_EPISODE_LENGTH_S
        self.scene.env_spacing = HOSPITAL_FLOOR_ENV_SPACING
        _set_structured_goal_termination(self, HOSPITAL_FLOOR_GOAL_DONE_RADIUS)
        _configure_hospital_teacher_play_obs(self.observations.policy)


# =============================================================================
# Hospital Teacher structured corridor play envs (v1 training lidar)
# =============================================================================


def _make_hospital_teacher_lidar_cfg() -> MultiMeshRayCasterCfg:
    """Training lidar for hospital maze teacher play."""
    return MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=HOSPITAL_RAYCAST_SENSOR_OFFSET),
        ray_alignment="yaw",
        pattern_cfg=_sensor_patterns.LidarPatternCfg(
            channels=HOSPITAL_RAYCAST_CHANNELS,
            vertical_fov_range=HOSPITAL_RAYCAST_VERTICAL_FOV,
            horizontal_fov_range=HOSPITAL_RAYCAST_HORIZONTAL_FOV,
            horizontal_res=HOSPITAL_RAYCAST_HORIZONTAL_RES,
        ),
        max_distance=HOSPITAL_RAYCAST_MAX_DISTANCE,
        mesh_prim_paths=[
            "/World/terrain",
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/obstacle_.*",
                track_mesh_transforms=True,
                is_shared=True,
            ),
        ],
        debug_vis=False,
    )


@configclass
class Go2wHospitalTeacherLCorridorPlayEnvCfgV1(Go2wHospitalTeacherLCorridorPlayEnvCfg):
    """L-corridor play env with hospital teacher training lidar."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.lidar = _make_hospital_teacher_lidar_cfg()


@configclass
class Go2wHospitalTeacherWardPlayEnvCfgV1(Go2wHospitalTeacherWardPlayEnvCfg):
    """Ward play env with hospital teacher training lidar."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.lidar = _make_hospital_teacher_lidar_cfg()


@configclass
class Go2wHospitalTeacherFloorPlayEnvCfgV1(Go2wHospitalTeacherFloorPlayEnvCfg):
    """Full-floor play env with hospital teacher training lidar."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.lidar = _make_hospital_teacher_lidar_cfg()


# =============================================================================
# Depth student ablation floor play environments
# (long-history abl-D, sparse abl-B, 4-cam abl-A)
# =============================================================================


@configclass
class Go2wHospitalFloorLongHistDepthPlayEnvCfg(Go2wNavDepthLongHistRLDistillEnvCfg_PLAY):
    """Full-floor depth play env — long-history abl-D (8 dense frames, faster camera).

    Same floor layout, ramps, and actors as ``Go2wHospitalFloorDepthPlayEnvCfg`` but
    uses the 8-frame dense-history observation (NavDepthRLDistillLongHistObsCfg) for
    the abl-D ablation checkpoint.
    """

    scene: DepthObstaclePlayTerrainWallSceneCfg = DepthObstaclePlayTerrainWallSceneCfg(
        num_envs=4, env_spacing=HOSPITAL_FLOOR_ENV_SPACING
    )

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = HOSPITAL_FLOOR_EPISODE_LENGTH_S
        self.scene.env_spacing = HOSPITAL_FLOOR_ENV_SPACING
        wall_count = 0
        self.events.reset_obstacles.func = mdp.reset_structured_astar_corridor
        self.events.reset_obstacles.params = {
            **_base_structured_reset_params(),
            "min_inter_obstacle_dist": 1.05,
        }
        self.events.hospital_velocity_resample = None
        self.events.hospital_group_update = None
        _apply_mesh_wall_overrides(
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
            ramp_asset_name=HOSPITAL_RAMP_ASSET_NAME,
            ramp_local_pose=HOSPITAL_FLOOR_RAMP_LOCAL_POSE,
            ramp_b_asset_name=HOSPITAL_RAMP_B_ASSET_NAME,
            ramp_b_local_pose=HOSPITAL_FLOOR_RAMP_B_LOCAL_POSE,
            robot_start_local_xy=HOSPITAL_FLOOR_ROBOT_START_LOCAL_XY,
            min_inter_obstacle_dist=1.05,
            tile_size=HOSPITAL_FLOOR_ENV_SPACING,
        )
        self.events.navigation_path_update = _structured_path_update_event()
        _set_structured_goal_termination(self, HOSPITAL_FLOOR_GOAL_DONE_RADIUS)


@configclass
class Go2wHospitalFloorSparseDepthPlayEnvCfg(Go2wNavDepthSparseRLDistillEnvCfg_PLAY):
    """Full-floor depth play env — sparse-history abl-B (stride-5 camera, slower update).

    Same floor layout, ramps, and actors as ``Go2wHospitalFloorDepthPlayEnvCfg`` but
    uses the stride-5 camera update schedule of the abl-B ablation checkpoint.
    """

    scene: DepthObstaclePlayTerrainWallSceneCfg = DepthObstaclePlayTerrainWallSceneCfg(
        num_envs=4, env_spacing=HOSPITAL_FLOOR_ENV_SPACING
    )

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = HOSPITAL_FLOOR_EPISODE_LENGTH_S
        self.scene.env_spacing = HOSPITAL_FLOOR_ENV_SPACING
        wall_count = 0
        self.events.reset_obstacles.func = mdp.reset_structured_astar_corridor
        self.events.reset_obstacles.params = {
            **_base_structured_reset_params(),
            "min_inter_obstacle_dist": 1.05,
        }
        self.events.hospital_velocity_resample = None
        self.events.hospital_group_update = None
        _apply_mesh_wall_overrides(
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
            ramp_asset_name=HOSPITAL_RAMP_ASSET_NAME,
            ramp_local_pose=HOSPITAL_FLOOR_RAMP_LOCAL_POSE,
            ramp_b_asset_name=HOSPITAL_RAMP_B_ASSET_NAME,
            ramp_b_local_pose=HOSPITAL_FLOOR_RAMP_B_LOCAL_POSE,
            robot_start_local_xy=HOSPITAL_FLOOR_ROBOT_START_LOCAL_XY,
            min_inter_obstacle_dist=1.05,
            tile_size=HOSPITAL_FLOOR_ENV_SPACING,
        )
        self.events.navigation_path_update = _structured_path_update_event()
        _set_structured_goal_termination(self, HOSPITAL_FLOOR_GOAL_DONE_RADIUS)


@configclass
class Go2wHospitalFloorMultiCamDepthPlayEnvCfg(Go2wNavDepthMultiCamRLDistillEnvCfg_PLAY):
    """Full-floor depth play env — 4-camera abl-A (360° FOV, 12 input channels).

    Same floor layout, ramps, and actors as ``Go2wHospitalFloorDepthPlayEnvCfg`` but
    uses the 4-camera observation (NavDepthRLDistillMultiCamObsCfg) for the abl-A
    ablation checkpoint.  Terrain and ramp prims are added to all four cameras.
    """

    scene: DepthObstacleMultiCamPlayTerrainWallSceneCfg = DepthObstacleMultiCamPlayTerrainWallSceneCfg(
        num_envs=4, env_spacing=HOSPITAL_FLOOR_ENV_SPACING
    )

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = HOSPITAL_FLOOR_EPISODE_LENGTH_S
        self.scene.env_spacing = HOSPITAL_FLOOR_ENV_SPACING
        wall_count = 0
        self.events.reset_obstacles.func = mdp.reset_structured_astar_corridor
        self.events.reset_obstacles.params = {
            **_base_structured_reset_params(),
            "min_inter_obstacle_dist": 1.05,
        }
        self.events.hospital_velocity_resample = None
        self.events.hospital_group_update = None
        _apply_mesh_wall_overrides(
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
            ramp_asset_name=HOSPITAL_RAMP_ASSET_NAME,
            ramp_local_pose=HOSPITAL_FLOOR_RAMP_LOCAL_POSE,
            ramp_b_asset_name=HOSPITAL_RAMP_B_ASSET_NAME,
            ramp_b_local_pose=HOSPITAL_FLOOR_RAMP_B_LOCAL_POSE,
            robot_start_local_xy=HOSPITAL_FLOOR_ROBOT_START_LOCAL_XY,
            min_inter_obstacle_dist=1.05,
            tile_size=HOSPITAL_FLOOR_ENV_SPACING,
        )
        self.events.navigation_path_update = _structured_path_update_event()
        _set_structured_goal_termination(self, HOSPITAL_FLOOR_GOAL_DONE_RADIUS)
