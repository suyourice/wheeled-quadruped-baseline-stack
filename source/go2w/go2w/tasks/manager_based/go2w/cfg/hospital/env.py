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
    Go2wHospitalDepthRLDistillEnvCfg,
    Go2wHospitalDepthLongHistRLDistillEnvCfg,
    Go2wHospitalDepthSparseRLDistillEnvCfg,
    Go2wHospitalDepthMultiCamRLDistillEnvCfg,
    Go2wNavDepthRLDistillEnvCfg_PLAY,
    Go2wNavDepthLongHistRLDistillEnvCfg_PLAY,
    Go2wNavDepthSparseRLDistillEnvCfg_PLAY,
    Go2wNavDepthMultiCamRLDistillEnvCfg_PLAY,
    Go2wNavTeacherEnvCfg_PLAY,
    HospitalTeacherEvalSceneCfg,
    HospitalTeacherDepthEvalSceneCfg,
    HospitalTeacherMultiCamDepthEvalSceneCfg,
    HospitalMazeEvalSceneCfg,
    HospitalMazeEvalDepthSceneCfg,
    HospitalMazeEvalMultiCamDepthSceneCfg,
    ObstaclePlayTerrainWallSceneCfg,
    DepthObstaclePlayTerrainWallSceneCfg,
    DepthObstacleMultiCamPlayTerrainWallSceneCfg,
    _configure_play_obstacle_obs,
    make_play_obstacle_cfg,
)
from ..navigation.observations import NavHospitalTeacherObsCfg
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

# Default hospital corridor layout for play testing:
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


def _lcorridor_wall_kwargs() -> dict:
    """Wall geometry for the L-corridor play envs."""
    return dict(
        corridor_kind=HOSPITAL_PLAY_CORRIDOR_KIND,
        leg_length=HOSPITAL_PLAY_LEG_LENGTH,
        corridor_width=HOSPITAL_PLAY_CORRIDOR_WIDTH,
        wall_thickness=HOSPITAL_PLAY_WALL_THICKNESS,
        dynamic_obstacle_count=HOSPITAL_PLAY_DYNAMIC_OBSTACLE_COUNT,
        tile_size=HOSPITAL_PLAY_ENV_SPACING,
    )


def _ward_wall_kwargs() -> dict:
    """Wall geometry for the hospital-ward play envs."""
    return dict(
        corridor_kind=HOSPITAL_WARD_CORRIDOR_KIND,
        leg_length=HOSPITAL_WARD_LEG_LENGTH,
        corridor_width=HOSPITAL_WARD_CORRIDOR_WIDTH,
        wall_thickness=HOSPITAL_WARD_WALL_THICKNESS,
        dynamic_obstacle_count=HOSPITAL_WARD_DYNAMIC_OBSTACLE_COUNT,
        speed_scale=0.7,
        min_inter_obstacle_dist=1.00,
        tile_size=HOSPITAL_WARD_ENV_SPACING,
    )


def _floor_wall_kwargs(wall_count: int = 0) -> dict:
    """Wall geometry, semantic actors, and ramps for the full-floor play envs."""
    return dict(
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


def _apply_structured_play(
    cfg,
    *,
    episode_length_s: float,
    env_spacing: float,
    min_inter_obstacle_dist: float,
    goal_done_radius: float,
    wall_kwargs: dict,
) -> None:
    """Configure a play env for structured A*-corridor navigation.

    Applies the sequence shared by every structured hospital play env:
    episode length, env spacing, structured reset event, terrain walls,
    per-step waypoint update, and the final-goal termination.
    """
    cfg.episode_length_s = episode_length_s
    cfg.scene.env_spacing = env_spacing
    cfg.events.reset_obstacles.func = mdp.reset_structured_astar_corridor
    cfg.events.reset_obstacles.params = {
        **_base_structured_reset_params(),
        "min_inter_obstacle_dist": min_inter_obstacle_dist,
    }
    cfg.events.hospital_velocity_resample = None
    cfg.events.hospital_group_update = None
    _apply_mesh_wall_overrides(cfg, **wall_kwargs)
    cfg.events.navigation_path_update = _structured_path_update_event()
    _set_structured_goal_termination(cfg, goal_done_radius)


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

    def _structured_play_spec(self) -> dict:
        """Structured-play timing and geometry; geometry subclasses override."""
        return dict(
            episode_length_s=HOSPITAL_PLAY_EPISODE_LENGTH_S,
            env_spacing=HOSPITAL_PLAY_ENV_SPACING,
            min_inter_obstacle_dist=0.80,
            goal_done_radius=HOSPITAL_PLAY_GOAL_DONE_RADIUS,
            wall_kwargs=_lcorridor_wall_kwargs(),
        )

    def __post_init__(self):
        super().__post_init__()
        _apply_structured_play(self, **self._structured_play_spec())
        _configure_play_obstacle_obs(self.observations.policy)


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
        _apply_structured_play(
            self,
            episode_length_s=HOSPITAL_PLAY_EPISODE_LENGTH_S,
            env_spacing=HOSPITAL_PLAY_ENV_SPACING,
            min_inter_obstacle_dist=0.80,
            goal_done_radius=HOSPITAL_PLAY_GOAL_DONE_RADIUS,
            wall_kwargs=_lcorridor_wall_kwargs(),
        )


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
        _apply_structured_play(
            self,
            episode_length_s=HOSPITAL_WARD_EPISODE_LENGTH_S,
            env_spacing=HOSPITAL_WARD_ENV_SPACING,
            min_inter_obstacle_dist=1.00,
            goal_done_radius=HOSPITAL_WARD_GOAL_DONE_RADIUS,
            wall_kwargs=_ward_wall_kwargs(),
        )


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
        _apply_structured_play(
            self,
            episode_length_s=HOSPITAL_FLOOR_EPISODE_LENGTH_S,
            env_spacing=HOSPITAL_FLOOR_ENV_SPACING,
            min_inter_obstacle_dist=1.05,
            goal_done_radius=HOSPITAL_FLOOR_GOAL_DONE_RADIUS,
            wall_kwargs=_floor_wall_kwargs(),
        )
        _configure_play_obstacle_obs(self.observations.policy)


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
        _apply_structured_play(
            self,
            episode_length_s=HOSPITAL_FLOOR_EPISODE_LENGTH_S,
            env_spacing=HOSPITAL_FLOOR_ENV_SPACING,
            min_inter_obstacle_dist=1.05,
            goal_done_radius=HOSPITAL_FLOOR_GOAL_DONE_RADIUS,
            wall_kwargs=_floor_wall_kwargs(),
        )


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

    Inherits the nav-teacher play base and swaps in ward geometry via
    ``_structured_play_spec``.
    """

    scene: ObstaclePlayTerrainWallSceneCfg = ObstaclePlayTerrainWallSceneCfg(
        num_envs=4, env_spacing=HOSPITAL_WARD_ENV_SPACING
    )
    observations: NavHospitalTeacherObsCfg = NavHospitalTeacherObsCfg()

    def _structured_play_spec(self) -> dict:
        return dict(
            episode_length_s=HOSPITAL_WARD_EPISODE_LENGTH_S,
            env_spacing=HOSPITAL_WARD_ENV_SPACING,
            min_inter_obstacle_dist=1.00,
            goal_done_radius=HOSPITAL_WARD_GOAL_DONE_RADIUS,
            wall_kwargs=_ward_wall_kwargs(),
        )

    def __post_init__(self):
        super().__post_init__()
        _configure_hospital_teacher_play_obs(self.observations.policy)


@configclass
class Go2wHospitalTeacherFloorPlayEnvCfg(Go2wHospitalPlayEnvCfg):
    """Full-floor play env with hospital-teacher privileged observations.

    Inherits the nav-teacher play base and swaps in full-floor geometry,
    ramps, semantic labels, and actor overrides via ``_structured_play_spec``.
    """

    scene: ObstaclePlayTerrainWallSceneCfg = ObstaclePlayTerrainWallSceneCfg(
        num_envs=4, env_spacing=HOSPITAL_FLOOR_ENV_SPACING
    )
    observations: NavHospitalTeacherObsCfg = NavHospitalTeacherObsCfg()

    def _structured_play_spec(self) -> dict:
        return dict(
            episode_length_s=HOSPITAL_FLOOR_EPISODE_LENGTH_S,
            env_spacing=HOSPITAL_FLOOR_ENV_SPACING,
            min_inter_obstacle_dist=1.05,
            goal_done_radius=HOSPITAL_FLOOR_GOAL_DONE_RADIUS,
            wall_kwargs=_floor_wall_kwargs(),
        )

    def __post_init__(self):
        super().__post_init__()
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
        _apply_structured_play(
            self,
            episode_length_s=HOSPITAL_FLOOR_EPISODE_LENGTH_S,
            env_spacing=HOSPITAL_FLOOR_ENV_SPACING,
            min_inter_obstacle_dist=1.05,
            goal_done_radius=HOSPITAL_FLOOR_GOAL_DONE_RADIUS,
            wall_kwargs=_floor_wall_kwargs(),
        )


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
        _apply_structured_play(
            self,
            episode_length_s=HOSPITAL_FLOOR_EPISODE_LENGTH_S,
            env_spacing=HOSPITAL_FLOOR_ENV_SPACING,
            min_inter_obstacle_dist=1.05,
            goal_done_radius=HOSPITAL_FLOOR_GOAL_DONE_RADIUS,
            wall_kwargs=_floor_wall_kwargs(),
        )


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
        _apply_structured_play(
            self,
            episode_length_s=HOSPITAL_FLOOR_EPISODE_LENGTH_S,
            env_spacing=HOSPITAL_FLOOR_ENV_SPACING,
            min_inter_obstacle_dist=1.05,
            goal_done_radius=HOSPITAL_FLOOR_GOAL_DONE_RADIUS,
            wall_kwargs=_floor_wall_kwargs(),
        )


# =============================================================================
# Hospital maze eval environments (training-isolated, episode_length_s=220s)
#
# Uses the same 5x5 maze terrain as training but adds 4 extra small-obstacle
# slots (fallen_object x2, iv_pole x1, trash_bin x1) for richer eval coverage.
# HOSPITAL_TRAIN_* constants and training env classes are untouched.
#
# Task IDs:
#   Nav-HospitalMaze-Teacher-Eval-Static-Go2w-v0
#   Nav-HospitalMaze-Teacher-Eval-Dynamic-Go2w-v0
#   Nav-HospitalMaze-Distill-Depth-Eval-Static-Go2w-v0
#   Nav-HospitalMaze-Distill-Depth-Eval-Dynamic-Go2w-v0
#   Nav-HospitalMaze-Distill-Depth-LongHist-Eval-Static-Go2w-v0
#   Nav-HospitalMaze-Distill-Depth-LongHist-Eval-Dynamic-Go2w-v0
#   Nav-HospitalMaze-Distill-Depth-Sparse-Eval-Static-Go2w-v0
#   Nav-HospitalMaze-Distill-Depth-Sparse-Eval-Dynamic-Go2w-v0
#   Nav-HospitalMaze-Distill-Depth-4Cam-Eval-Static-Go2w-v0
#   Nav-HospitalMaze-Distill-Depth-4Cam-Eval-Dynamic-Go2w-v0
# =============================================================================

HOSPITAL_MAZE_EVAL_EPISODE_LENGTH_S = 220.0
HOSPITAL_MAZE_EVAL_NUM_ENVS = 8
# Full-density phase offset: matches hospital teacher play (iteration 900+ = phase 4).
_HOSPITAL_MAZE_EVAL_CURRICULUM_OFFSET = 1100


def _configure_maze_eval_env(cfg) -> None:
    """Apply eval overrides: 20 obstacle slots, 220 s episodes.

    Long-horizon scenario: like training, the episode never terminates on goal
    reached — goal_reached_and_resample keeps sampling new routes for the full
    220 s so path_progress_mean reflects sustained multi-route exposure, not a
    single attempt. Callers that need single-route success/SPL semantics (the
    maze_success protocol) add a termination on top via play.py's
    --terminate_on_final_goal, not here — adding it unconditionally in this
    shared function would silently convert maze_static/maze_dynamic into
    single-route evaluations too.

    Teacher policy obs: training-time noise ON, matching
        NavHospitalTeacherObsCfg.PolicyCfg (enable_corruption=True during
        training). A deterministic teacher oracle would be a separate,
        explicitly labelled diagnostic rather than a fair ablation reference.
    Student student_state: noise ON (matches training conditions; proprio IMU
        noise was enabled during distillation and should remain on for realistic eval).
    Student student_depth: noise OFF (per-pixel depth noise is not a valid model
        of real depth camera noise; distillation handles sim-to-real gap).
    """
    cfg.scene.num_envs = HOSPITAL_MAZE_EVAL_NUM_ENVS
    cfg.episode_length_s = HOSPITAL_MAZE_EVAL_EPISODE_LENGTH_S
    cfg.events.reset_obstacles.params.update({
        "obstacle_names": HOSPITAL_MAZE_EVAL_OBSTACLE_NAMES,
        "curriculum_iteration_offset": _HOSPITAL_MAZE_EVAL_CURRICULUM_OFFSET,
        "obstacle_widths": HOSPITAL_MAZE_EVAL_OBSTACLE_WIDTHS,
        "obstacle_depths": HOSPITAL_MAZE_EVAL_OBSTACLE_DEPTHS,
        "obstacle_center_zs": HOSPITAL_MAZE_EVAL_OBSTACLE_CENTER_ZS,
        "obstacle_heights": HOSPITAL_MAZE_EVAL_OBSTACLE_HEIGHTS,
        "obstacle_shape_ids": HOSPITAL_MAZE_EVAL_OBSTACLE_SHAPE_IDS,
        "obstacle_class_ids": HOSPITAL_MAZE_EVAL_OBSTACLE_CLASS_IDS,
        "obstacle_priorities": HOSPITAL_MAZE_EVAL_OBSTACLE_PRIORITIES,
        "actor_count_override": HOSPITAL_MAZE_EVAL_ACTOR_SLOTS,
    })
    obs = cfg.observations
    if hasattr(obs, "policy"):
        obs.policy.enable_corruption = True
    if hasattr(obs, "student_depth"):
        obs.student_depth.enable_corruption = False

    # Retarget obstacle-aware reward terms to the 20-slot eval set so that
    # priority/active tensors remain consistent with the expanded scene.
    # Note: this list intentionally differs from
    # _retarget_nav_rewards_to_play_obstacles (cfg/navigation/env.py) — the
    # maze-eval reward set has hospital_centerline but no open-path terms.
    # When adding a new obstacle-aware reward term, update both lists.
    _eval_obs_names = list(HOSPITAL_MAZE_EVAL_OBSTACLE_NAMES)
    for _rname in (
        "nav_lateral_escape",
        "nav_dense_recovery",
        "hospital_centerline",
        "nav_clearance",
        "obstacle_ttc",
        "nav_grazing",
    ):
        _term = getattr(cfg.rewards, _rname, None)
        if _term is not None and _term.params and "obstacle_names" in _term.params:
            _term.params["obstacle_names"] = _eval_obs_names

    # Retarget privileged obstacle observations to the 20-slot eval set.
    # Two groups carry this: "policy" for the teacher task (drives its
    # actions — without this fix the teacher's oracle only tracks the 16
    # training slots and the 4 extra eval obstacles are invisible to it),
    # and "teacher" for the depth-student distillation tasks (the privileged
    # distillation-loss target; unused for action selection at eval/play
    # time since play.py loads student weights only, but still computed
    # every step by the observation manager — left at 16 slots it silently
    # mismatches the 20-slot scene, which the preflight check correctly
    # treats as a hard error). Output dims are unaffected either way: nav
    # features are a fixed 16D aggregate and geometry pads/truncates to
    # num_slots regardless of how many named slots are searched.
    for _group_name in ("policy", "teacher"):
        _group = getattr(obs, _group_name, None)
        if _group is None:
            continue
        for _oname in ("obstacle_nav_features", "obstacle_full_geometry"):
            _oterm = getattr(_group, _oname, None)
            if _oterm is not None and _oterm.params and "obstacle_names" in _oterm.params:
                _oterm.params["obstacle_names"] = _eval_obs_names


def _add_maze_eval_dynamic_events(cfg) -> None:
    """Add per-step pose advancement for the dynamic-obstacle eval version.

    kinematic rigid bodies only move when write_root_pose_to_sim is called each step.
    move_hospital_dynamic_obstacles runs at interval_range_s=(0.0, 0.0) to advance
    poses every step; it internally resamples velocity directions every 3-6 s.
    """
    cfg.events.hospital_dynamic_motion = EventTerm(
        func=_hospital_events.move_hospital_dynamic_obstacles,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "obstacle_names": list(HOSPITAL_MAZE_EVAL_OBSTACLE_NAMES),
            "obstacle_labels": list(HOSPITAL_MAZE_EVAL_OBSTACLE_LABELS),
            "obstacle_center_zs": tuple(HOSPITAL_MAZE_EVAL_OBSTACLE_CENTER_ZS),
            "speed_scale": 0.7,
            "velocity_resample_interval_range": (3.0, 6.0),
            "active_distance": 100.0,
        },
    )


# -----------------------------------------------------------------------------
# Static versions (no moving obstacles - mirrors training distribution)
# -----------------------------------------------------------------------------


@configclass
class Go2wHospitalMazeStaticEvalTeacherEnvCfg(Go2wHospitalTeacherEnvCfg):
    """Hospital maze eval env for the teacher policy - 20 obstacles, 220 s."""

    scene: HospitalMazeEvalSceneCfg = HospitalMazeEvalSceneCfg(
        num_envs=HOSPITAL_MAZE_EVAL_NUM_ENVS, env_spacing=48.0
    )

    def __post_init__(self):
        super().__post_init__()
        _configure_maze_eval_env(self)


@configclass
class Go2wHospitalMazeStaticEvalBaselineEnvCfg(Go2wHospitalDepthRLDistillEnvCfg):
    """Maze eval env for baseline depth student - 20 obstacles, 220 s."""

    scene: HospitalMazeEvalDepthSceneCfg = HospitalMazeEvalDepthSceneCfg(
        num_envs=HOSPITAL_MAZE_EVAL_NUM_ENVS, env_spacing=48.0
    )

    def __post_init__(self):
        super().__post_init__()
        _configure_maze_eval_env(self)


@configclass
class Go2wHospitalMazeStaticEvalLongHistEnvCfg(Go2wHospitalDepthLongHistRLDistillEnvCfg):
    """Maze eval env for long-history depth student - 20 obstacles, 220 s."""

    scene: HospitalMazeEvalDepthSceneCfg = HospitalMazeEvalDepthSceneCfg(
        num_envs=HOSPITAL_MAZE_EVAL_NUM_ENVS, env_spacing=48.0
    )

    def __post_init__(self):
        super().__post_init__()
        _configure_maze_eval_env(self)


@configclass
class Go2wHospitalMazeStaticEvalSparseEnvCfg(Go2wHospitalDepthSparseRLDistillEnvCfg):
    """Maze eval env for sparse-history depth student - 20 obstacles, 220 s."""

    scene: HospitalMazeEvalDepthSceneCfg = HospitalMazeEvalDepthSceneCfg(
        num_envs=HOSPITAL_MAZE_EVAL_NUM_ENVS, env_spacing=48.0
    )

    def __post_init__(self):
        super().__post_init__()
        _configure_maze_eval_env(self)


@configclass
class Go2wHospitalMazeStaticEvalMultiCamEnvCfg(Go2wHospitalDepthMultiCamRLDistillEnvCfg):
    """Maze eval env for 4-camera depth student - 20 obstacles, 220 s."""

    scene: HospitalMazeEvalMultiCamDepthSceneCfg = HospitalMazeEvalMultiCamDepthSceneCfg(
        num_envs=HOSPITAL_MAZE_EVAL_NUM_ENVS, env_spacing=48.0
    )

    def __post_init__(self):
        super().__post_init__()
        _configure_maze_eval_env(self)


# -----------------------------------------------------------------------------
# Dynamic versions (label-driven velocity resample every 3-6 s)
# -----------------------------------------------------------------------------


@configclass
class Go2wHospitalMazeDynamicEvalTeacherEnvCfg(Go2wHospitalMazeStaticEvalTeacherEnvCfg):
    """Maze eval env (teacher) with label-driven obstacle velocity resample."""

    def __post_init__(self):
        super().__post_init__()
        _add_maze_eval_dynamic_events(self)


@configclass
class Go2wHospitalMazeDynamicEvalBaselineEnvCfg(Go2wHospitalMazeStaticEvalBaselineEnvCfg):
    """Maze eval env (baseline depth) with label-driven obstacle velocity resample."""

    def __post_init__(self):
        super().__post_init__()
        _add_maze_eval_dynamic_events(self)


@configclass
class Go2wHospitalMazeDynamicEvalLongHistEnvCfg(Go2wHospitalMazeStaticEvalLongHistEnvCfg):
    """Maze eval env (long-hist depth) with label-driven obstacle velocity resample."""

    def __post_init__(self):
        super().__post_init__()
        _add_maze_eval_dynamic_events(self)


@configclass
class Go2wHospitalMazeDynamicEvalSparseEnvCfg(Go2wHospitalMazeStaticEvalSparseEnvCfg):
    """Maze eval env (sparse depth) with label-driven obstacle velocity resample."""

    def __post_init__(self):
        super().__post_init__()
        _add_maze_eval_dynamic_events(self)


@configclass
class Go2wHospitalMazeDynamicEvalMultiCamEnvCfg(Go2wHospitalMazeStaticEvalMultiCamEnvCfg):
    """Maze eval env (4-cam depth) with label-driven obstacle velocity resample."""

    def __post_init__(self):
        super().__post_init__()
        _add_maze_eval_dynamic_events(self)


# -----------------------------------------------------------------------------
# Training-distribution eval (16-slot training scene, last curriculum phase,
# no actor_count_override → training schedule caps at 12 obstacles)
# -----------------------------------------------------------------------------


def _configure_maze_train_dist_eval_env(cfg) -> None:
    """Apply eval overrides for training-distribution conditions.

    Uses the 16-slot training scene without actor_count_override so obstacle
    placement follows the last training phase (max 12 at 2.0 m spacing).
    Long-horizon, uninterrupted-on-goal semantics and noise policy match
    _configure_maze_eval_env exactly (teacher ON, student_state ON,
    student_depth OFF) so maze_train is a same-conditions baseline for
    maze_static/maze_dynamic rather than a separately-configured scenario.
    """
    cfg.scene.num_envs = HOSPITAL_MAZE_EVAL_NUM_ENVS
    cfg.episode_length_s = HOSPITAL_MAZE_EVAL_EPISODE_LENGTH_S
    cfg.events.reset_obstacles.params.update({
        "curriculum_iteration_offset": _HOSPITAL_MAZE_EVAL_CURRICULUM_OFFSET,
    })
    obs = cfg.observations
    if hasattr(obs, "policy"):
        obs.policy.enable_corruption = True
    if hasattr(obs, "student_depth"):
        obs.student_depth.enable_corruption = False


@configclass
class Go2wHospitalMazeTrainDistEvalTeacherEnvCfg(Go2wHospitalTeacherEnvCfg):
    """Hospital maze eval for teacher: 16-slot training scene, last curriculum phase."""

    scene: HospitalTeacherEvalSceneCfg = HospitalTeacherEvalSceneCfg(
        num_envs=HOSPITAL_MAZE_EVAL_NUM_ENVS, env_spacing=48.0
    )

    def __post_init__(self):
        super().__post_init__()
        _configure_maze_train_dist_eval_env(self)


@configclass
class Go2wHospitalMazeTrainDistEvalBaselineEnvCfg(Go2wHospitalDepthRLDistillEnvCfg):
    """Hospital maze eval for baseline depth student: 16-slot training scene."""

    scene: HospitalTeacherDepthEvalSceneCfg = HospitalTeacherDepthEvalSceneCfg(
        num_envs=HOSPITAL_MAZE_EVAL_NUM_ENVS, env_spacing=48.0
    )

    def __post_init__(self):
        super().__post_init__()
        _configure_maze_train_dist_eval_env(self)


@configclass
class Go2wHospitalMazeTrainDistEvalLongHistEnvCfg(Go2wHospitalDepthLongHistRLDistillEnvCfg):
    """Hospital maze eval for long-history depth student: 16-slot training scene."""

    scene: HospitalTeacherDepthEvalSceneCfg = HospitalTeacherDepthEvalSceneCfg(
        num_envs=HOSPITAL_MAZE_EVAL_NUM_ENVS, env_spacing=48.0
    )

    def __post_init__(self):
        super().__post_init__()
        _configure_maze_train_dist_eval_env(self)


@configclass
class Go2wHospitalMazeTrainDistEvalSparseEnvCfg(Go2wHospitalDepthSparseRLDistillEnvCfg):
    """Hospital maze eval for sparse-history depth student: 16-slot training scene."""

    scene: HospitalTeacherDepthEvalSceneCfg = HospitalTeacherDepthEvalSceneCfg(
        num_envs=HOSPITAL_MAZE_EVAL_NUM_ENVS, env_spacing=48.0
    )

    def __post_init__(self):
        super().__post_init__()
        _configure_maze_train_dist_eval_env(self)


@configclass
class Go2wHospitalMazeTrainDistEvalMultiCamEnvCfg(Go2wHospitalDepthMultiCamRLDistillEnvCfg):
    """Hospital maze eval for 4-camera depth student: 16-slot training scene."""

    scene: HospitalTeacherMultiCamDepthEvalSceneCfg = HospitalTeacherMultiCamDepthEvalSceneCfg(
        num_envs=HOSPITAL_MAZE_EVAL_NUM_ENVS, env_spacing=48.0
    )

    def __post_init__(self):
        super().__post_init__()
        _configure_maze_train_dist_eval_env(self)
