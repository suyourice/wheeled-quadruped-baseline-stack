# Scripts

This directory contains the command-line entry points for training, policy playback, environment inspection, and validation.

All commands below are intended to be executed from the repository root using the Isaac Lab Python launcher:

```bash
./isaaclab.sh -p <script> [options]
```

## Overview

| Script | Purpose |
|---|---|
| `list_envs.py` | List registered Go2-W task IDs |
| `rsl_rl/train.py` | Train locomotion, navigation-teacher, and distillation policies |
| `rsl_rl/play_cmd.py` | Play a locomotion policy with fixed or random velocity commands |
| `rsl_rl/play.py` | Play navigation policies and configure navigation scenarios |
| `preflight_validation.py` | Check validation task configurations without running rollouts |
| `run_validation.py` | Run the configured policy and scenario validation suite |
| `plot_validation.py` | Aggregate validation manifests and generate summary files |
| `slurm/` | Cluster-specific training and validation launch scripts |

See the [root README](../README.md) for the overall system architecture.

## Checkpoint Paths

Pretrained checkpoints are not included in the repository.

Use local paths when running training or playback:

```bash
export LLC_CKPT="/path/to/locomotion_checkpoint.pt"
export TEACHER_CKPT="/path/to/navigation_teacher_checkpoint.pt"
export STUDENT_CKPT="/path/to/depth_student_checkpoint.pt"
```

The navigation stack has the following dependency:

```text
Locomotion checkpoint
    ├── required for teacher training
    ├── required for student training
    └── required for navigation playback

Teacher checkpoint
    └── required for student distillation
```

## Listing Registered Tasks

List all registered Go2-W environments:

```bash
./isaaclab.sh -p scripts/list_envs.py
```

Filter task IDs by keyword:

```bash
./isaaclab.sh -p scripts/list_envs.py --keyword HospitalMaze
```

Other useful filters include:

```bash
./isaaclab.sh -p scripts/list_envs.py --keyword FastFlat
./isaaclab.sh -p scripts/list_envs.py --keyword Distill
./isaaclab.sh -p scripts/list_envs.py --keyword Eval
```

Task registrations are defined in:

```text
source/go2w/go2w/tasks/manager_based/go2w/__init__.py
```

## Training

Training is performed with:

```text
scripts/rsl_rl/train.py
```

General form:

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py \
  --task <TRAIN_TASK_ID> \
  [options]
```

Task IDs containing `-Play` are playback configurations and are intentionally rejected by `train.py`.

### Common Training Options

| Option | Description |
|---|---|
| `--task` | Registered training task ID |
| `--num_envs` | Number of parallel simulation environments |
| `--seed` | Random seed; `-1` selects a random seed |
| `--max_iterations` | Override the configured number of training iterations |
| `--run_name` | Suffix added to the run directory |
| `--experiment_name` | Override the configured experiment directory |
| `--logger` | Logging backend: `tensorboard`, `wandb`, or `neptune` |
| `--log_project_name` | Project name for W&B or Neptune |
| `--video` | Record training videos |
| `--video_length` | Number of simulation steps per recorded video |
| `--video_interval` | Interval between training videos |
| `--headless` | Run without the graphical viewport |
| `--distributed` | Enable distributed multi-GPU or multi-node training |

Show all available arguments:

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py --help
```

### Train the FastFlat LLC

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Loco-FastFlat-Go2w-v0 \
  --num_envs 8192 \
  --headless
```

Use a smaller `--num_envs` value when GPU memory is limited:

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Loco-FastFlat-Go2w-v0 \
  --num_envs 2048 \
  --headless
```

Override the number of iterations:

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Loco-FastFlat-Go2w-v0 \
  --num_envs 8192 \
  --max_iterations 2000 \
  --run_name fastflat_run \
  --headless
```

### Train the HospitalMaze Teacher

The HospitalMaze teacher executes its navigation commands through a frozen locomotion policy. A pretrained LLC checkpoint is therefore required.

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Nav-HospitalMaze-Teacher-Go2w-v0 \
  --num_envs 8192 \
  --locomotion_checkpoint "$LLC_CKPT" \
  --headless
```

Optional seeded run:

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Nav-HospitalMaze-Teacher-Go2w-v0 \
  --num_envs 8192 \
  --locomotion_checkpoint "$LLC_CKPT" \
  --seed 42 \
  --run_name teacher_seed42 \
  --headless
```

### Train a Depth Student

Depth-student training requires:

- a trained teacher checkpoint;
- the LLC checkpoint used by the navigation stack; and
- the Isaac Sim camera pipeline.

Baseline student:

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Nav-HospitalMaze-Distill-Depth-Go2w-v0 \
  --num_envs 1024 \
  --teacher_checkpoint "$TEACHER_CKPT" \
  --locomotion_checkpoint "$LLC_CKPT" \
  --enable_cameras \
  --headless
```

Available HospitalMaze student tasks:

| Variant | Task ID |
|---|---|
| Baseline | `Nav-HospitalMaze-Distill-Depth-Go2w-v0` |
| LongHist | `Nav-HospitalMaze-Distill-Depth-LongHist-Go2w-v0` |
| Sparse | `Nav-HospitalMaze-Distill-Depth-Sparse-Go2w-v0` |
| 4Cam | `Nav-HospitalMaze-Distill-Depth-4Cam-Go2w-v0` |

Example for LongHist:

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Nav-HospitalMaze-Distill-Depth-LongHist-Go2w-v0 \
  --num_envs 1024 \
  --teacher_checkpoint "$TEACHER_CKPT" \
  --locomotion_checkpoint "$LLC_CKPT" \
  --enable_cameras \
  --headless
```

### Resume Training

Resume from a previous run:

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py \
  --task <TRAIN_TASK_ID> \
  --resume \
  --load_run <RUN_DIRECTORY_NAME> \
  --checkpoint <CHECKPOINT_FILE> \
  [task-specific options]
```

For a navigation task using the frozen LLC:

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Nav-HospitalMaze-Teacher-Go2w-v0 \
  --resume \
  --load_run <RUN_DIRECTORY_NAME> \
  --checkpoint <CHECKPOINT_FILE> \
  --locomotion_checkpoint "$LLC_CKPT" \
  --headless
```

## LLC Playback with Velocity Commands

Use:

```text
scripts/rsl_rl/play_cmd.py
```

This script locks the locomotion command to values specified through the CLI. It can also retain the task's native random command sampler.

General form:

```bash
./isaaclab.sh -p scripts/rsl_rl/play_cmd.py \
  --task Loco-FastFlat-Go2w-Play-v0 \
  --checkpoint "$LLC_CKPT" \
  [command options]
```

### Command Convention

| Option | Unit | Meaning |
|---|---:|---|
| `--cmd_vx` | m/s | Forward or backward linear velocity |
| `--cmd_vy` | m/s | Left or right lateral velocity |
| `--cmd_wz` | rad/s | Yaw angular velocity |
| `--random_commands` | — | Use the environment's native command sampler |

Positive and negative command directions follow the robot and environment frame conventions implemented by the task.

### Stand Still

All fixed commands default to zero:

```bash
./isaaclab.sh -p scripts/rsl_rl/play_cmd.py \
  --task Loco-FastFlat-Go2w-Play-v0 \
  --checkpoint "$LLC_CKPT"
```

### Forward Motion

```bash
./isaaclab.sh -p scripts/rsl_rl/play_cmd.py \
  --task Loco-FastFlat-Go2w-Play-v0 \
  --checkpoint "$LLC_CKPT" \
  --cmd_vx 1.0
```

### Backward Motion

```bash
./isaaclab.sh -p scripts/rsl_rl/play_cmd.py \
  --task Loco-FastFlat-Go2w-Play-v0 \
  --checkpoint "$LLC_CKPT" \
  --cmd_vx -0.5
```

### Lateral Motion

```bash
./isaaclab.sh -p scripts/rsl_rl/play_cmd.py \
  --task Loco-FastFlat-Go2w-Play-v0 \
  --checkpoint "$LLC_CKPT" \
  --cmd_vy 0.5
```

### Rotation in Place

```bash
./isaaclab.sh -p scripts/rsl_rl/play_cmd.py \
  --task Loco-FastFlat-Go2w-Play-v0 \
  --checkpoint "$LLC_CKPT" \
  --cmd_wz 1.0
```

`--cmd_yaw` is accepted as an alias for `--cmd_wz`.

### Combined Motion

```bash
./isaaclab.sh -p scripts/rsl_rl/play_cmd.py \
  --task Loco-FastFlat-Go2w-Play-v0 \
  --checkpoint "$LLC_CKPT" \
  --cmd_vx 1.0 \
  --cmd_vy 0.3 \
  --cmd_wz 0.5
```

### Native Random Commands

```bash
./isaaclab.sh -p scripts/rsl_rl/play_cmd.py \
  --task Loco-FastFlat-Go2w-Play-v0 \
  --checkpoint "$LLC_CKPT" \
  --random_commands
```

### Multiple Environments

```bash
./isaaclab.sh -p scripts/rsl_rl/play_cmd.py \
  --task Loco-FastFlat-Go2w-Play-v0 \
  --checkpoint "$LLC_CKPT" \
  --cmd_vx 1.0 \
  --num_envs 16
```

### Real-Time Playback

```bash
./isaaclab.sh -p scripts/rsl_rl/play_cmd.py \
  --task Loco-FastFlat-Go2w-Play-v0 \
  --checkpoint "$LLC_CKPT" \
  --cmd_vx 0.5 \
  --real-time
```

### Reproducible Playback

```bash
./isaaclab.sh -p scripts/rsl_rl/play_cmd.py \
  --task Loco-FastFlat-Go2w-Play-v0 \
  --checkpoint "$LLC_CKPT" \
  --cmd_vx 1.0 \
  --seed 42
```

Show the full option list:

```bash
./isaaclab.sh -p scripts/rsl_rl/play_cmd.py --help
```

## Navigation Playback

Use:

```text
scripts/rsl_rl/play.py
```

This script supports trained navigation policies, HospitalMaze environments, structured scenes, dynamic obstacles, videos, debug output, and finite-episode evaluation.

General form:

```bash
./isaaclab.sh -p scripts/rsl_rl/play.py \
  --task <PLAY_OR_EVAL_TASK_ID> \
  --checkpoint <NAVIGATION_CHECKPOINT> \
  --locomotion_checkpoint "$LLC_CKPT" \
  [options]
```

### Play the HospitalMaze Teacher

```bash
./isaaclab.sh -p scripts/rsl_rl/play.py \
  --task Nav-HospitalMaze-Teacher-Go2w-Play-v0 \
  --checkpoint "$TEACHER_CKPT" \
  --locomotion_checkpoint "$LLC_CKPT"
```

A teacher checkpoint must be used with a compatible teacher observation configuration.

### Play the Baseline Student

```bash
./isaaclab.sh -p scripts/rsl_rl/play.py \
  --task Nav-HospitalMaze-Distill-Depth-Eval-Static-Go2w-v0 \
  --checkpoint "$STUDENT_CKPT" \
  --locomotion_checkpoint "$LLC_CKPT" \
  --enable_cameras
```

Headless playback:

```bash
./isaaclab.sh -p scripts/rsl_rl/play.py \
  --task Nav-HospitalMaze-Distill-Depth-Eval-Static-Go2w-v0 \
  --checkpoint "$STUDENT_CKPT" \
  --locomotion_checkpoint "$LLC_CKPT" \
  --enable_cameras \
  --headless
```

### Student Evaluation Task IDs

| Variant | Static task | Dynamic task |
|---|---|---|
| Baseline | `Nav-HospitalMaze-Distill-Depth-Eval-Static-Go2w-v0` | `Nav-HospitalMaze-Distill-Depth-Eval-Dynamic-Go2w-v0` |
| LongHist | `Nav-HospitalMaze-Distill-Depth-LongHist-Eval-Static-Go2w-v0` | `Nav-HospitalMaze-Distill-Depth-LongHist-Eval-Dynamic-Go2w-v0` |
| Sparse | `Nav-HospitalMaze-Distill-Depth-Sparse-Eval-Static-Go2w-v0` | `Nav-HospitalMaze-Distill-Depth-Sparse-Eval-Dynamic-Go2w-v0` |
| 4Cam | `Nav-HospitalMaze-Distill-Depth-4Cam-Eval-Static-Go2w-v0` | `Nav-HospitalMaze-Distill-Depth-4Cam-Eval-Dynamic-Go2w-v0` |

## Navigation Scenario Options

### Obstacle Count

Override the number of active obstacles:

```bash
--num_obstacles 10
```

Example:

```bash
./isaaclab.sh -p scripts/rsl_rl/play.py \
  --task <NAVIGATION_TASK_ID> \
  --checkpoint <NAVIGATION_CHECKPOINT> \
  --locomotion_checkpoint "$LLC_CKPT" \
  --num_obstacles 10
```

### Fixed Scenario Templates

Use `--scenario` to select an obstacle layout:

```text
random
empty
head_on
left_edge
right_edge
diag_left
diag_right
off_left
off_right
narrow_gap
narrow_gap_wide
narrow_gap_barely
partial_blockage_left_open
partial_blockage_right_open
cluttered
```

Example:

```bash
./isaaclab.sh -p scripts/rsl_rl/play.py \
  --task <NAVIGATION_TASK_ID> \
  --checkpoint <NAVIGATION_CHECKPOINT> \
  --locomotion_checkpoint "$LLC_CKPT" \
  --scenario narrow_gap \
  --fixed_layout \
  --seed 42
```

Relevant options:

| Option | Description |
|---|---|
| `--scenario` | Force a predefined navigation scenario |
| `--fixed_layout` | Reuse the same layout after each reset |
| `--seed` | Set the layout and environment seed |
| `--nav_fixed_start` | Use a fixed navigation start pose |
| `--nav_start_x` | Override start position x |
| `--nav_start_y` | Override start position y |
| `--nav_start_yaw` | Override start yaw |
| `--nav_goal_forward` | Override forward goal offset |
| `--nav_goal_lateral` | Override lateral goal offset |

### Dynamic Obstacles

Enable obstacle motion:

```bash
--dynamic_obstacles
```

Example:

```bash
./isaaclab.sh -p scripts/rsl_rl/play.py \
  --task <NAVIGATION_TASK_ID> \
  --checkpoint <NAVIGATION_CHECKPOINT> \
  --locomotion_checkpoint "$LLC_CKPT" \
  --dynamic_obstacles
```

Selected dynamic-obstacle options:

| Option | Description |
|---|---|
| `--dynamic_obstacle_speed_range MIN MAX` | Longitudinal speed range in m/s |
| `--dynamic_obstacle_lateral_speed` | Maximum lateral speed in m/s |
| `--dynamic_obstacle_longitudinal_extent` | Maximum longitudinal excursion |
| `--dynamic_obstacle_lateral_extent` | Maximum lateral excursion |
| `--dynamic_obstacle_min_separation` | Minimum center-to-center separation |
| `--dynamic_obstacle_mixed_motion` | Enable time-varying and mixed trajectories |
| `--dynamic_obstacle_speed_change_interval MIN MAX` | Time between speed changes |
| `--dynamic_obstacle_wander_fraction` | Fraction of obstacles using random trajectories |

Example with mixed motion:

```bash
./isaaclab.sh -p scripts/rsl_rl/play.py \
  --task <NAVIGATION_TASK_ID> \
  --checkpoint <NAVIGATION_CHECKPOINT> \
  --locomotion_checkpoint "$LLC_CKPT" \
  --dynamic_obstacles \
  --dynamic_obstacle_speed_range 0.25 0.70 \
  --dynamic_obstacle_mixed_motion \
  --dynamic_obstacle_wander_fraction 0.35
```

### Random Obstacle Shapes

```bash
--random_obstacle_shapes
```

Optional footprint range:

```bash
--random_obstacle_footprint_range 0.12 0.60
```

### Structured Environments

Available values for `--structured_env` are:

```text
none
l_corridor
serpentine_corridor
t_corridor
hospital_ward
```

Example:

```bash
./isaaclab.sh -p scripts/rsl_rl/play.py \
  --task <NAVIGATION_TASK_ID> \
  --checkpoint <NAVIGATION_CHECKPOINT> \
  --locomotion_checkpoint "$LLC_CKPT" \
  --structured_env l_corridor \
  --corridor_width 1.8 \
  --corridor_leg_length 6.0
```

Selected structured-environment options:

| Option | Description |
|---|---|
| `--corridor_width` | Free corridor width in metres |
| `--corridor_leg_length` | Main corridor-leg length |
| `--corridor_turn_length` | Spacing between serpentine legs |
| `--corridor_wall_thickness` | Wall thickness |
| `--structured_goal_done_radius` | Final-goal termination radius |

## A* Playback Options

Selected A* options:

| Option | Description |
|---|---|
| `--astar_grid_resolution` | Occupancy-grid resolution in metres |
| `--astar_lookahead_distance` | Local waypoint lookahead distance |
| `--astar_waypoint_reach_radius` | Radius used to advance held waypoints |
| `--astar_clearance_cost_weight` | Wall-proximity cost weight |
| `--astar_clearance_cost_sigma` | Clearance-cost decay length |
| `--no_adaptive_lookahead` | Disable adaptive lookahead |
| `--lookahead_min` | Minimum lookahead near turns |
| `--curvature_scan_horizon` | Path distance scanned for upcoming turns |
| `--curvature_threshold` | Turn threshold for reducing lookahead |
| `--corner_rounding` | Enable path-corner rounding |
| `--corner_radius` | Corner-rounding radius |
| `--no_astar` | Bypass A* and navigate directly to the final goal |

Example:

```bash
./isaaclab.sh -p scripts/rsl_rl/play.py \
  --task <NAVIGATION_TASK_ID> \
  --checkpoint <NAVIGATION_CHECKPOINT> \
  --locomotion_checkpoint "$LLC_CKPT" \
  --structured_env l_corridor \
  --astar_grid_resolution 0.20 \
  --astar_lookahead_distance 1.25 \
  --astar_clearance_cost_weight 2.0 \
  --astar_clearance_cost_sigma 0.4
```

## Logging and Video

### Named Play Run

Use `--play_name` to save outputs under:

```text
logs/nav_play/<PLAY_NAME>/
```

Example:

```bash
--play_name hospital_demo
```

### Viewport Video

```bash
./isaaclab.sh -p scripts/rsl_rl/play.py \
  --task <NAVIGATION_TASK_ID> \
  --checkpoint <NAVIGATION_CHECKPOINT> \
  --locomotion_checkpoint "$LLC_CKPT" \
  --video \
  --video_length 1000 \
  --headless
```

### Depth-Camera Video

For depth-student tasks:

```bash
./isaaclab.sh -p scripts/rsl_rl/play.py \
  --task <DEPTH_STUDENT_TASK_ID> \
  --checkpoint "$STUDENT_CKPT" \
  --locomotion_checkpoint "$LLC_CKPT" \
  --enable_cameras \
  --depth_video \
  --depth_video_steps 2000 \
  --play_name depth_demo \
  --headless
```

Selected logging options:

| Option | Description |
|---|---|
| `--play_name` | Output directory name under `logs/nav_play/` |
| `--video` | Record the viewport |
| `--video_length` | Viewport-video length in steps |
| `--depth_video` | Record the student depth-camera view |
| `--depth_video_steps` | Maximum number of depth-video steps; `0` records the full run |
| `--nav_log_interval` | Navigation debug print interval |
| `--nav_log_env` | Environment index used for debug logs |
| `--nav_live_obstacle_labels` | Draw live obstacle labels in the viewport |
| `--nav_contact_debug` | Print contact and obstacle diagnostics |
| `--disable_export` | Skip JIT and ONNX export before playback |

## Finite-Episode Evaluation

By default, `play.py` continues until the simulator is closed.

Run a fixed number of navigation trajectories:

```bash
--nav_eval_episodes 100
```

Additional options:

| Option | Description |
|---|---|
| `--nav_eval_episodes` | Number of trajectories to evaluate |
| `--stuck_timeout_steps` | Force-reset an environment after no movement for the specified number of steps |
| `--stuck_threshold` | Displacement threshold used for stuck detection |
| `--seed_per_episode` | Advance the fixed-layout seed after each completed episode |
| `--hospital_maze_route_steps MIN MAX` | Override the sampled route-length range |
| `--terminate_on_final_goal` | End the episode after completing the first final route |

`--terminate_on_final_goal` changes the evaluation protocol. Use it only when single-route completion is the intended outcome.

Show all navigation playback options:

```bash
./isaaclab.sh -p scripts/rsl_rl/play.py --help
```

## Validation

The validation pipeline is configured through:

```text
scripts/configs/validation.yaml
```

Before running validation, replace the checkpoint paths in that file with valid local paths.

### Preflight Check

Check all selected task configurations before launching full rollouts:

```bash
./isaaclab.sh -p scripts/preflight_validation.py
```

Check one policy:

```bash
./isaaclab.sh -p scripts/preflight_validation.py \
  --ablation baseline
```

Check one scenario:

```bash
./isaaclab.sh -p scripts/preflight_validation.py \
  --scenario maze_static
```

The preflight script checks task configuration compatibility. It does not run environment physics or policy rollouts.

### Run the Full Validation Configuration

```bash
./isaaclab.sh -p scripts/run_validation.py \
  --out_name validation_run
```

The default configuration path is:

```text
scripts/configs/validation.yaml
```

Use a different configuration:

```bash
./isaaclab.sh -p scripts/run_validation.py \
  --config /path/to/validation.yaml \
  --out_name validation_run
```

### Selected Validation Options

| Option | Description |
|---|---|
| `--config` | Validation YAML path |
| `--out_name` | Output directory name under `logs/nav_play/` |
| `--ablation` | Run one policy: `teacher`, `baseline`, `longhist`, `sparse`, or `4cam` |
| `--scenario` | Run one scenario |
| `--maze_episodes` | Number of trajectories per policy and scenario |
| `--seed` | Run one seed |
| `--seeds` | Run multiple seeds |
| `--num_envs` | Number of parallel environments |
| `--stuck_timeout` | Stuck timeout in steps |
| `--depth_video_steps` | Maximum depth-video length |
| `--locomotion_checkpoint` | Override the LLC path from the YAML file |
| `--dry_run` | Print generated commands without running them |
| `--skip_plot` | Do not run the plotting script after validation |

Dry run:

```bash
./isaaclab.sh -p scripts/run_validation.py \
  --out_name validation_test \
  --dry_run
```

One policy and one scenario:

```bash
./isaaclab.sh -p scripts/run_validation.py \
  --out_name baseline_static \
  --ablation baseline \
  --scenario maze_static
```

Multiple seeds:

```bash
./isaaclab.sh -p scripts/run_validation.py \
  --out_name validation_multiseed \
  --seeds 42 43 44
```

## Plotting Existing Validation Outputs

Aggregate an existing validation directory:

```bash
./isaaclab.sh -p scripts/plot_validation.py \
  logs/nav_play/<VALIDATION_DIRECTORY>
```

The plotting script reads `session_manifest.json` files and writes summary tables and plots into the validation directory.

Typical outputs include:

```text
summary.csv
comparison_bar.png
seed_points.png
contact_peak_ecdf.png
safety_progress.png
termination_mix.png
```

## Cluster Scripts

Cluster launchers are located under:

```text
scripts/slurm/
```

They contain machine-specific paths, scheduler settings, container configuration, and resource requests.

See [`slurm/README.md`](slurm/README.md) before submitting jobs on Karolina or MeluXina.

## Related Documentation

- [Project overview](../README.md)
- [Go2-W extension and task catalog](../source/go2w/README.md)
- [Locomotion configuration](../source/go2w/go2w/tasks/manager_based/go2w/cfg/locomotion/README.md)
- [Navigation configuration](../source/go2w/go2w/tasks/manager_based/go2w/cfg/navigation/README.md)