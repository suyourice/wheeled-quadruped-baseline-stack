# Go2-W Isaac Lab Extension

This directory contains the custom Isaac Lab extension for locomotion and hierarchical indoor navigation of the **Unitree Go2-W wheeled quadruped**.

The extension provides:

- Go2-W robot and actuator configurations;
- low-level locomotion tasks;
- privileged navigation-teacher tasks;
- depth-based student-distillation tasks;
- HospitalMaze training, playback, and evaluation environments;
- navigation and locomotion MDP terms; and
- RSL-RL agent configurations.

See the [root README](../../README.md) for the overall system architecture and usage overview.

## Installation

Install the extension from the repository root:

```bash
./isaaclab.sh -p -m pip install -e source/go2w
```

The package name is:

```text
go2w
```

Importing `go2w.tasks` registers the Go2-W Gymnasium environments.

To confirm the installation and inspect the registered tasks:

```bash
./isaaclab.sh -p scripts/list_envs.py
```

Filter the task list:

```bash
./isaaclab.sh -p scripts/list_envs.py --keyword HospitalMaze
```

## Package Structure

```text
source/go2w/
├── config/
│   └── extension.toml
├── go2w/
│   ├── assets/
│   └── tasks/
│       └── manager_based/
│           └── go2w/
│               ├── agents/
│               ├── cfg/
│               │   ├── hospital/
│               │   ├── locomotion/
│               │   └── navigation/
│               ├── mdp/
│               │   ├── locomotion/
│               │   └── navigation/
│               ├── distillation_algorithms.py
│               └── __init__.py
└── setup.py
```

| Path | Purpose |
|---|---|
| `config/extension.toml` | Isaac Lab extension metadata |
| `go2w/assets/` | Go2-W robot and actuator configurations |
| `go2w/tasks/manager_based/go2w/__init__.py` | Gymnasium task registration |
| `go2w/tasks/manager_based/go2w/agents/` | RSL-RL runner, PPO, and distillation configurations |
| `go2w/tasks/manager_based/go2w/cfg/locomotion/` | Locomotion training and playback environments |
| `go2w/tasks/manager_based/go2w/cfg/navigation/` | Navigation teacher and depth-student training environments |
| `go2w/tasks/manager_based/go2w/cfg/hospital/` | Hospital and HospitalMaze playback/evaluation environments |
| `go2w/tasks/manager_based/go2w/mdp/locomotion/` | Locomotion observations, rewards, curricula, and events |
| `go2w/tasks/manager_based/go2w/mdp/navigation/` | Navigation actions, observations, rewards, planning, and obstacle logic |
| `go2w/tasks/manager_based/go2w/distillation_algorithms.py` | Teacher–student distillation implementation |

## Task Naming

Task IDs follow the general form:

```text
Loco-<Scene>-Go2w-v0
Nav-<Scene>-<Policy>-Go2w-v0
```

Additional tokens describe:

- training or playback mode;
- teacher or student policy type;
- student observation variant;
- hospital venue;
- evaluation distribution; and
- static or dynamic obstacle configuration.

Examples:

```text
Loco-FastFlat-Go2w-v0
Loco-FastFlat-Go2w-Play-v0

Nav-HospitalMaze-Teacher-Go2w-v0
Nav-HospitalMaze-Teacher-Go2w-Play-v0

Nav-HospitalMaze-Distill-Depth-Go2w-v0
Nav-HospitalMaze-Distill-Depth-LongHist-Go2w-v0
Nav-HospitalMaze-Distill-Depth-Sparse-Go2w-v0
Nav-HospitalMaze-Distill-Depth-4Cam-Go2w-v0
```

Use the task-listing script rather than relying on this README as an exhaustive registry:

```bash
./isaaclab.sh -p scripts/list_envs.py
```

## Locomotion Tasks

The primary low-level locomotion task is:

```text
Loco-FastFlat-Go2w-v0
```

Its playback configuration is:

```text
Loco-FastFlat-Go2w-Play-v0
```

The locomotion policy receives robot proprioception and a planar velocity command:

```text
[v_x, v_y, omega_z]
```

It produces a 16-dimensional action interpreted as:

- 4 wheel velocity targets;
- 4 hip position targets; and
- 8 thigh/calf position targets.

The trained locomotion policy is used as a frozen actuator-level controller by the hierarchical navigation tasks.

For detailed LLC documentation, see:

[`go2w/tasks/manager_based/go2w/cfg/locomotion/README.md`](go2w/tasks/manager_based/go2w/cfg/locomotion/README.md)

## Navigation Tasks

The navigation stack consists of:

```text
Final goal
    ↓
A* global route
    ↓
Rolling local waypoint
    ↓
HLC navigation command
    ↓
Frozen LLC
    ↓
Go2-W wheel and leg targets
```

The high-level navigation action is:

```text
[v_x, v_y, omega_z]
```

Navigation tasks using `FrozenLLCActionTerm` require a compatible pretrained locomotion checkpoint.

Pass it through:

```bash
--locomotion_checkpoint /path/to/locomotion_checkpoint.pt
```

## Privileged Teacher

The primary HospitalMaze teacher task is:

```text
Nav-HospitalMaze-Teacher-Go2w-v0
```

The teacher is trained with PPO using privileged navigation and geometric observations.

Example:

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Nav-HospitalMaze-Teacher-Go2w-v0 \
  --locomotion_checkpoint /path/to/locomotion_checkpoint.pt \
  --num_envs 8192 \
  --headless
```

Representative playback task:

```text
Nav-HospitalMaze-Teacher-Go2w-Play-v0
```

Example:

```bash
./isaaclab.sh -p scripts/rsl_rl/play.py \
  --task Nav-HospitalMaze-Teacher-Go2w-Play-v0 \
  --checkpoint /path/to/teacher_checkpoint.pt \
  --locomotion_checkpoint /path/to/locomotion_checkpoint.pt
```

Teacher checkpoints must be used with a compatible teacher observation configuration.

## Depth Students

Depth students imitate a frozen navigation teacher while retaining the same three-dimensional HLC action interface.

The primary student variants are:

| Variant | Training task |
|---|---|
| Baseline | `Nav-HospitalMaze-Distill-Depth-Go2w-v0` |
| LongHist | `Nav-HospitalMaze-Distill-Depth-LongHist-Go2w-v0` |
| Sparse | `Nav-HospitalMaze-Distill-Depth-Sparse-Go2w-v0` |
| 4Cam | `Nav-HospitalMaze-Distill-Depth-4Cam-Go2w-v0` |

Student training requires:

- a navigation-teacher checkpoint;
- a locomotion checkpoint;
- camera support; and
- the student task matching the intended observation configuration.

Example:

```bash
./isaaclab.sh -p scripts/rsl_rl/train.py \
  --task Nav-HospitalMaze-Distill-Depth-Go2w-v0 \
  --teacher_checkpoint /path/to/teacher_checkpoint.pt \
  --locomotion_checkpoint /path/to/locomotion_checkpoint.pt \
  --num_envs 1024 \
  --enable_cameras \
  --headless
```

For detailed navigation documentation, see:

[`go2w/tasks/manager_based/go2w/cfg/navigation/README.md`](go2w/tasks/manager_based/go2w/cfg/navigation/README.md)

## Hospital and HospitalMaze Environments

Hospital-related configurations are located in:

```text
go2w/tasks/manager_based/go2w/cfg/hospital/
```

They provide configurations for:

- HospitalMaze teacher playback;
- HospitalMaze student playback;
- corridor and ward visualization;
- training-distribution evaluation;
- static-obstacle evaluation; and
- dynamic-obstacle evaluation.

Representative evaluation tasks include:

```text
Nav-HospitalMaze-Teacher-Eval-TrainDist-Go2w-v0
Nav-HospitalMaze-Teacher-Eval-Static-Go2w-v0
Nav-HospitalMaze-Teacher-Eval-Dynamic-Go2w-v0

Nav-HospitalMaze-Distill-Depth-Eval-TrainDist-Go2w-v0
Nav-HospitalMaze-Distill-Depth-Eval-Static-Go2w-v0
Nav-HospitalMaze-Distill-Depth-Eval-Dynamic-Go2w-v0
```

Equivalent evaluation registrations are provided for LongHist, Sparse, and 4Cam students.

## Frozen LLC Interface

The navigation policy does not directly output joint actions.

`FrozenLLCActionTerm` performs the following sequence:

1. receives the HLC command `[v_x, v_y, omega_z]`;
2. reconstructs the 60-dimensional LLC observation;
3. runs the frozen locomotion actor;
4. obtains a 16-dimensional LLC action;
5. applies wheel velocity targets; and
6. applies leg position targets.

The implementation is located at:

```text
go2w/tasks/manager_based/go2w/mdp/navigation/actions.py
```

A locomotion checkpoint is mandatory for tasks configured with this action term.

## Configuration Layers

The extension separates task configuration into three main areas.

### Locomotion

```text
go2w/tasks/manager_based/go2w/cfg/locomotion/
```

Contains:

- robot scene configuration;
- locomotion observations;
- velocity-command configuration;
- hybrid wheel/leg action configuration;
- reward and curriculum configuration; and
- training and playback variants.

### Navigation

```text
go2w/tasks/manager_based/go2w/cfg/navigation/
```

Contains:

- privileged teacher environments;
- depth-student environments;
- teacher and student observation groups;
- Frozen LLC action configuration;
- obstacle and navigation rewards; and
- distillation variants.

### Hospital

```text
go2w/tasks/manager_based/go2w/cfg/hospital/
```

Contains:

- HospitalMaze scene configurations;
- teacher and student playback variants;
- static and dynamic evaluation variants; and
- training-distribution evaluation configurations.

## Agent Configurations

RSL-RL configurations are located in:

```text
go2w/tasks/manager_based/go2w/agents/
```

These configurations define:

- experiment names;
- runner classes;
- PPO hyperparameters;
- actor and critic network configuration;
- teacher model configuration;
- student model configuration; and
- training iteration limits.

Command-line arguments can override selected values such as:

```text
--seed
--num_envs
--max_iterations
--run_name
--experiment_name
--logger
```

## Checkpoints

Checkpoints are generated under the configured RSL-RL log directory and are not included in the repository.

Use explicit paths when a task depends on a previously trained component:

```bash
export LLC_CKPT="/path/to/locomotion_checkpoint.pt"
export TEACHER_CKPT="/path/to/teacher_checkpoint.pt"
export STUDENT_CKPT="/path/to/student_checkpoint.pt"
```

Do not assume that a checkpoint is compatible solely because its action dimension matches. The checkpoint must also match:

- the registered task;
- the observation layout;
- the runner type;
- the network configuration; and
- the frozen LLC dependency used during training.

## Legacy Configurations

The extension retains earlier Flat and ObstacleFlat tasks in addition to the final HospitalMaze pipeline.

These tasks remain useful for:

- locomotion testing;
- obstacle-navigation debugging;
- structured-scene playback; and
- comparison with earlier environment configurations.

Use:

```bash
./isaaclab.sh -p scripts/list_envs.py --keyword ObstacleFlat
```

to inspect the available legacy registrations.

## Related Documentation

- [Project overview](../../README.md)
- [Training, playback, and validation scripts](../../scripts/README.md)
- [Locomotion configuration](go2w/tasks/manager_based/go2w/cfg/locomotion/README.md)
- [Navigation configuration](go2w/tasks/manager_based/go2w/cfg/navigation/README.md)
- [Cluster scripts](../../scripts/slurm/README.md)