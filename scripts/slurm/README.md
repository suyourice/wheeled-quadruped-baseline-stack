# Slurm Scripts

This directory contains Slurm launch scripts for running Go2-W training, playback, and validation jobs on GPU clusters.

The scripts currently provide configurations for:

```text
scripts/slurm/
├── karolina/
│   ├── train_go2w.sh
│   ├── play_go2w.sh
│   └── validate_go2w.sh
└── meluxina/
    ├── train_go2w.sh
    ├── play_go2w.sh
    ├── validate_go2w.sh
    ├── plot_go2w.sh
    └── submit_validation.sh
```

These files are cluster-specific templates. Update the account, partition, repository path, container path, and requested resources before submitting jobs on another system.

See:

- [Project overview](../../README.md)
- [Training and playback scripts](../README.md)

## Prerequisites

The cluster scripts assume:

- a Slurm workload manager;
- NVIDIA GPU nodes;
- Apptainer with NVIDIA GPU support;
- an Isaac Sim container image;
- a writable repository checkout;
- the `go2w` package installed in the container environment; and
- valid local checkpoint paths for dependent training or playback stages.

The current scripts use the following default locations:

```text
Repository:      $HOME/go2w
Container image: $HOME/isaacsim.sif
```

Change these paths in the shell scripts when the repository or container is stored elsewhere.

## Required Customization

Before the first submission, inspect the selected cluster script and update the following fields.

### Slurm Account

The scripts contain:

```bash
#SBATCH -A YOUR_ACCOUNT
```

Replace `YOUR_ACCOUNT` with the allocation or project account used on the target cluster.

### Partition and QoS

The current templates use:

```text
Karolina:
    partition: qgpu

MeluXina:
    partition: gpu
    qos: default
```

These values are cluster-specific. Confirm the correct partition, QoS, time limit, and GPU request with the cluster documentation.

### Repository Path

The scripts currently enter:

```bash
cd $HOME/wheeled-quadruped-baseline-stack
```

Replace this path when the repository is checked out elsewhere.

### Container Path

The scripts currently launch:

```text
$HOME/isaacsim.sif
```

Replace this path with the local Isaac Sim Apptainer image.

### Checkpoint Paths

Checkpoints are not included in the repository.

Pass valid paths through the corresponding command-line options:

```bash
--checkpoint /path/to/policy_checkpoint.pt
--locomotion_checkpoint /path/to/locomotion_checkpoint.pt
--teacher_checkpoint /path/to/teacher_checkpoint.pt
```

## Argument Forwarding

The training and playback wrappers forward all arguments after the shell-script name to the underlying Python script.

For example:

```bash
sbatch scripts/slurm/karolina/train_go2w.sh \
  --task Loco-FastFlat-Go2w-v0 \
  --num_envs 8192 \
  --headless
```

runs the equivalent container command:

```bash
scripts/rsl_rl/train.py \
  --task Loco-FastFlat-Go2w-v0 \
  --num_envs 8192 \
  --headless
```

Use the same task and CLI options documented in:

```text
scripts/README.md
```

## Karolina

### Train a Policy

Script:

```text
scripts/slurm/karolina/train_go2w.sh
```

Default resource request:

| Resource | Value |
|---|---:|
| Partition | `qgpu` |
| GPUs | `1` |
| Time limit | `8 hours` |

Train the FastFlat LLC:

```bash
sbatch scripts/slurm/karolina/train_go2w.sh \
  --task Loco-FastFlat-Go2w-v0 \
  --num_envs 8192 \
  --headless
```

Train the HospitalMaze teacher:

```bash
sbatch scripts/slurm/karolina/train_go2w.sh \
  --task Nav-HospitalMaze-Teacher-Go2w-v0 \
  --num_envs 8192 \
  --locomotion_checkpoint /path/to/locomotion_checkpoint.pt \
  --headless
```

Train a depth student:

```bash
sbatch scripts/slurm/karolina/train_go2w.sh \
  --task Nav-HospitalMaze-Distill-Depth-Go2w-v0 \
  --num_envs 512 \
  --teacher_checkpoint /path/to/teacher_checkpoint.pt \
  --locomotion_checkpoint /path/to/locomotion_checkpoint.pt \
  --enable_cameras \
  --headless
```

### Play a Policy

Script:

```text
scripts/slurm/karolina/play_go2w.sh
```

Default resource request:

| Resource | Value |
|---|---:|
| Partition | `qgpu` |
| GPUs | `1` |
| Time limit | `2 hours` |

Example:

```bash
sbatch scripts/slurm/karolina/play_go2w.sh \
  --task Nav-HospitalMaze-Teacher-Go2w-Play-v0 \
  --checkpoint /path/to/teacher_checkpoint.pt \
  --locomotion_checkpoint /path/to/locomotion_checkpoint.pt \
  --headless
```

For a depth student:

```bash
sbatch scripts/slurm/karolina/play_go2w.sh \
  --task Nav-HospitalMaze-Distill-Depth-Eval-Static-Go2w-v0 \
  --checkpoint /path/to/student_checkpoint.pt \
  --locomotion_checkpoint /path/to/locomotion_checkpoint.pt \
  --enable_cameras \
  --headless
```

### Run Validation

Script:

```text
scripts/slurm/karolina/validate_go2w.sh
```

Default resource request:

| Resource | Value |
|---|---:|
| Partition | `qgpu` |
| GPUs | `5` |
| Time limit | `12 hours` |

The script assigns one policy to each GPU:

```text
GPU 0: Baseline
GPU 1: LongHist
GPU 2: Sparse
GPU 3: 4Cam
GPU 4: Teacher
```

Each policy runs through the scenarios requested by `run_validation.py`. Plotting begins after all policy processes finish.

Submit a validation run:

```bash
OUT_NAME=validation_run \
sbatch scripts/slurm/karolina/validate_go2w.sh
```

Specify seeds and episode count:

```bash
OUT_NAME=validation_multiseed \
sbatch scripts/slurm/karolina/validate_go2w.sh \
  --seeds 42 43 44 \
  --maze_episodes 100
```

Override the validation configuration:

```bash
OUT_NAME=validation_custom \
sbatch scripts/slurm/karolina/validate_go2w.sh \
  --config /path/to/validation.yaml
```

The current script launches validation with:

```text
--num_envs 32
```

A later `--num_envs` argument can be supplied when a different parallel rollout count is required.

## MeluXina

The MeluXina scripts load Apptainer through:

```bash
module load Apptainer
```

Confirm the module name on the target system.

### Train a Policy

Script:

```text
scripts/slurm/meluxina/train_go2w.sh
```

Default resource request:

| Resource | Value |
|---|---:|
| Partition | `gpu` |
| QoS | `default` |
| GPUs | `1` |
| Time limit | `24 hours` |

Train the FastFlat LLC:

```bash
sbatch scripts/slurm/meluxina/train_go2w.sh \
  --task Loco-FastFlat-Go2w-v0 \
  --num_envs 8192 \
  --headless
```

Train the HospitalMaze teacher:

```bash
sbatch scripts/slurm/meluxina/train_go2w.sh \
  --task Nav-HospitalMaze-Teacher-Go2w-v0 \
  --num_envs 8192 \
  --locomotion_checkpoint /path/to/locomotion_checkpoint.pt \
  --headless
```

Train a depth student:

```bash
sbatch scripts/slurm/meluxina/train_go2w.sh \
  --task Nav-HospitalMaze-Distill-Depth-Go2w-v0 \
  --num_envs 512 \
  --teacher_checkpoint /path/to/teacher_checkpoint.pt \
  --locomotion_checkpoint /path/to/locomotion_checkpoint.pt \
  --enable_cameras \
  --headless
```

### Play a Policy

Script:

```text
scripts/slurm/meluxina/play_go2w.sh
```

Default resource request:

| Resource | Value |
|---|---:|
| Partition | `gpu` |
| QoS | `default` |
| GPUs | `1` |
| Time limit | `2 hours` |

Example:

```bash
sbatch scripts/slurm/meluxina/play_go2w.sh \
  --task Nav-HospitalMaze-Teacher-Go2w-Play-v0 \
  --checkpoint /path/to/teacher_checkpoint.pt \
  --locomotion_checkpoint /path/to/locomotion_checkpoint.pt \
  --headless
```

For depth-based playback:

```bash
sbatch scripts/slurm/meluxina/play_go2w.sh \
  --task Nav-HospitalMaze-Distill-Depth-Eval-Static-Go2w-v0 \
  --checkpoint /path/to/student_checkpoint.pt \
  --locomotion_checkpoint /path/to/locomotion_checkpoint.pt \
  --enable_cameras \
  --headless
```

### Submit the Full Validation Pipeline

The recommended MeluXina entry point is:

```text
scripts/slurm/meluxina/submit_validation.sh
```

It performs two submissions:

1. a four-task validation array; and
2. a dependent CPU plotting job that starts only after the validation array succeeds.

General form:

```bash
bash scripts/slurm/meluxina/submit_validation.sh \
  <OUTPUT_NAME> \
  [run_validation.py options]
```

Example:

```bash
bash scripts/slurm/meluxina/submit_validation.sh \
  validation_run
```

Multi-seed example:

```bash
bash scripts/slurm/meluxina/submit_validation.sh \
  validation_multiseed \
  --seeds 42 43 44 \
  --maze_episodes 100
```

The first argument is used as the output directory name. Remaining arguments are forwarded to `run_validation.py`.

### Validation Array

The array script is:

```text
scripts/slurm/meluxina/validate_go2w.sh
```

Default resource request per array task:

| Resource | Value |
|---|---:|
| Nodes | `1` |
| GPUs | `4` |
| Time limit | `8 hours` |
| Array range | `0-3` |

Array-task mapping:

| Array index | Scenario |
|---:|---|
| `0` | `maze_train` |
| `1` | `maze_static` |
| `2` | `maze_dynamic` |
| `3` | `maze_success` |

Within each array task:

```text
GPU 0: Baseline followed by Teacher
GPU 1: LongHist
GPU 2: Sparse
GPU 3: 4Cam
```

The validation processes use separate CUDA devices and isolated Isaac Sim runtime/cache directories.

The current script launches each policy with:

```text
--num_envs 224
```

This value may be overridden through the forwarded validation arguments when required.

Direct submission without the automatic plot dependency is also possible:

```bash
OUT_NAME=validation_run \
sbatch scripts/slurm/meluxina/validate_go2w.sh \
  --seeds 42 43 44 \
  --maze_episodes 100
```

### Plot Job

The dependent plot script is:

```text
scripts/slurm/meluxina/plot_go2w.sh
```

Default resource request:

| Resource | Value |
|---|---:|
| Partition | `cpu` |
| CPUs | `4` |
| Time limit | `4 hours` |

The script produces separate summaries for:

- the long-horizon scenarios `maze_train`, `maze_static`, and `maze_dynamic`; and
- the short-route `maze_success` scenario.

The plot job expects:

```bash
OUT_NAME=<validation output directory>
```

Direct submission:

```bash
OUT_NAME=validation_run \
sbatch scripts/slurm/meluxina/plot_go2w.sh
```

Normally, `submit_validation.sh` supplies this variable and creates the dependency automatically.

## Runtime Isolation

Isaac Sim creates runtime, document, and terrain caches. Concurrent jobs or processes must not write to the same cache directories.

The validation scripts create per-job or per-GPU runtime directories under:

```text
${XDG_CACHE_HOME:-$HOME/.cache}/go2w_isaacsim/
```

They also set a separate terrain cache through:

```text
GO2W_TERRAIN_CACHE_DIR
```

This prevents concurrent terrain-generation processes from sharing partially written cache files.

Do not remove the per-job cache isolation when running multiple Isaac Sim processes on the same node.

## Logs and Outputs

### Training and Playback Logs

The default Slurm output pattern is:

```text
logs/slurm/<JOB_NAME>_<JOB_ID>.out
```

### Validation Logs

Karolina validation logs are written under:

```text
logs/slurm/validate/<JOB_ID>/
```

MeluXina array-validation logs are written under:

```text
logs/slurm/validate/<ARRAY_JOB_ID>/
```

### Validation Results

Validation outputs are written under:

```text
logs/nav_play/<OUT_NAME>/
```

Monitor active jobs:

```bash
squeue -u "$USER"
```

Inspect a Slurm output:

```bash
tail -f logs/slurm/<LOG_FILE>
```

Inspect a validation policy or scenario log:

```bash
tail -f logs/slurm/validate/<JOB_ID>/<LOG_FILE>
```

## Resource Adjustment

The checked-in resource values reflect the current cluster scripts and are not universal recommendations.

Before increasing or decreasing resources, consider:

- GPU memory required by the selected task;
- the number of simulated environments;
- the number of depth cameras;
- whether the task uses camera rendering;
- the number of policies launched concurrently;
- the number of scenarios and seeds; and
- the cluster's maximum job duration.

The 4Cam student generally requires more camera-processing resources than the single-camera variants. Reduce `--num_envs` first when a job exceeds GPU memory.

## Common Problems

### Invalid Account

Error:

```text
Invalid account or account/partition combination
```

Check:

```bash
#SBATCH -A YOUR_ACCOUNT
```

and verify that the account is allowed on the selected partition.

### Container Not Found

Error:

```text
$HOME/isaacsim.sif: No such file or directory
```

Update the container path in the selected shell script.

### Repository Not Found

Error:

```text
cd: $HOME/go2w: No such file or directory
```

Update the repository path in the script.

### Apptainer Module Not Found

If:

```bash
module load Apptainer
```

fails, inspect the available modules:

```bash
module avail
```

and replace the module name with the one provided by the cluster.

### Missing Checkpoint

Navigation tasks require explicit checkpoint paths. Confirm that the paths are visible inside the container through the `$HOME:$HOME` bind mount.

### Camera Initialization Failure

Depth-student tasks require:

```bash
--enable_cameras
```

Use `--headless` for noninteractive cluster jobs.

### Out of GPU Memory

Reduce:

```bash
--num_envs
```

The appropriate value depends on the task, camera configuration, and GPU model.

### Terrain Cache Errors

Keep the per-process `GO2W_TERRAIN_CACHE_DIR` isolation used by the validation scripts. Shared terrain-cache directories can cause failures when several Isaac Sim processes generate the same terrain concurrently.

## Related Documentation

- [Project overview](../../README.md)
- [Scripts and command-line usage](../README.md)
- [Go2-W extension overview](../../source/go2w/README.md)
- [Locomotion configuration](../../source/go2w/go2w/tasks/manager_based/go2w/cfg/locomotion/README.md)
- [Navigation configuration](../../source/go2w/go2w/tasks/manager_based/go2w/cfg/navigation/README.md)