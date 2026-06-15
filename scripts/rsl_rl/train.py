# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import sys
from collections import defaultdict

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from checkpoint_utils import (  # isort: skip
    configure_frozen_llc_action,
    find_state_dict,
    load_padded_state_dict,
    load_teacher_locomotion_checkpoint,
)

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--teacher_checkpoint", type=str, default=None,
    help="Absolute path to teacher checkpoint for distillation (overrides --load_run/--checkpoint).",
)
parser.add_argument(
    "--locomotion_checkpoint", type=str, default=None,
    help="Path to a pre-trained fast-flat locomotion checkpoint. For HLC navigation tasks this is injected into "
         "FrozenLLCActionTerm before env creation. Legacy obstacle tasks still use it for model warm-starts.",
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument(
    "--teacher_only_eval",
    action="store_true",
    default=False,
    help="Run the active distillation teacher directly for episode-metric evaluation instead of training.",
)
parser.add_argument(
    "--eval_num_episodes",
    type=int,
    default=256,
    help="Number of completed episodes to aggregate in --teacher_only_eval mode.",
)
parser.add_argument(
    "--eval_print_interval",
    type=int,
    default=64,
    help="How often to print progress in --teacher_only_eval mode, measured in completed episodes.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


def _reject_play_task_for_training(task_name: str | None) -> None:
    """Prevent accidentally training with evaluation/play environment configs."""
    if task_name is not None and "-Play" in task_name:
        raise ValueError(
            f"Refusing to train a Play task: {task_name}\n"
            "Use scripts/rsl_rl/play.py for Play tasks, or choose a non-Play training task."
        )


_reject_play_task_for_training(args_cli.task)

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import os
import time
from datetime import datetime
from types import MethodType

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from rsl_rl.utils import resolve_callable

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import go2w.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _install_resilient_runner_save(runner, retry_count: int = 3, retry_delay_s: float = 2.0) -> None:
    """Protect long runs from transient filesystem failures during checkpoint saves.

    Some HPC filesystem backends occasionally fail to open the final checkpoint path
    even though the directory is valid and earlier checkpoints were written. We do not
    want a multi-hour training run to die on a periodic save, so save atomically via a
    temporary file, retry a few times, and finally fall back to `/tmp` with a warning.
    """

    def _resilient_save(self, path: str, infos: dict | None = None) -> None:
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos

        target_dir = os.path.dirname(path)
        os.makedirs(target_dir, exist_ok=True)
        base_name = os.path.basename(path)
        last_exc: Exception | None = None

        for attempt in range(1, retry_count + 1):
            tmp_path = os.path.join(target_dir, f".{base_name}.tmp.{os.getpid()}.{attempt}")
            try:
                torch.save(saved_dict, tmp_path)
                os.replace(tmp_path, path)
                self.logger.save_model(path, self.current_learning_iteration)
                return
            except Exception as exc:  # noqa: BLE001 - keep long training alive on FS hiccups
                last_exc = exc
                print(
                    f"[WARN] Checkpoint save attempt {attempt}/{retry_count} failed for '{path}': {exc}"
                )
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except OSError:
                    pass
                if attempt < retry_count:
                    time.sleep(retry_delay_s)

        fallback_dir = os.path.join("/tmp", "go2w_checkpoint_fallback")
        os.makedirs(fallback_dir, exist_ok=True)
        fallback_path = os.path.join(
            fallback_dir,
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{base_name}",
        )
        torch.save(saved_dict, fallback_path)
        print(
            f"[WARN] Failed to save checkpoint to '{path}' after {retry_count} attempts. "
            f"Wrote fallback checkpoint to '{fallback_path}' instead. Last error: {last_exc}"
        )

    runner.save = MethodType(_resilient_save, runner)


def _load_locomotion_checkpoint(runner: OnPolicyRunner, ckpt_path: str, device: str) -> None:
    """Warm-start actor and critic from a flat checkpoint with obs-dim padding.

    Handles the 60D→90D mismatch between flat env and obstacle env: the extra 30 dims
    (obstacle positions) start at zero and receive gradient once obstacles appear.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    actor_key, actor_sd = find_state_dict(
        ckpt,
        ("actor_state_dict", "model_state_dict", "policy_state_dict"),
        "actor",
    )
    actor_target = getattr(runner.alg.actor, "frozen_actor", runner.alg.actor)
    actor_label = "frozen actor" if actor_target is not runner.alg.actor else "actor"
    load_padded_state_dict(actor_target, actor_sd, device, actor_label, strip_distribution=True)
    print(f"[INFO] Loaded actor from '{actor_key}'")

    try:
        critic_key, critic_sd = find_state_dict(ckpt, ("critic_state_dict", "value_state_dict"), "critic")
    except ValueError as exc:
        print(f"[WARN] {exc}; critic remains randomly initialized")
    else:
        load_padded_state_dict(runner.alg.critic, critic_sd, device, "critic")
        print(f"[INFO] Loaded critic from '{critic_key}'")

    print(f"[INFO] Loaded locomotion checkpoint from: {ckpt_path}")



def _build_teacher_for_eval(env, obs, agent_cfg: RslRlBaseRunnerCfg, device: str):
    """Instantiate the active distillation teacher for direct evaluation."""
    teacher_cfg = getattr(agent_cfg, "teacher", None)
    if teacher_cfg is None:
        raise ValueError("--teacher_only_eval requires a distillation runner config with a 'teacher' model config.")

    teacher_cfg_dict = copy.deepcopy(teacher_cfg.to_dict())
    teacher_class = resolve_callable(teacher_cfg_dict.pop("class_name"))
    teacher = teacher_class(obs, {"teacher": ["teacher"]}, "teacher", env.num_actions, **teacher_cfg_dict)
    teacher = teacher.to(device)
    teacher.eval()
    return teacher


def _format_eval_metrics(metrics: dict[str, float], completed_episodes: int, avg_episode_length: float) -> str:
    """Format the most useful teacher-eval summary metrics for quick reading."""
    preferred_keys = [
        "goal_reached_rate",
        "time_out_rate",
        "base_contact_rate",
        "root_height_below_minimum_rate",
        "multi_term_fraction",
    ]
    parts = [f"episodes={completed_episodes}", f"avg_episode_len={avg_episode_length:.2f}"]
    for key in preferred_keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    return " ".join(parts)


def _run_teacher_only_eval(env, agent_cfg: RslRlBaseRunnerCfg) -> None:
    """Run pure teacher rollouts and aggregate current task metrics."""
    if args_cli.locomotion_checkpoint is None:
        raise ValueError("--teacher_only_eval requires --locomotion_checkpoint for the teacher LLC.")
    if agent_cfg.class_name != "DistillationRunner":
        raise ValueError("--teacher_only_eval is only supported for distillation tasks.")

    # Use the same RSL-RL wrapper path as play/train so observation and step APIs stay consistent.
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    obs = env.get_observations()
    teacher = _build_teacher_for_eval(env, obs, agent_cfg, env.unwrapped.device)
    load_teacher_locomotion_checkpoint(teacher, args_cli.locomotion_checkpoint, env.unwrapped.device)

    num_envs = env.unwrapped.num_envs
    episode_lengths = torch.zeros(num_envs, device=env.unwrapped.device, dtype=torch.long)
    completed_episodes = 0
    total_episode_length = 0.0
    termination_names = list(env.unwrapped.termination_manager.active_terms)
    termination_counts: dict[str, int] = defaultdict(int)
    multi_term_episodes = 0

    print(
        "[INFO] Running teacher-only evaluation on the active goal-conditioned local-navigation task: "
        f"target episodes={args_cli.eval_num_episodes}"
    )

    while simulation_app.is_running() and completed_episodes < args_cli.eval_num_episodes:
        with torch.inference_mode():
            actions = teacher(obs)
            obs, _, dones, extras = env.step(actions)
            teacher.reset(dones.bool())

        episode_lengths += 1
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        num_done = int(done_ids.numel())
        if num_done == 0:
            continue

        total_episode_length += float(episode_lengths[done_ids].sum().item())
        episode_lengths[done_ids] = 0
        completed_episodes += num_done

        done_terms = env.unwrapped.termination_manager._last_episode_dones[done_ids]
        if done_terms.numel() > 0:
            multi_term_episodes += int((done_terms.sum(dim=1) > 1).sum().item())
            for idx, term_name in enumerate(termination_names):
                termination_counts[term_name] += int(done_terms[:, idx].sum().item())

        if (
            completed_episodes >= args_cli.eval_num_episodes
            or (
                args_cli.eval_print_interval > 0
                and completed_episodes % args_cli.eval_print_interval < num_done
            )
        ):
            averaged = {
                f"{term_name}_rate": termination_counts[term_name] / max(completed_episodes, 1)
                for term_name in termination_names
            }
            averaged["multi_term_fraction"] = multi_term_episodes / max(completed_episodes, 1)
            avg_episode_length = total_episode_length / max(completed_episodes, 1)
            print("[TEACHER-EVAL] " + _format_eval_metrics(averaged, completed_episodes, avg_episode_length))

    averaged = {
        f"{term_name}_rate": termination_counts[term_name] / max(completed_episodes, 1)
        for term_name in termination_names
    }
    averaged["multi_term_fraction"] = multi_term_episodes / max(completed_episodes, 1)
    avg_episode_length = total_episode_length / max(completed_episodes, 1)
    print("[INFO] Teacher-only evaluation complete.")
    print("[TEACHER-EVAL][FINAL] " + _format_eval_metrics(averaged, completed_episodes, avg_episode_length))


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir
    uses_frozen_llc_action = configure_frozen_llc_action(env_cfg, args_cli.locomotion_checkpoint, args_cli.task)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.teacher_only_eval:
        _run_teacher_only_eval(env, agent_cfg)
        env.close()
        return

    # save resume path before creating a new log_dir
    resume_path = None
    if args_cli.teacher_checkpoint is not None:
        resume_path = args_cli.teacher_checkpoint
    elif agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    elif agent_cfg.class_name == "DistillationRunner" and args_cli.locomotion_checkpoint is None:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    _install_resilient_runner_save(runner)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)

    # PPO paths load locomotion weights into model modules.
    # HLC navigation tasks load the same checkpoint inside FrozenLLCActionTerm before gym.make().
    if args_cli.locomotion_checkpoint is not None and not uses_frozen_llc_action:
        if isinstance(runner, OnPolicyRunner):
            _load_locomotion_checkpoint(runner, args_cli.locomotion_checkpoint, agent_cfg.device)
        else:
            raise ValueError("--locomotion_checkpoint is only supported for OnPolicyRunner tasks")

    # load the checkpoint
    if resume_path is not None:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # When loading a PPO checkpoint into DistillationRunner, the actor state_dict
        # contains distribution parameters (e.g. std_param) that the teacher MLPModel
        # does not have. Strip them before loading.
        if isinstance(runner, DistillationRunner):
            ckpt = torch.load(resume_path, weights_only=False)
            if "teacher_state_dict" in ckpt or "student_state_dict" in ckpt:
                runner.alg.load(ckpt, load_cfg=None, strict=True)
            elif "actor_state_dict" in ckpt:
                ckpt["actor_state_dict"] = {
                    k: v for k, v in ckpt["actor_state_dict"].items()
                    if not k.startswith("distribution.")
                }
                runner.alg.load(ckpt, load_cfg=None, strict=True)
        else:
            runner.load(resume_path)
    elif isinstance(runner, DistillationRunner):
        print(
            "[INFO] No distillation checkpoint was loaded for the student/teacher heads. "
            "Starting distillation from fresh navigation weights; only the frozen LLC(s) were initialized "
            "from --locomotion_checkpoint if provided."
        )

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
