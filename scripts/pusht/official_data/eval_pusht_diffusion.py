#!/usr/bin/env python
"""Evaluate a policy trained on the official PushT dataset and save rollout videos."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", default="/home/liyi/lerobot/outputs/train/official/pusht_diffusion_v3_5000")
    parser.add_argument("--output-dir", default="outputs/eval/official/pusht_diffusion_v3_5000")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-videos", type=int, default=10)
    parser.add_argument("--success-rate-threshold", type=float, default=50.0)
    parser.add_argument("--fail-on-unsuccessful", action="store_true")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--use-async-envs", action="store_true")
    parser.add_argument("--episode-length", type=int, default=300)
    parser.add_argument("--observation-size", type=int, default=384)
    return parser.parse_args()


def configure_hub(args: argparse.Namespace) -> None:
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    if args.hf_home:
        hf_home = Path(args.hf_home).expanduser()
        hf_home.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(hf_home)
        os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
        os.environ.setdefault("TORCH_HOME", str(hf_home / "torch"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def collect_episode_results(info: dict) -> list[dict]:
    results = []
    episode_index = 0
    for task_info in info.get("per_task", []):
        metrics = task_info.get("metrics", {})
        sum_rewards = list(metrics.get("sum_rewards", []))
        max_rewards = list(metrics.get("max_rewards", []))
        successes = list(metrics.get("successes", []))
        video_paths = list(metrics.get("video_paths", []))
        for task_episode_index, success in enumerate(successes):
            results.append(
                {
                    "episode_index": episode_index,
                    "task_group": task_info.get("task_group"),
                    "task_id": task_info.get("task_id"),
                    "task_episode_index": task_episode_index,
                    "success": bool(success),
                    "sum_reward": sum_rewards[task_episode_index]
                    if task_episode_index < len(sum_rewards)
                    else None,
                    "max_reward": max_rewards[task_episode_index]
                    if task_episode_index < len(max_rewards)
                    else None,
                    "video_path": video_paths[task_episode_index]
                    if task_episode_index < len(video_paths)
                    else None,
                }
            )
            episode_index += 1
    return results


def main() -> None:
    args = parse_args()
    configure_hub(args)

    import torch

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
    from lerobot.envs.configs import PushtEnv
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy_all
    from lerobot.utils.device_utils import get_safe_torch_device
    from lerobot.utils.random_utils import set_seed

    policy_path = Path(args.policy_path).expanduser()
    if not policy_path.exists():
        raise FileNotFoundError(f"policy path does not exist: {policy_path}")

    output_dir = Path(args.output_dir).expanduser()
    videos_dir = output_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_safe_torch_device(args.device, log=True)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    policy_cfg = PreTrainedConfig.from_pretrained(policy_path)
    policy_cfg.pretrained_path = policy_path
    policy_cfg.device = str(device)
    policy_cfg.use_amp = args.use_amp

    env_cfg = PushtEnv(
        episode_length=args.episode_length,
        observation_height=args.observation_size,
        observation_width=args.observation_size,
    )
    envs = make_env(env_cfg, n_envs=args.batch_size, use_async_envs=args.use_async_envs)

    try:
        policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=str(policy_path),
            preprocessor_overrides={
                "device_processor": {"device": str(policy.config.device)},
                "rename_observations_processor": {"rename_map": {}},
            },
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy_cfg)

        print(f"policy_path={policy_path}")
        print(f"output_dir={output_dir}")
        print(f"device={device} n_episodes={args.n_episodes} batch_size={args.batch_size}")
        print(f"observation_size={args.observation_size}")

        amp_ctx = torch.autocast(device_type=device.type) if args.use_amp else nullcontext()
        with torch.no_grad(), amp_ctx:
            info = eval_policy_all(
                envs=envs,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=args.n_episodes,
                max_episodes_rendered=args.max_videos,
                videos_dir=videos_dir,
                start_seed=args.seed,
                max_parallel_tasks=env_cfg.max_parallel_tasks,
            )
    finally:
        close_envs(envs)

    overall = info["overall"]
    episode_results = collect_episode_results(info)
    success_rate = float(overall.get("pc_success", 0.0))
    generated_videos = list(overall.get("video_paths", []))
    payload = {
        "args": vars(args),
        "env": asdict(env_cfg),
        "success_check": {
            "is_successful": success_rate >= args.success_rate_threshold,
            "success_rate": success_rate,
            "success_rate_threshold": args.success_rate_threshold,
            "generated_video_count": len(generated_videos),
            "generated_videos": generated_videos,
        },
        "episode_results": episode_results,
        "overall": overall,
        "per_group": info["per_group"],
        "per_task": info["per_task"],
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)

    print("overall_metrics=")
    print(json.dumps(overall, indent=2))
    print("success_check=")
    print(json.dumps(payload["success_check"], indent=2))
    print("episode_results=")
    print(json.dumps(episode_results, indent=2))
    print(f"metrics_path={metrics_path}")
    for video_path in generated_videos:
        print(f"video_path={video_path}")

    if args.fail_on_unsuccessful and not payload["success_check"]["is_successful"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
