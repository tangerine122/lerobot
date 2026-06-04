#!/usr/bin/env python
"""Train an ACT policy on the official ALOHA transfer-cube v3.0 dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-repo-id", default="lerobot/aloha_sim_transfer_cube_human")
    parser.add_argument("--dataset-revision", default="v3.0")
    parser.add_argument("--output-dir", default="outputs/train/aloha_transfer_cube_act_v3")
    parser.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--hf-home", default=os.environ.get("HF_HOME"))
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--video-backend", choices=["pyav", "torchcodec", "video_reader"], default="pyav")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--n-action-steps", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
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


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0.0]
    return [i / fps for i in delta_indices]


def main() -> None:
    args = parse_args()
    configure_hub(args)

    import torch

    from lerobot.configs import FeatureType
    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act import ACTConfig, ACTPolicy
    from lerobot.utils.feature_utils import dataset_to_policy_features

    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, revision=args.dataset_revision)
    features = dataset_to_policy_features(metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        optimizer_lr=args.lr,
        optimizer_weight_decay=args.weight_decay,
    )
    policy = ACTPolicy(cfg).to(device)
    policy.train()
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=metadata.stats)

    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, metadata.fps),
    }
    delta_timestamps.update(
        {
            key: make_delta_timestamps(cfg.observation_delta_indices, metadata.fps)
            for key in cfg.image_features
        }
    )

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        revision=args.dataset_revision,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())

    print(f"dataset={args.dataset_repo_id}@{args.dataset_revision}")
    print(f"frames={dataset.num_frames} episodes={dataset.num_episodes} fps={metadata.fps}")
    print(f"features={list(metadata.features.keys())}")
    print(f"device={device} batch_size={args.batch_size} steps={args.steps}")
    print(f"chunk_size={cfg.chunk_size} n_action_steps={cfg.n_action_steps}")
    print(f"video_backend={args.video_backend}")
    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}")
    print(f"HF_HOME={os.environ.get('HF_HOME')}")

    step = 0
    while step < args.steps:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % args.log_freq == 0:
                print(f"step={step} loss={loss.item():.6f}")
            step += 1
            if step >= args.steps:
                break

    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    print(f"saved_policy={output_dir}")


if __name__ == "__main__":
    main()
