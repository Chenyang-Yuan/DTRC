"""Verify final-model checksums and image-to-action interfaces."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dtrc.agent import load_agent
from dtrc.backends.lewm import load_lewm_backend


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_fingerprint(state):
    digest = hashlib.sha256()
    for name in (
        "policy",
        "rep",
        "rep_t",
        "quasi",
        "quasi_t",
        "latent_step",
        "dyn",
        "support",
    ):
        for key, value in sorted(state[name].items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(
                (
                    name + "." + key + str(tensor.dtype) + str(tuple(tensor.shape))
                ).encode()
            )
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def resolve_asset(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("model index contains a path outside the asset root")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = args.root.resolve()
    index = json.loads((root / "models/model_index.json").read_text())
    torch.set_num_threads(1)
    torch.manual_seed(0)
    for item in index["models"]:
        checkpoint = resolve_asset(root, item["checkpoint"])
        encoder_dir = resolve_asset(root, item["encoder"])
        for path, expected in (
            (checkpoint, item["checkpoint_sha256"]),
            (encoder_dir / "weights.pt", item["encoder_weights_sha256"]),
            (encoder_dir / "config.json", item["encoder_config_sha256"]),
        ):
            if sha256_file(path) != expected:
                raise ValueError(f"checksum mismatch: {path.name} ({item['task']})")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if tensor_fingerprint(state) != item["tensor_fingerprint_sha256"]:
            raise ValueError(f"parameter fingerprint mismatch: {item['task']}")
        agent, meta = load_agent(state, args.device, "single")
        encoder = load_lewm_backend(encoder_dir, args.device)
        if encoder.weights_sha256 != meta["lewm_weights_sha256"]:
            raise ValueError(f"encoder does not match checkpoint: {item['task']}")
        frames = torch.zeros(2, 64, 64, 3, dtype=torch.uint8)
        frames[1] = 127
        with torch.inference_mode():
            features = encoder.encode_uint8(frames)
            mean = torch.from_numpy(
                np.asarray(meta["feature_mean"], dtype=np.float32)
            ).to(args.device)
            scale = (
                torch.from_numpy(np.asarray(meta["feature_std"], dtype=np.float32))
                .to(args.device)
                .clamp_min(1e-6)
            )
            normalized = (features - mean) / scale
            current, goal = normalized[:1], normalized[1:]
            horizon = torch.ones(1, device=args.device)
            action = agent.act(current, goal, horizon)
            distance = agent.distance(current, goal)
        if action.shape != (1, state["cfg"]["chunk"], item["action_dim"]):
            raise ValueError(f"unexpected action shape: {item['task']}")
        if not (
            torch.isfinite(features).all()
            and torch.isfinite(action).all()
            and torch.isfinite(distance).all()
        ):
            raise ValueError(f"nonfinite model output: {item['task']}")
        print(
            f"[OK] {item['task']}: step={state['step']}, action_dim={item['action_dim']}"
        )
        del agent, encoder, state
    print(
        f"Verified {len(index['models'])} final models. No simulator evaluation was run."
    )


if __name__ == "__main__":
    main()
