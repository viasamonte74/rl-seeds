from pathlib import Path
import sys
import types

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


def downsample_depth(depth: np.ndarray, size: int) -> np.ndarray:
    """Block-pool depth to size×size×1. Must match RL/common.py and RL/agent_template.py."""
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    h, w = int(d.shape[0]), int(d.shape[1])
    if h == size and w == size:
        out = d
    elif h < size or w < size:
        ys = (np.arange(size) * h // size).clip(0, max(0, h - 1))
        xs = (np.arange(size) * w // size).clip(0, max(0, w - 1))
        out = d[ys][:, xs]
    else:
        bh, bw = h // size, w // size
        cropped = d[: size * bh, : size * bw]
        blocks = cropped.reshape(size, bh, size, bw).transpose(0, 2, 1, 3)
        flat = blocks.reshape(size, size, bh * bw)
        out = flat.min(axis=-1)
        if size >= 256 and bh * bw >= 4:
            p10 = np.percentile(flat, 10, axis=-1)
            p90 = np.percentile(flat, 90, axis=-1)
            mixed = (p90 - p10) > 0.25
            if np.any(mixed):
                mid = p10 + 0.5 * (p90 - p10)
                far_only = np.where(flat >= mid[..., None], flat, 1.0)
                out = np.where(mixed, far_only.min(axis=-1), out)
    return np.ascontiguousarray(out.reshape(size, size, 1), dtype=np.float32)


class InterceptorFeaturesExtractor(BaseFeaturesExtractor):
    """CNN on NCHW depth, LayerNorm+MLP on state, extra head on the XY clue.

    Lives in this file so SAC.load works inside the packaged drone_agent.py.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        depth_shape = observation_space["depth"].shape
        state_dim = int(observation_space["state"].shape[0])
        height, width = int(depth_shape[0]), int(depth_shape[1])
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_cnn = int(self.cnn(torch.zeros(1, 1, height, width)).shape[1])
        self.state_norm = nn.LayerNorm(state_dim)
        self.state_mlp = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.clue_mlp = nn.Sequential(nn.Linear(2, 32), nn.ReLU())
        self.merge = nn.Sequential(nn.Linear(n_cnn + 128 + 32, features_dim), nn.ReLU())

    def forward(self, observations):
        depth = observations["depth"]
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth.permute(0, 3, 1, 2).contiguous()
        state = observations["state"]
        clue = state[..., -2:]
        fused = torch.cat(
            [self.cnn(depth), self.state_mlp(self.state_norm(state)), self.clue_mlp(clue)],
            dim=1,
        )
        return self.merge(fused)


def _register_extractor_for_load():
    """SAC.zip pickles this class under agent_sac/common_sac/drone_agent."""
    for name in ("agent_sac", "common_sac", "drone_agent", Path(__file__).stem):
        mod = sys.modules.get(name)
        if mod is None:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        setattr(mod, "InterceptorFeaturesExtractor", InterceptorFeaturesExtractor)


def _resolve_policy_path() -> Path:
    here = Path(__file__).resolve().parent
    for name in ("sac_policy.zip", "ppo_policy.zip"):
        path = here / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"No SAC weights next to {here}: expected sac_policy.zip"
    )


class DroneFlightController:
    """SAC interceptor controller. Eval is the neural net only (no teacher).

    Loads sac_policy.zip packaged next to this file. Depth is min-pooled to
    the resolution the policy was trained on; extra keys (rgb) are ignored.
    """

    def __init__(self):
        policy_path = _resolve_policy_path()
        _register_extractor_for_load()
        self._model = SAC.load(str(policy_path), device="cpu")
        self._depth_size = int(self._model.observation_space["depth"].shape[0])
        self._low = self._model.action_space.low
        self._high = self._model.action_space.high

    def _policy_obs(self, depth, state):
        return {
            "depth": downsample_depth(depth, self._depth_size),
            "state": np.asarray(state, dtype=np.float32),
        }

    def _predict(self, depth, state):
        action, _ = self._model.predict(self._policy_obs(depth, state), deterministic=True)
        return np.clip(action, self._low, self._high)

    def act(self, observation):
        depth = np.asarray(observation["depth"])
        state = np.asarray(observation["state"])
        if state.ndim == 2:
            actions = [self._predict(depth[i], state[i]) for i in range(state.shape[0])]
            return np.stack(actions).astype(np.float32)
        return np.asarray(self._predict(depth, state), dtype=np.float32)

    def reset(self):
        pass
