from pathlib import Path

import numpy as np
from stable_baselines3 import PPO


def downsample_depth(depth: np.ndarray, size: int) -> np.ndarray:
    """Block min-pool depth to size×size×1. Must match RL/common.py.

    Env depth is 0=near, 1=far. Min-pool keeps the nearest surface in each
    block so a few-pixel interceptor target still appears after 1024→256.
    """
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
        out = cropped.reshape(size, bh, size, bw).min(axis=(1, 3))
    return np.ascontiguousarray(out.reshape(size, size, 1), dtype=np.float32)


class DroneFlightController:
    """Baseline PPO controller: loads the ppo_policy.zip packaged next to this file.

    Works for every challenge family: depth frames are min-pooled to the
    resolution the policy was trained on, extra observation keys (rgb) are
    ignored, and multi-drone observations run the same policy once per drone.
    """

    def __init__(self):
        policy_path = Path(__file__).resolve().parent / "ppo_policy.zip"
        self._model = PPO.load(str(policy_path), device="cpu")
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
