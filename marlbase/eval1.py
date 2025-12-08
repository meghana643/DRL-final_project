import os
from pathlib import Path
import csv, time
from typing import Any, Sequence, Mapping

import hydra
import numpy as np
from omegaconf import OmegaConf, DictConfig, open_dict
import torch

try:
    import gymnasium as gym
    from gymnasium.wrappers import TimeLimit as GymTimeLimit
except ImportError:
    import gym as gym
    from gym.wrappers import TimeLimit as GymTimeLimit

# ================================================
# Helpers
# ================================================

def _to_list(x: Any) -> list:
    try:
        import numpy as np
        import torch as th
    except Exception:
        np = None
        th = None

    if isinstance(x, Mapping):
        keys = sorted(x.keys())
        out = []
        for k in keys:
            v = _to_list(x[k])
            out.append(v[0] if isinstance(v, list) and len(v) == 1 else v)
        return out

    if 'th' in globals() and isinstance(x, th.Tensor):
        x = x.detach().cpu().numpy()

    if 'np' in globals() and np is not None and isinstance(x, np.ndarray):
        return x.reshape(-1).tolist()

    if isinstance(x, Sequence) and not isinstance(x, (str, bytes, bytearray)):
        return list(x)

    return [x]

def _format_list(nums):
    out = []
    for a in nums:
        if isinstance(a, (int, float)) and float(a).is_integer():
            out.append(str(int(a)))
        else:
            # compact float
            out.append(f"{a:.2f}" if isinstance(a, float) else str(a))
    return "[" + ", ".join(out) + "]"

def unwrap_to_base(env):
    e = env
    seen = set()
    while hasattr(e, "env") and id(e.env) not in seen:
        seen.add(id(e))
        e = e.env
    return e

def force_horizon(env, horizon: int):
    """
    Best-effort: set base env's internal horizon, then wrap with TimeLimit(horizon).
    Returns wrapped env and a dict of diagnostics.
    """
    diag = {}
    base = unwrap_to_base(env)

    # Try to print/adjust spec
    spec_steps = None
    try:
        spec_steps = getattr(getattr(base, "spec", None), "max_episode_steps", None)
    except Exception:
        spec_steps = None
    diag["before.spec.max_episode_steps"] = spec_steps

    # Try to set hidden attributes some envs use
    try:
        if hasattr(base, "_max_episode_steps"):
            base._max_episode_steps = int(horizon)  # type: ignore[attr-defined]
            diag["set_base._max_episode_steps"] = True
        else:
            diag["set_base._max_episode_steps"] = False
    except Exception:
        diag["set_base._max_episode_steps"] = "error"

    # Wrap with hard TimeLimit cap at the top (guarantee)
    env_wrapped = GymTimeLimit(env, max_episode_steps=int(horizon))

    # After wrap: report top wrapper’s max
    try:
        diag["after.wrapper.max_episode_steps"] = env_wrapped._max_episode_steps  # type: ignore[attr-defined]
    except Exception:
        diag["after.wrapper.max_episode_steps"] = None

    return env_wrapped, diag

# ================================================
# Step logger with slowdown + end-reason prints
# ================================================

class StepLogger(gym.Wrapper):
    """
    Prints:
      Step N: Actions=[...]  Reward=[...]
    Throttling: print_every, sleep_ms (per printed step), pause_every
    Also prints episode-end reason (terminated vs truncated) and steps_taken.
    """
    def __init__(self, env, print_every: int = 1, sleep_ms: int = 0, pause_every: int = 0, to_csv: str = None):
        super().__init__(env)
        self.t = 0
        self.print_every = max(1, int(print_every))
        self.sleep_ms = max(0, int(sleep_ms))
        self.pause_every = max(0, int(pause_every))
        self.printed_count = 0

        self._csv_file = None
        self._writer = None
        if to_csv:
            self._csv_file = open(to_csv, "w", newline="")
            self._writer = csv.writer(self._csv_file)
            self._writer.writerow(["step", "actions", "rewards", "terminated", "truncated"])

    def reset(self, *args, **kwargs):
        if self.t > 0:
            # episode ended before reset() was called again
            print(f"[episode] reset → new episode\n", flush=True)
        self.t = 0
        self.printed_count = 0
        obs, info = self.env.reset(*args, **kwargs)
        return obs, info

    def step(self, action):
        self.t += 1
        obs, reward, terminated, truncated, info = self.env.step(action)

        if (self.t % self.print_every) == 0:
            acts = _to_list(action)
            rews = _to_list(reward)
            print(f"Step {self.t}: Actions={_format_list(acts)}  Reward={_format_list(rews)}", flush=True)
            self.printed_count += 1
            if self._writer:
                self._writer.writerow([self.t, acts, rews, terminated, truncated])
            if self.sleep_ms > 0:
                time.sleep(self.sleep_ms / 1000.0)
            if self.pause_every > 0 and (self.printed_count % self.pause_every) == 0:
                try:
                    input("⏸ Press Enter to continue… ")
                except EOFError:
                    pass

        # When episode ends, print reason immediately
        ended = False
        # Gymnasium returns bool or dicts; normalize to any True
        def any_true(x):
            if isinstance(x, Mapping):
                return any(bool(v) for v in x.values())
            return bool(x)

        if any_true(terminated) or any_true(truncated):
            ended = True
            reason = []
            if any_true(terminated):
                reason.append("terminated")
            if any_true(truncated):
                reason.append("truncated(TimeLimit)")
            print(f"[episode] end after {self.t} steps → {' + '.join(reason)}", flush=True)

        return obs, reward, terminated, truncated, info

    def close(self):
        if self._csv_file:
            self._csv_file.close()
        return super().close()

# ================================================
# Main
# ================================================

OmegaConf.register_new_resolver("random", lambda x: os.urandom(x).hex())

@hydra.main(config_path="configs", config_name="eval", version_base="1.3")
def main(cfg: DictConfig):
    path = Path(__file__).parent / cfg.path
    assert path.exists(), f"Path {path} does not exist."
    assert path.is_dir(), f"Path {path} is not a directory."

    config_path = path / "config.yaml"
    assert config_path.exists(), f"Config file {config_path} does not exist."
    run_config = OmegaConf.load(config_path)

    with open_dict(run_config):
        if "parallel_envs" in run_config.env:
            del run_config.env["parallel_envs"]

        # CLI overrides
        if getattr(cfg, "time_limit", None) is not None:
            run_config.env.time_limit = int(cfg.time_limit)

        if getattr(cfg, "steps", None) is not None:
            steps_val = int(cfg.steps)
            run_config.algorithm["steps"] = steps_val
            run_config.algorithm["eval_steps"] = steps_val
        else:
            if "eval_steps" not in run_config.algorithm:
                run_config.algorithm["eval_steps"] = int(run_config.env.time_limit)

    # Build env
    env = hydra.utils.call(run_config.env, enable_video=True, seed=cfg.seed)

    # HARD-SET the horizon to match desired time_limit
    desired_horizon = int(run_config.env.time_limit)
    env, diag = force_horizon(env, desired_horizon)

    print("──────────────────────────────────────────────")
    print(f"[eval] desired time_limit:     {desired_horizon}")
    print(f"[eval] before.spec.max_steps:  {diag.get('before.spec.max_episode_steps')}")
    print(f"[eval] set_base._max_steps?:   {diag.get('set_base._max_episode_steps')}")
    print(f"[eval] wrapper.max_steps now:  {diag.get('after.wrapper.max_episode_steps')}")
    print(f"[eval] eval_steps:             {run_config.algorithm.get('eval_steps','N/A')}")
    print(f"[eval] eval_episodes:          {run_config.algorithm.get('eval_episodes','N/A')}")
    print("──────────────────────────────────────────────")

    # Wrap with step logger; slow it so you can see it
    env = StepLogger(env, print_every=1, sleep_ms=150, pause_every=0, to_csv=None)

    # Seeds/threads
    torch.set_num_threads(1)
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    # Switch to eval target
    if hasattr(run_config.algorithm, "_target_"):
        run_config.algorithm._target_ = run_config.algorithm._target_.replace("train", "eval")

    # Pick checkpoint
    if cfg.load_step is not None:
        load_step = int(cfg.load_step)
    else:
        ckpts = list((path / "checkpoints").glob("model_s*.pt"))
        assert len(ckpts) > 0, f"No checkpoints found under {path / 'checkpoints'}"
        load_step = max(int(f.stem.split("_")[-1][1:]) for f in ckpts)

    ckpt_path = path / "checkpoints" / f"model_s{load_step}.pt"
    assert ckpt_path.exists(), f"Checkpoint {ckpt_path} does not exist."

    print(f"[eval] using checkpoint: {ckpt_path}")

    # Run eval
    hydra.utils.call(run_config.algorithm, env, ckpt_path, _recursive_=False)

if __name__ == "__main__":
    main()
