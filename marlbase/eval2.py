import os
from pathlib import Path
import time
import random
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig

from ac.model import PPONetwork

try:
    import gymnasium as gym
    from gymnasium.wrappers import TimeLimit as GymTimeLimit
except ImportError:
    import gym
    from gym.wrappers import TimeLimit as GymTimeLimit

# Allow ${random:...} in configs
OmegaConf.register_new_resolver("random", lambda x: os.urandom(x).hex())


# ===========================================================
# Utility helpers
# ===========================================================
def _get_num_agents_from_env(env, obs_first_step):
    """Infer number of agents robustly (works with lbforaging)."""
    if isinstance(obs_first_step, (list, tuple)):
        return len(obs_first_step)
    num = getattr(getattr(env, "unwrapped", env), "players", None)
    if isinstance(num, int) and num > 0:
        return num
    return 1


def _ensure_batch1_hidden(h):
    """Force hidden state shape to (num_layers, 1, hidden) if needed."""
    if h is None:
        return None
    if h.dim() != 3:
        raise RuntimeError(f"Hidden state has wrong rank: expected 3D, got {h.shape}")
    if h.size(1) == 1:
        return h
    return h[:, :1, :]


def unwrap_to_base(env):
    e = env
    seen = set()
    while hasattr(e, "env") and id(e.env) not in seen:
        seen.add(id(e))
        e = e.env
    return e


def force_horizon(env, horizon: int):
    """
    Force environment to run for `horizon` steps before truncation.
    Works even if inner wrappers have 50-step defaults.
    """
    base = unwrap_to_base(env)
    try:
        if hasattr(base, "_max_episode_steps"):
            base._max_episode_steps = int(horizon)
        if getattr(base, "spec", None) is not None:
            base.spec.max_episode_steps = int(horizon)
    except Exception:
        pass
    env = GymTimeLimit(env, max_episode_steps=int(horizon))
    return env


# ===========================================================
# Main
# ===========================================================
@hydra.main(config_path="configs", config_name="eval", version_base="1.3")
def main(cfg: DictConfig):
    # ========= paths/config =========
    run_dir = Path(__file__).parent / cfg.path
    assert run_dir.is_dir(), f"Invalid path: {run_dir}"
    cfg_file = run_dir / "config.yaml"
    assert cfg_file.exists(), f"Missing {cfg_file}"
    run_cfg = OmegaConf.load(cfg_file)

    # ========= env =========
    if "parallel_envs" in run_cfg.env:
        del run_cfg.env.parallel_envs  # eval is single-env / live render

    env = hydra.utils.call(run_cfg.env, enable_video=False, seed=cfg.seed)

    # ---- force the horizon to 75 (or CLI override) ----
    time_limit = int(getattr(cfg, "time_limit", 75))
    env = force_horizon(env, time_limit)

    # Print verification
    tl_attr = getattr(getattr(env, "unwrapped", env), "time_limit", None)
    print(f"[INFO] Environment created. Forcing horizon={time_limit}")
    if hasattr(env, "_max_episode_steps"):
        print(f"[DEBUG] Top wrapper max_episode_steps={env._max_episode_steps}")
    if tl_attr:
        print(f"[DEBUG] Env.unwrapped.time_limit={tl_attr}")

    # ========= seeds =========
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

    # ========= device =========
    device_str = str(run_cfg.algorithm.model.device).lower()
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available. Falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"[INFO] Using device: {device}")

    # ========= checkpoint =========
    if cfg.get("load_step") is not None:
        load_step = int(cfg.load_step)
    else:
        ckpts = sorted((run_dir / "checkpoints").glob("model_s*.pt"))
        assert ckpts, f"No checkpoints in {run_dir / 'checkpoints'}"
        load_step = int(ckpts[-1].stem.split("_")[-1][1:])
    ckpt_path = run_dir / "checkpoints" / f"model_s{load_step}.pt"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    print(f"[INFO] Loading checkpoint: {ckpt_path}")

    # ========= model =========
    policy = PPONetwork(
        obs_space=env.observation_space,
        action_space=env.action_space,
        cfg=run_cfg.algorithm,
        actor=run_cfg.algorithm.model.actor,
        critic=run_cfg.algorithm.model.critic,
        device=device_str,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(state, strict=True)
    policy.eval()
    print("[INFO] Model loaded. Running STOCHASTIC policy (sampling).")

    # ========= rollout (single episode) =========
    obs, _ = env.reset()
    num_agents = _get_num_agents_from_env(env, obs)
    print(f"[DEBUG] Players: {num_agents}")

    actor_hiddens = policy.init_actor_hiddens(batch_size=1)
    actor_hiddens = [(_ensure_batch1_hidden(h.to(device)) if h is not None else None)
                     for h in actor_hiddens]
    assert len(actor_hiddens) == num_agents, (
        f"Hidden states per agent mismatch: {len(actor_hiddens)} vs {num_agents}"
    )

    done = False
    step = 0
    print("\n[INFO] Starting stochastic rollout...\n")

    while not done:
        # ---- render live ----
        try:
            env.unwrapped.render()
        except Exception as e:
            print(f"[WARN] render() failed: {e}")

        # ---- per-agent obs -> tensors ----
        obs_list = obs if isinstance(obs, (list, tuple)) else [obs]
        if len(obs_list) != len(actor_hiddens):
            n = min(len(obs_list), len(actor_hiddens))
            obs_list = list(obs_list)[:n]
            actor_hiddens = actor_hiddens[:n]

        obs_tensors = [
            torch.as_tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
            for o in obs_list if o is not None
        ]
        if not obs_tensors:
            print(f"[WARN] Empty observations at step {step}. Stopping.")
            break

        actor_hiddens = [_ensure_batch1_hidden(h) if h is not None else None
                         for h in actor_hiddens]

        # ---- policy act (stochastic) ----
        with torch.no_grad():
            actions, actor_hiddens = policy.act(obs_tensors, actor_hiddens)

        actions_flat = actions.detach().to("cpu").view(-1).tolist()
        actions_list = [int(a) for a in actions_flat]
        if len(actions_list) != num_agents:
            n = min(len(actions_list), num_agents)
            actions_list = actions_list[:n] + [0] * (num_agents - n)

        # ---- env step ----
        obs, reward, terminated, truncated, info = env.step(actions_list)
        step += 1

        print(f"Step {step}: Actions={actions_list}  Reward={reward}")

        done = bool(terminated or truncated)

        # Slow down a bit for visualization
        time.sleep(0.08)

        if done:
            reason_bits = []
            if terminated:
                reason_bits.append("terminated (task complete)")
            if truncated:
                reason_bits.append("truncated (time limit reached)")
            reason = " and ".join(reason_bits) if reason_bits else "done"
            print(f"\n[INFO] Episode ended after {step} steps — {reason}\n")

    env.close()
    print("[INFO] Evaluation complete.")


if __name__ == "__main__":
    main()
