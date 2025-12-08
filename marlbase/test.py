import os
from pathlib import Path
import time
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig

from ac.model import PPONetwork
from utils.envs import make_env  # or your appropriate import

# Register resolver
OmegaConf.register_new_resolver("random", lambda x: os.urandom(x).hex())


@hydra.main(config_path="configs", config_name="eval", version_base="1.3")
def main(cfg: DictConfig):
    # --- Config + checkpoint paths ---
    path = Path(__file__).parent / cfg.path
    assert path.exists() and path.is_dir(), f"Path {path} is invalid."

    config_path = path / "config.yaml"
    assert config_path.exists(), f"Missing config: {config_path}"
    run_config = OmegaConf.load(config_path)

    if "parallel_envs" in run_config.env:
        del run_config.env.parallel_envs

    # Create env
    env = hydra.utils.call(run_config.env, enable_video=True, seed=cfg.seed)

    torch.set_num_threads(1)
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    # --- Checkpoint selection ---
    if cfg.load_step is not None:
        load_step = cfg.load_step
    else:
        load_step = max(
            int(f.stem.split("_")[-1][1:])
            for f in (path / "checkpoints").glob("model_s*.pt")
        )
    ckpt_path = path / "checkpoints" / f"model_s{load_step}.pt"
    assert ckpt_path.exists(), f"Checkpoint not found: {ckpt_path}"
    print(f"Loading model from {ckpt_path}")

    # --- Load PPO policy ---
    policy = PPONetwork(
        obs_space=env.observation_space,
        action_space=env.action_space,
        cfg=run_config.algorithm,
        actor=run_config.algorithm.model.actor,
        critic=run_config.algorithm.model.critic,
        device=run_config.algorithm.model.device,
    )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    policy.load_state_dict(state_dict)
    policy.eval()

    # === Rollout with action printing ===
    obs, _ = env.reset()
    actor_hiddens = policy.init_actor_hiddens(batch_size=1)
    done = False
    step = 0

    env.unwrapped.render()  # Optional: render the environment
    time.sleep(3)  # Slow down rendering

    while not done:
        env.unwrapped.render()  # Optional: render the environment
        #time.sleep(1) 
        obs_tensors = [
            torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            for o in obs
        ]

        with torch.no_grad():
            actions, actor_hiddens = policy.act(obs_tensors, actor_hiddens)

        # --- FIX: Convert to Python ints ---
        actions_np = actions.squeeze(0).cpu().numpy()
        actions_list = [int(a) for a in actions_np.flatten()]

        print(f"Step {step}:")
        for i, action in enumerate(actions_list):
            print(f"  Agent {i}: Action {action}")

        obs, _, done, _, _ = env.step(actions_list)
        step += 1

    print("[INFO] Evaluation complete. Check video output if enabled.")


if __name__ == "__main__":
    main()
