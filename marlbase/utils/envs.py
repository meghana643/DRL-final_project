from functools import partial
import random

import gymnasium as gym
from omegaconf import DictConfig

from marlbase.utils import wrappers as mwrappers
from marlbase.utils.smaclite_wrapper import SMACliteWrapper
from marlbase.utils.foraging_obstacle_wrapper import ForagingObstacleWrapper


def _make_parallel_envs(
    name,
    parallel_envs,
    wrappers,
    time_limit,
    clear_info,
    observe_id,
    standardise_rewards,
    seed,
    enable_video,
    **kwargs,
):
    """
    Create multiple environments in parallel (AsyncVectorEnv).
    Ensures 'render_mode' is specified exactly once by popping it from kwargs
    (if present) and falling back to 'rgb_array' when enable_video=True.
    """
    num_obstacles = kwargs.pop("num_obstacles", 0)

    def _env_thunk(seed_):
        # Copy kwargs so each thunk can mutate safely
        local_kwargs = dict(kwargs)

        # Remove from kwargs if someone already inserted render_mode via Hydra/YAML
        rm = local_kwargs.pop("render_mode", None)
        chosen_rm = rm if rm is not None else ("rgb_array" if enable_video else None)

        if "smaclite" in name:
            import smaclite  # noqa: F401

            env = gym.make(
                name,
                seed=seed_,
                render_mode=chosen_rm,
                **local_kwargs,
            )
            env = SMACliteWrapper(env)
        else:
            env = gym.make(
                name,
                render_mode=chosen_rm,
                **local_kwargs,
            )

        if num_obstacles > 0:
            env = ForagingObstacleWrapper(env, num_obstacles=num_obstacles, seed=seed_)

        if clear_info:
            env = mwrappers.ClearInfo(env)
        if time_limit:
            env = gym.wrappers.TimeLimit(env, time_limit)
        env = mwrappers.RecordEpisodeStatistics(env)
        if observe_id:
            env = mwrappers.ObserveID(env)
        if standardise_rewards:
            env = mwrappers.StandardiseReward(env)
        if wrappers is not None:
            for wrapper in wrappers:
                wrapper_cls = (
                    getattr(mwrappers, wrapper)
                    if hasattr(mwrappers, wrapper)
                    else getattr(gym.wrappers, wrapper)
                )
                env = wrapper_cls(env)

        env.reset(seed=seed_)
        return env

    if seed is None:
        seed = random.randint(0, 99999)

    envs = gym.vector.AsyncVectorEnv(
        [partial(_env_thunk, seed + i) for i in range(parallel_envs)]
    )
    return envs


def _make_env(
    name,
    time_limit,
    clear_info,
    observe_id,
    standardise_rewards,
    wrappers,
    seed,
    enable_video,
    **kwargs,
):
    """
    Create a single environment.
    Same render_mode de-duplication as in _make_parallel_envs.
    """
    # Copy kwargs so we can mutate without side effects
    local_kwargs = dict(kwargs)
    num_obstacles = local_kwargs.pop("num_obstacles", 0)
    print("num_obstacles---------------------------------------------------:", num_obstacles)

    # Ensure we only pass render_mode once
    rm = local_kwargs.pop("render_mode", None)
    chosen_rm = rm if rm is not None else ("rgb_array" if enable_video else None)

    if "smaclite" in name:
        import smaclite  # noqa: F401

        env = gym.make(
            name,
            seed=seed,
            render_mode=chosen_rm,
            **local_kwargs,
        )
        env = SMACliteWrapper(env)
    else:
        env = gym.make(
            name,
            render_mode=chosen_rm,
            **local_kwargs,
        )

    if num_obstacles > 0:
        env = ForagingObstacleWrapper(env, num_obstacles=num_obstacles, seed=seed)

    if clear_info:
        env = mwrappers.ClearInfo(env)
    if time_limit:
        env = gym.wrappers.TimeLimit(env, time_limit)
    env = mwrappers.RecordEpisodeStatistics(env)
    if observe_id:
        env = mwrappers.ObserveID(env)
    if standardise_rewards:
        env = mwrappers.StandardiseReward(env)
    if wrappers is not None:
        for wrapper in wrappers:
            wrapper_cls = (
                getattr(mwrappers, wrapper)
                if hasattr(mwrappers, wrapper)
                else getattr(gym.wrappers, wrapper)
            )
            env = wrapper_cls(env)

    # If you want deterministic seeding here too, uncomment:
    # env.reset(seed=seed)
    return env


def make_env(seed, enable_video=False, **env_config):
    """
    Public factory called by Hydra. DictConfig -> python args.
    Uses parallel envs if 'parallel_envs' is set; otherwise a single env.
    """
    env_config = DictConfig(env_config)
    if "parallel_envs" in env_config and env_config.parallel_envs:
        return _make_parallel_envs(**env_config, enable_video=enable_video, seed=seed)
    return _make_env(**env_config, enable_video=enable_video, seed=seed)
