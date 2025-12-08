from datetime import timedelta
from hashlib import sha256
import json
import logging
import math
import time
from typing import Dict, List

import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd


def squash_info(info):
    new_info = {}
    keys = set([k for i in info for k in i.keys()])
    keys.discard("TimeLimit.truncated")
    keys.discard("terminal_observation")
    for key in keys:
        values = [d[key] for d in info if key in d]
        if len(values) == 1:
            new_info[key] = values[0]
            continue

        mean = np.mean([np.array(v).sum() for v in values])
        std = np.std([np.array(v).sum() for v in values])

        split_key = key.rsplit("/", 1)
        mean_key = split_key[:]
        std_key = split_key[:]
        mean_key[-1] = "mean_" + mean_key[-1]
        std_key[-1] = "std_" + std_key[-1]

        new_info["/".join(mean_key)] = mean
        new_info["/".join(std_key)] = std
    return new_info


class Logger:
    def __init__(self, project_name, cfg: DictConfig) -> None:
        self.config_hash = sha256(
            json.dumps(
                {k: v for k, v in OmegaConf.to_container(cfg).items() if k != "seed"},
                sort_keys=True,
            ).encode("utf8")
        ).hexdigest()[-10:]

        self._total_steps = cfg.algorithm.total_steps
        self._start_time = time.time()
        self._prev_time = None
        self._prev_steps = (0, 0)  # steps (updates) and env_samples

    def log_metrics(self, metrics: List[Dict]): ...

    def print_progress(self, updates, steps, mean_returns, episodes):
        self.info(f"Updates {updates}, Environment timesteps {steps}")

        time_now = time.time()

        elapsed_wallclock = time_now - self._prev_time[0] if self._prev_time else None
        elapsed_cpu = (
            time.process_time() - self._prev_time[1] if self._prev_time else None
        )
        elapsed_from_start = timedelta(seconds=math.ceil((time_now - self._start_time)))

        completed = steps / self._total_steps

        if elapsed_wallclock:
            ups = (updates - self._prev_steps[0]) / elapsed_wallclock
            fps = (steps - self._prev_steps[1]) / elapsed_wallclock
            self.info(f"UPS: {ups:.2f}, FPS: {fps:.2f} (wall time)")

            # ups = (updates - self._prev_steps[0]) / elapsed_cpu
            # fps = (steps - self._prev_steps[1]) / elapsed_cpu
            # self.info(f"UPS: {ups:.2f}, FPS: {fps:.2f} (cpu time)")

            eta = elapsed_from_start * (1 - completed) / completed
            eta = timedelta(seconds=math.ceil(eta.total_seconds()))
            self.info(f"Elapsed Time: {elapsed_from_start}")
            self.info(f"Estim. Time Left: {eta}")

        self.info(f"Completed: {100*completed:.2f}%")

        self._prev_steps = (updates, steps)
        self._prev_time = time.time(), time.process_time()

        self.info(f"Last {episodes} episodes with mean returns: {mean_returns:.3f}")
        self.info("-------------------------------------------")

    def watch(self, model):
        self.debug(model)

    def debug(self, *args, **kwargs):
        return logging.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        return logging.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        return logging.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        return logging.error(*args, **kwargs)

    def critical(self, *args, **kwargs):
        return logging.critical(*args, **kwargs)

    def get_state(self):
        return None


class WandbLogger(Logger):
    def __init__(self, project_name, cfg: DictConfig) -> None:
        import wandb

        super().__init__(project_name, cfg)
        self._run = wandb.init(
            project=project_name,
            config=OmegaConf.to_container(cfg),
            monitor_gym=True,
            group=self.config_hash,
        )

    def log_metrics(self, metrics: List[Dict]):
        d = squash_info(metrics)
        self._run.log(d)

        self.print_progress(
            d["updates"],
            d["environment_steps"],
            d["mean_episode_returns"],
            len(metrics) - 1,
        )

    def watch(self, model):
        self.debug(model)
        self._run.watch(model)


class FileSystemLogger(Logger):
    def __init__(self, project_name, cfg):
        super().__init__(project_name, cfg)

        self.results_path = "results.csv"
        self.config_path = "config.yaml"
        with open(self.config_path, "w") as f:
            OmegaConf.save(cfg, f)

    def log_metrics(self, metrics):
        d = squash_info(metrics)

        # ---- strict normalization: keys -> str, values -> python scalars/strings ----
        def _to_scalar(v):
            # numpy scalars
            if isinstance(v, (np.floating, np.integer, np.bool_)):
                return v.item()
            # numpy arrays
            if isinstance(v, np.ndarray):
                if v.ndim == 0:
                    return v.item()
                # reduce arrays to mean (float)
                return float(np.asarray(v, dtype=np.float64).mean())
            # lists/tuples -> try numeric mean else JSON
            if isinstance(v, (list, tuple)):
                try:
                    return float(np.asarray(v, dtype=np.float64).mean())
                except Exception:
                    return json.dumps(v)
            # dicts -> JSON
            if isinstance(v, dict):
                return json.dumps(v)
            # already a python scalar or string/bool/None
            return v

        # sanitize keys & values
        safe = {}
        for k, v in d.items():
            ks = str(k)
            safe[ks] = _to_scalar(v)

        # ensure required fields exist (and are scalars)
        safe.setdefault("environment_steps", 0)
        safe.setdefault("updates", 0)
        safe.setdefault("mean_episode_returns", 0.0)

        # build ordered columns and a single sanitized row
        cols = ["environment_steps"] + sorted([c for c in safe.keys() if c != "environment_steps"])
        row = {c: safe.get(c, np.nan) for c in cols}

        # final safety assertions (helpful if something weird sneaks in)
        for ck in cols:
            assert isinstance(ck, str), f"Non-string column name: {type(ck)} -> {ck!r}"
        for cv in row.values():
            assert isinstance(cv, (int, float, str, bool, type(np.nan))), f"Bad value type: {type(cv)}"

        # write with pandas; fall back to csv module if pandas still errors
        try:
            df = pd.DataFrame([row], columns=cols)
            with open(self.results_path, "a") as f:
                df.to_csv(f, header=f.tell() == 0, index=False)
        except Exception as e:
            # robust fallback: write CSV manually
            import csv, os
            write_header = not os.path.exists(self.results_path) or os.stat(self.results_path).st_size == 0
            with open(self.results_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                if write_header:
                    w.writeheader()
                w.writerow(row)

        # use sanitized values for progress printing
        self.print_progress(
            int(safe.get("updates", 0)),
            int(safe.get("environment_steps", 0)),
            float(safe.get("mean_episode_returns", 0.0)),
            len(metrics) - 1,
        )

    def get_state(self):
        import os
        import pandas as pd

        p = self.results_path
        if not os.path.exists(p) or os.stat(p).st_size == 0:
            return pd.DataFrame()  # nothing to read yet

        # Try the normal read once; if it fails, just return empty to avoid crashing training
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()



