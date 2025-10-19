#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import subprocess
import yaml
import os
from copy import deepcopy

BASE_CFG = "config.yaml"      # base config file
TEMP_CFG = "config_tmp.yaml"  # temporary config for each run
GRID_CFG = "grid.yaml"        # parameter grid
TRAIN    = "train.py"         # training entrypoint

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def dump_yaml(obj, path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def product_dict(d):
    """Convert dict of lists into iterator of dicts with all combinations."""
    keys = list(d.keys())
    vals = [d[k] if isinstance(d[k], list) else [d[k]] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def main():
    base = load_yaml(BASE_CFG)
    grid = load_yaml(GRID_CFG)

    # Optional: turn plot_sets=[[]] into an empty list (to avoid plotting)
    if "plot_sets" in grid:
        grid["plot_sets"] = [lst if isinstance(lst, list) else [] for lst in grid["plot_sets"]]

    combos = list(product_dict(grid))
    print(f"Total runs: {len(combos)}")

    for i, upd in enumerate(combos, 1):
        cfg = deepcopy(base)
        # apply parameter updates
        for k, v in upd.items():
            cfg[k] = v
        # write temporary config
        dump_yaml(cfg, TEMP_CFG)

        tag = " ".join(f"{k}={v}" for k, v in upd.items())
        print(f"\n=== [{i}/{len(combos)}] {tag} ===")

        try:
            # run training with the temporary config file (use --config flag)
            subprocess.run(["python", TRAIN, f"--config={TEMP_CFG}"], check=True)
        finally:
            # clean up temp config file after each run
            if os.path.exists(TEMP_CFG):
                os.remove(TEMP_CFG)

if __name__ == "__main__":
    main()
