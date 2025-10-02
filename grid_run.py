#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import subprocess
import yaml
from copy import deepcopy

BASE_CFG = "config.yaml"   # your base config
TEMP_CFG = "config_tmp.yaml" # temporary config for each run
GRID_CFG = "grid.yaml"     # param grid
TRAIN    = "train.py"      # training entrypoint

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def dump_yaml(obj, path):
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def product_dict(d):
    """dict of lists -> iterator of dicts with all combinations."""
    keys = list(d.keys())
    vals = [d[k] if isinstance(d[k], list) else [d[k]] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def main():
    base = load_yaml(BASE_CFG)
    grid = load_yaml(GRID_CFG)

    # turn plot_sets=[[]] into an empty list (to avoid plotting)
    if "plot_sets" in grid:
        grid["plot_sets"] = [lst if isinstance(lst, list) else [] for lst in grid["plot_sets"]]

    combos = list(product_dict(grid))
    print(f"Total runs: {len(combos)}")

    for i, upd in enumerate(combos, 1):
        cfg = deepcopy(base)
        # apply updates
        for k, v in upd.items():
            cfg[k] = v
        # write config.yaml
        dump_yaml(cfg, TEMP_CFG)

        tag = " ".join(f"{k}={v}" for k, v in upd.items())
        print(f"\n=== [{i}/{len(combos)}] {tag} ===")
        # call train.py (blocking)
        subprocess.run(["python", TRAIN], check=True)

if __name__ == "__main__":
    main()
