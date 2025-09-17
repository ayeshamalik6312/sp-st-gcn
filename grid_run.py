#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import subprocess
import yaml
from copy import deepcopy
import random
import argparse

BASE_CFG = "config.yaml"      # your base config
TEMP_CFG = "config_tmp.yaml"  # temporary config for each run
GRID_CFG  = "grid.yaml"       # param grid
TRAIN     = "train.py"        # training entrypoint

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for shuffling (default: None)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only this many random combos (default: all)")
    args = parser.parse_args()

    base = load_yaml(BASE_CFG)
    grid = load_yaml(GRID_CFG)

    # Optional: if grid.yaml contains plot_sets entries that shouldn't plot by default
    if "plot_sets" in grid:
        grid["plot_sets"] = [lst if isinstance(lst, list) else [] for lst in grid["plot_sets"]]

    combos = list(product_dict(grid))

    # --- Randomize order (and optionally sample a subset) ---
    rng = random.Random(args.seed) if args.seed is not None else random
    rng.shuffle(combos)
    if args.limit is not None:
        n = min(args.limit, len(combos))
        combos = combos[:n]

    print(f"Total runs: {len(combos)}")

    for i, upd in enumerate(combos, 1):
        cfg = deepcopy(base)
        # apply updates
        for k, v in upd.items():
            cfg[k] = v
        # write temp config
        dump_yaml(cfg, TEMP_CFG)

        tag = " ".join(f"{k}={v}" for k, v in upd.items())
        print(f"\n=== [{i}/{len(combos)}] {tag} ===")

        # call train.py (blocking)
        subprocess.run(["python", TRAIN, "--config", TEMP_CFG], check=True)

if __name__ == "__main__":
    main()
