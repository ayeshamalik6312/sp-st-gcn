#!/usr/bin/env python3
import os, sys, json, time, csv, hashlib, itertools, subprocess, random
from pathlib import Path
import yaml
from copy import deepcopy

BASE_CFG_KEY = "base_config"
OUT_DIR_KEY  = "out_dir"

def load_yaml(p): 
    with open(p, "r") as f:
        return yaml.safe_load(f)

def dump_yaml(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def product_dict(d):
    """Convert dict of lists into iterator of dicts with all combinations."""
    keys = list(d.keys())
    val_lists = []
    for k in keys:
        v = d[k]
        if isinstance(v, (list, tuple)):
            val_lists.append(list(v))
        else:
            val_lists.append([v])
    for combo in itertools.product(*val_lists):
        yield dict(zip(keys, combo))

def write_row_dynamic_csv(csv_path, row):
    """
    Append a row to CSV; keep a stable, preferred column order.
    If new columns appear, include them at the end (sorted).
    """
    # Preferred order (tweak as you like)
    PREFERRED = [
        "run_dir", "output_path",
        "best_val_loss", "best_epoch", "n_params",
        "loss_recon", "loss_kl_lat", "loss_kl_mix", "loss_smooth",
        "loss_ent", "loss_align", "loss_l1",
        "MSE", "RMSE", "MAE", "Pearson", "Spearman", "CCC", "Cosine",
        "JSD", "Hellinger", "EMD",
    ]

    # Load existing rows (if any)
    rows = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            rdr = csv.DictReader(f)
            rows = list(rdr)

    # Compute superset of headers
    all_headers = set(row.keys())
    for r in rows:
        all_headers.update(r.keys())

    # Build stable header list:
    #  - take PREFERRED in that order if present
    #  - then append any leftover headers alphabetically
    preferred_present = [h for h in PREFERRED if h in all_headers]
    leftovers = sorted(h for h in all_headers if h not in PREFERRED)
    headers = preferred_present + leftovers

    # Always rewrite to enforce clean header order
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in headers})
        w.writerow({h: row.get(h, "") for h in headers})

def combo_hash(d):
    s = json.dumps(d, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]

def run_once(train_py, cfg_path, log_path):
    with open(log_path, "w") as lf:
        # Use the same interpreter
        proc = subprocess.run([sys.executable, train_py, "-c", cfg_path],
                              stdout=lf, stderr=subprocess.STDOUT)
    return proc.returncode

def read_metrics_file(metrics_csv_path):
    if not os.path.exists(metrics_csv_path):
        return None
    # read the LAST line (latest)
    with open(metrics_csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None

def main():
    grid = load_yaml("grid.yaml")
    base_cfg_path = grid.get(BASE_CFG_KEY, "config.yaml")
    sweep_root = Path(grid.get(OUT_DIR_KEY, "./sweeps/losses")).resolve()
    max_concurrent = int(grid.get("max_concurrent", 1))
    rerun_existing = bool(grid.get("rerun_existing", False))
    disable_plots  = bool(grid.get("disable_plots", True))

    base_cfg = load_yaml(base_cfg_path)
    train_py = "train.py"   # same folder as this script

    # Prepare sweep results aggregator
    sweep_root.mkdir(parents=True, exist_ok=True)
    master_csv = sweep_root / "results.csv"
    
    sweeps = grid.get("sweeps", {})

    # Build all combos
    all_combos = list(product_dict(sweeps))

    # (Optional) pre-skip combos already in master to avoid wasting shuffle slots
    if master_csv.exists() and not rerun_existing:
        with open(master_csv, newline="") as f:
            done_rows = list(csv.DictReader(f))
        done = set()
        if done_rows:
            sweep_keys = list(sweeps.keys())
            for r in done_rows:
                done.add(tuple(str(r.get(k, "")) for k in sweep_keys))
        all_combos = [c for c in all_combos if tuple(str(c[k]) for k in sweeps.keys()) not in done]

    # Randomize if requested
    if grid.get("randomize", False):
        seed = grid.get("shuffle_seed", None)
        rng = random.Random(seed)  # deterministic if seed provided
        rng.shuffle(all_combos)

    for combo in all_combos:


        # Create a stable, human-readable run name
        tag = "_".join(f"{k}={v}" for k,v in combo.items())
        h   = combo_hash(combo)
        run_dir = sweep_root / f"W.{time.strftime('%Y-%m-%d_%H%M%S')}.{h}"
        time.sleep(0.2)

        # Skip if the exact combo already exists in master CSV (unless rerun_existing)
        if master_csv.exists() and not rerun_existing:
            same = False  # <-- FIX: initialize before the loop
            with open(master_csv, newline="") as f:
                for row in csv.DictReader(f):
                    # compare only the swept keys
                    if all(str(combo[k]) == str(row.get(k, "")) for k in sweeps.keys()):
                        same = True
                        print(f"[SKIP] existing combo: {combo}")
                        break
            if same:
                continue


        # Merge base config with combo
        cfg = deepcopy(base_cfg)

        # Ensure a unique output path per run
        cfg["paths"]["output_path"] = str(run_dir)

        # Optionally disable plots for speed/stability during sweeps
        opts = cfg.setdefault("options", {})
        if disable_plots:
            opts["plot_loss_curves"] = False
            opts["plot_spatial_pies"] = False

        # Inject losses
        # Inject losses
        for k, v in combo.items():
            try:
                cfg[k] = float(v)   # all your loss_* are scalars
            except Exception:
                raise TypeError(f"Value for {k} must be numeric, got {v!r}. "
                                "Check grid.yaml -> sweeps.")

        # Write temp config
        tmp_cfg = run_dir / "config.yaml"
        dump_yaml(cfg, tmp_cfg)

        # Logs
        log_path = run_dir / "train.log"
        print(f"[RUN] {run_dir.name} :: {combo}")

        # Execute
        rc = run_once(train_py, str(tmp_cfg), str(log_path))
        if rc != 0:
            print(f"[FAIL] {run_dir.name} (rc={rc}) — see log: {log_path}")
            continue

                # Collect metrics from the run’s own metrics.csv (last row)
        run_metrics = read_metrics_file(run_dir / "metrics.csv")  # dict or None

        # Also read run_summary.json if present
        summary = {}
        summary_path = run_dir / "run_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)

        # Base row: identifiers + losses + summary scalars
        row = {
            "run_dir":       run_dir.name,
            "output_path":   str(run_dir),
            "best_val_loss": summary.get("best_val_loss", ""),
            "best_epoch":    summary.get("best_epoch", ""),
            "n_params":      summary.get("n_params", ""),
            "loss_recon":    combo["loss_recon"],
            "loss_kl_lat":   combo["loss_kl_lat"],
            "loss_kl_mix":   combo["loss_kl_mix"],
            "loss_smooth":   combo["loss_smooth"],
            "loss_ent":      combo["loss_ent"],
            "loss_align":    combo["loss_align"],
            "loss_l1":       combo["loss_l1"],
        }

        # Merge ALL metrics columns as-is (preserve your exact header names)
        if run_metrics:
            row.update(run_metrics)

            # pretty print a compact summary
            # prefer correlation if present, else RMSE/MAE/MSE
            def num(x):
                try: return float(x)
                except: return None
            msg = []
            for k in ("Pearson","Spearman","CCC","RMSE","MAE","MSE","Cosine","JSD","Hellinger","EMD"):
                if k in run_metrics and run_metrics[k] != "":
                    val = run_metrics[k]
                    # format nicely if numeric
                    n = num(val)
                    msg.append(f"{k}={n:.4f}" if n is not None else f"{k}={val}")
            print(f"[RESULT] {run_dir.name} :: " + "  ".join(msg))
            print("\n")

        # Write/append to master CSV with dynamic header mgmt
        write_row_dynamic_csv(str(master_csv), row)


    print("\nSweep complete. Aggregated results at:")
    print(master_csv)

if __name__ == "__main__":
    main()
