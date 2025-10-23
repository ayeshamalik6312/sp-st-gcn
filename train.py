import argparse
import yaml
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import os
import matplotlib.pyplot as plt

from preprocess import preprocess_SPGVAE
from utils import *
from adjacency import *
from models import SpatialVAE
from losses import *
from metrics import evaluate_all


def main():
    parser = argparse.ArgumentParser(description="Run SPGVAE using a YAML config.")
    parser.add_argument("--config", "-c", default="config.yaml", help="Path to YAML config file.")
    args = parser.parse_args()
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Required sections
    paths = cfg.get("paths", {})
    find_marker_genes_paras = cfg.get("find_marker_genes_paras", {})
    pseudo_spot_simulation_paras = cfg.get("pseudo_spot_simulation_paras", {})
    data_normalization_paras = cfg.get("data_normalization_paras", {})
    integration_for_feature_paras = cfg.get("integration_for_feature_paras", {})
    options = cfg.get("options", {})

    outpath = paths.get("output_path", "output")
    os.makedirs(outpath, exist_ok=True)

    # ---------------- STEP 1: Preprocess ----------------
    results = preprocess_SPGVAE(
        paths,
        use_marker_genes = options.get("use_marker_genes", True),
        external_genes = options.get("external_genes", False),
        find_marker_genes_paras = find_marker_genes_paras,
        generate_new_pseudo_spots = options.get("generate_new_pseudo_spots", True),
        pseudo_spot_simulation_paras = pseudo_spot_simulation_paras,
        data_normalization_paras = data_normalization_paras,
        integration_for_feature_paras = integration_for_feature_paras,
        n_jobs = options.get("n_jobs", -1),
        GCN_device = options.get("GCN_device", "GPU"),
    )

    Y_real_df = results.get("real_spots")
    Y_real_loc_df = results.get("real_coordinates")
    Y_pseudo_df = results.get("pseudo_spots")
    Y_all_df = results.get("all_spots")
    cell_types = results.get("cell_types")
    B_pseudo_gt_df = results.get("pseudo_dist")
    X_df = results.get("sc_ref")
    X = X_df.groupby("cell_type").mean()

    # bring everything to torch tensors on the right device
    Y_real = torch.tensor(Y_real_df.values, dtype=torch.float32, device=device)
    Y_real_loc = torch.tensor(Y_real_loc_df.values, dtype=torch.float32, device=device)
    Y_pseudo = torch.tensor(Y_pseudo_df.values, dtype=torch.float32, device=device)
    Y_all = torch.tensor(Y_all_df.values, dtype=torch.float32, device=device)
    B_pseudo_gt = torch.tensor(B_pseudo_gt_df.values, dtype=torch.float32, device=device)
    X = torch.tensor(X.values, dtype=torch.float32, device=device)

    # ---------------- STEP 2: Build Graph ----------------
    num_real_spots = Y_real.shape[0]
    num_pseudo_spots = Y_pseudo.shape[0]
    num_total_spots = num_real_spots + num_pseudo_spots

    # Config knobs (add to your config as needed)
    gcfg = cfg.get("graph", {})
    defaults = {
        "alpha_spatial": 1.0,
        "alpha_expr_real": 1.0,
        "alpha_expr_pseudo": 1.0,
        "alpha_cross": 1.0,
        "k_real": 10,
        "k_pseudo": 10,
        "k_cross_real_to_pseudo": 10,
        "k_cross_pseudo_to_real": 10,
        "expr_thresh": 0.0,
        "mutual_knn": True,
        "use_spatial_knn": True,
        "adjacency_sigma": 2.0,
        "adjacency_threshold": 1e-4,
    }
    for k, v in defaults.items():
        gcfg.setdefault(k, v)



    A_combined, edge_index_combined, edge_weight_combined = build_graphs_and_edges(
        Y_real=Y_real,
        Y_pseudo=Y_pseudo,
        Y_real_loc=Y_real_loc,
        device=device,
        gcfg=gcfg,
    )


    # ---------------- STEP 3: Define splits ----------------
    real_indices = np.arange(num_real_spots)
    pseudo_indices = np.arange(num_real_spots, num_total_spots)

    if cfg.get("split_seed") is not None:
        split_seed = int(cfg["split_seed"])
    else:
        split_seed = int(time.time()) % 10000

    np.random.seed(split_seed)
    random.seed(split_seed)
    np.random.shuffle(real_indices)

    print(f"\nData Split Seed: {split_seed}\n")
    val_size = int(num_real_spots * cfg["val_size"])
    val_idx = real_indices[:val_size]

    train_mask = torch.ones(num_total_spots, dtype=torch.bool, device=device)           # train_real + pseudo
    train_mask[val_idx] = False

    val_mask = torch.zeros(num_total_spots, dtype=torch.bool, device=device)            # only real
    val_mask[val_idx] = True

    pseudo_mask = torch.zeros(num_total_spots, dtype=torch.bool, device=device)         # only pseudo
    pseudo_mask[pseudo_indices] = True

    train_real_mask = train_mask & ~pseudo_mask                                         # only train real 
    real_mask = torch.zeros(num_total_spots, dtype=torch.bool, device=device)
    real_mask[:num_real_spots] = True

    masks = {
        "train": train_mask,                # train set with both real + pseudo spots
        "train_real": train_real_mask,      # train set with only real spots
        "val": val_mask,                    # validation set with only real spots
        "real": real_mask,                  # all real spots (train_real + val)
        "pseudo": pseudo_mask,              # all pseudo spots  
    }

    print(f"Number of total training spots: {train_mask.sum().item()}")
    print(f"Number of real training spots: {train_real_mask.sum().item()}")
    print(f"Number of pseudo training spots: {pseudo_mask.sum().item()}")
    print(f"Number of validation spots: {val_mask.sum().item()}")


    # ---------------- STEP 4: Define model ----------------
    input_dim = Y_all.shape[1]
    n_celltypes = len(cell_types)
    model = SpatialVAE(input_dim, cfg["hidden_dim"], cfg["latent_dim"],
                    n_celltypes, tau=cfg["tau"], drop=cfg["dropout_rate"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["l2_lambda"]))

    # ---------------- STEP 5: Training ----------------
    total_loss_history = []
    recon_loss_history = []
    kl_lat_loss_history = []
    kl_mix_loss_history = []
    smooth_loss_history = []
    entropy_loss_history = []
    align_loss_history = []
    l1_loss_history = []
    val_loss_history = []
    best_val_loss, patience_counter = float("inf"), 0
    patience = int(cfg["patience"])

    for epoch in range(int(cfg["n_epochs"])):
        model.train()
        optimizer.zero_grad()

        # Forward
        Y_hat_all, mu_all, logvar_all, B_pred_all, *rest = model(
            Y_all, edge_index_combined, edge_weight_combined, X
        )

        losses = vae_loss(
            Y_all=Y_all,
            Y_hat_all=Y_hat_all,
            mu=mu_all,
            logvar=logvar_all,
            A_all=A_combined,
            B_all=B_pred_all,
            B_pseudo_gt=B_pseudo_gt,
            masks=masks,
            lambda_recon=float(cfg["loss_recon"]),
            lambda_kl_lat=float(cfg["loss_kl_lat"]),
            lambda_kl_mix=float(cfg['loss_kl_mix']),
            lambda_smooth=float(cfg["loss_smooth"]),
            lambda_ent=float(cfg["loss_ent"]),
            lambda_align=float(cfg['loss_align']),
            lambda_l1=float(cfg['loss_l1']),
            evaluate=False
        )

        loss = losses["total"]
        loss.backward()
        optimizer.step()
        total_loss_history.append(loss.item())
        recon_loss_history.append(cfg["loss_recon"] * losses["recon"].item())
        kl_lat_loss_history.append(cfg["loss_kl_lat"] * losses["kl_lat"].item())
        kl_mix_loss_history.append(cfg['loss_kl_mix'] * losses.get("kl_mix", 0.0).item())
        smooth_loss_history.append(cfg["loss_smooth"] * losses["smooth"].item())
        entropy_loss_history.append(cfg["loss_ent"] * losses["entropy"].item())
        align_loss_history.append(cfg['loss_align'] * losses.get("align", 0.0).item())
        l1_loss_history.append(cfg['loss_l1'] * losses.get("l1", 0.0).item())

        # ---------------- Validation ----------------
        model.eval()
        with torch.no_grad():
            Y_hat_all_eval, mu_all_eval, logvar_all_eval, B_pred_all_eval, *rest = model(
                Y_all, edge_index_combined, edge_weight_combined, X
            )

            val_losses = vae_loss(
                Y_all=Y_all,
                Y_hat_all=Y_hat_all_eval,
                mu=mu_all_eval,
                logvar=logvar_all_eval,
                A_all=A_combined,
                B_all=B_pred_all_eval,
                B_pseudo_gt=B_pseudo_gt,
                masks=masks,
                lambda_recon=float(cfg["loss_recon"]),
                lambda_kl_lat=float(cfg["loss_kl_lat"]),
                lambda_kl_mix=float(cfg['loss_kl_mix']),
                lambda_smooth=float(cfg["loss_smooth"]),
                lambda_ent=float(cfg["loss_ent"]),
                lambda_align=float(cfg['loss_align']),
                lambda_l1=float(cfg['loss_l1']),
                evaluate=True
            )

            val_loss = val_losses["total"]
            val_loss_history.append(val_loss.item())

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:03d}: "
                f"Train={loss.item():.4f} "
                f"(recon={losses['recon']:.4f}, "
                f"kl_lat={losses['kl_lat']:.4f}, "
                f"kl_mix={losses.get('kl_mix', 0.0):.4f}, "
                f"smooth={losses['smooth']:.4f}, "
                f"ent={losses['entropy']:.4f} "
                f'align={losses.get("align", 0.0):.4f}, '
                f'l1={losses.get("l1", 0.0):.4f}) '

                f"Val={val_loss.item():.4f} "
            )

        # ---------------- Early Stopping ----------------
        if val_loss < best_val_loss:
            best_val_loss, patience_counter = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}!")
                break


    # restore best model
    model.load_state_dict(best_state)

    # save loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(total_loss_history, label="Train (Real + Pseudo)", linewidth=4)
    plt.plot(val_loss_history, label="Val (Real)", linewidth=4)
    plt.plot(recon_loss_history, label="Reconstruction", linestyle='--')
    plt.plot(kl_lat_loss_history, label="KL Latent", linestyle='--')
    plt.plot(kl_mix_loss_history, label="KL Mixture", linestyle='--')
    plt.plot(smooth_loss_history, label="Graph Smoothness", linestyle='--')
    plt.plot(entropy_loss_history, label="Entropy", linestyle='--')
    plt.plot(align_loss_history, label="OT Alignment", linestyle='--')
    plt.plot(l1_loss_history, label="L1 Sparsity", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.title("Semi-Supervised Training Loss")
    plt.savefig(os.path.join(cfg["paths"]["output_path"], "loss_curves.svg"), format="svg", dpi=300)
    plt.close()

    # ---------------- Test Evaluation ----------------
    print("\n### Final Evaluation on the Test Set ###")
    model.eval()
    with torch.no_grad():
        _ , _, _, B_pred_all, *rest = model(Y_all, edge_index_combined, edge_weight_combined, X)

    B_pred_all = B_pred_all[real_mask]
    G_all_df = pd.read_csv(os.path.join(cfg["paths"]["ST_path"], "ST_ground_truth.tsv"), sep="\t", index_col=0)

    # Convert to numpy for evaluation and saving
    y_pred = B_pred_all.cpu().numpy()
    y_true = G_all_df[cell_types].values

    # Get the original spot names for the test set using the test_idx array
    B_df = pd.DataFrame(y_pred, index=G_all_df.index, columns=cell_types)
    B_df.to_csv(os.path.join(cfg["paths"]["output_path"], "predicted_proportions.csv"))

    # Evaluate metrics
    results = evaluate_all(y_true, y_pred)
    print("\nEvaluation Results (Test):")
    for k, v in results.items():
        print(f"{k:12s}: {v:.4f}")

    # append if results already exist
    metrics_file = os.path.join(cfg["paths"]["output_path"], "metrics.csv")
    if os.path.exists(metrics_file):
        existing_results = pd.read_csv(metrics_file)
        results_df = pd.DataFrame([results])
        results_df = pd.concat([existing_results, results_df], ignore_index=True)
        results_df.to_csv(metrics_file, index=False)
    else:
        pd.DataFrame([results]).to_csv(os.path.join(cfg["paths"]["output_path"], "metrics.csv"), index=False)


    # ---------------- Optional Plots ----------------
    colors = [plt.cm.tab10(i % 10) for i in range(len(cell_types))]
    B_pred_real_df = pd.DataFrame(B_pred_all[:num_real_spots].cpu().detach().numpy(), index=Y_real_df.index, columns=cell_types)

    print("Plotting all real spots...")
    gt = G_all_df
    pred = B_pred_real_df
    draw_pie_hex_grid(gt, cell_types, colors, Y_real_loc_df,
                    title="Ground Truth (All Real Spots)",
                    save_path=os.path.join(cfg["paths"]["output_path"], "ground_truth_all.svg"))
    draw_pie_hex_grid(pred, cell_types, colors, Y_real_loc_df,
                    title="SP-VAE-GCN (All Real Spots)",
                    save_path=os.path.join(cfg["paths"]["output_path"], "predicted_all.svg"))


if __name__ == "__main__":
    main()
