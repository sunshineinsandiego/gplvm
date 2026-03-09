import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import gpytorch

import sys
# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from shared_gplvm import SharedVariationalGPLVM, create_multimodal_likelihoods
from train_gplvm import MultimodalDataset
from gpytorch.distributions import MultitaskMultivariateNormal

def infer_latent_map(model, likelihood_img, y1_data, n_steps=100, lr=0.05):
    """
    Infers the latent variable x* given ONLY the image data Y1.
    Optimizes x* to maximize P(Y1 | x*) * P(x*).
    """
    device = y1_data.device
    # Initialize x at the prior mean (0)
    x_param = torch.nn.Parameter(torch.zeros(1, model.latent_dim, device=device))
    optimizer = torch.optim.Adam([x_param], lr=lr)
    
    model.eval()
    likelihood_img.eval()
    
    # We need to temporarily enable grad for x_param even in eval mode
    with torch.enable_grad():
        for _ in range(n_steps):
            optimizer.zero_grad()
            
            # 1. Prior term: -log P(x) ~ 0.5 * ||x||^2 (Standard Normal Prior)
            loss_prior = 0.5 * torch.sum(x_param ** 2)
            
            # 2. Likelihood term: -log P(Y1 | x)
            # Get the predictive distribution for the image modality
            mvn_img = model.gp_img(x_param)
            mt_img = MultitaskMultivariateNormal.from_repeated_mvn(mvn_img, num_tasks=model.Y1_dim)
            pred_dist = likelihood_img(mt_img)
            
            # Calculate negative log probability of the observed image
            loss_lik = -pred_dist.log_prob(y1_data)
            
            loss = loss_lik + loss_prior
            loss.backward()
            optimizer.step()
            
    return x_param.detach()

def get_bucket(val, col_name):
    """Maps continuous values to buckets based on clinical definitions."""
    if "V_SCORE" in col_name:
        if val < 70: return 0 # <70
        if val < 80: return 1 # 70-80
        if val < 90: return 2 # 80-90
        return 3              # 90+
    if "AT_THICK" in col_name:
        if val < 4.5: return 0 # Healthy
        if val < 5.5: return 1 # Thickened
        return 2               # Pathological
    if "AVG_MIN" in col_name:
        if val <= 20: return 0 # Low
        return 1               # High
    return -1 # No bucket defined

def evaluate_imputation(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Load Stats and Data
    stats_path = os.path.join(args.model_dir, "y_stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"Stats file not found at {stats_path}. Train model first.")
        
    with open(stats_path, 'r') as f:
        stats = json.load(f)
        # Convert lists back to numpy
        stats['xtab_mean'] = np.array(stats['xtab_mean'], dtype=np.float32)
        stats['xtab_std'] = np.array(stats['xtab_std'], dtype=np.float32)
        stats['ytarget_mean'] = np.array(stats['ytarget_mean'], dtype=np.float32)
        stats['ytarget_std'] = np.array(stats['ytarget_std'], dtype=np.float32)
    
    print(f"Loading datasets for evaluation: {args.manifest}")
    dataset = MultimodalDataset(
        args.manifest, 
        encoder=args.encoder,
        device=args.device,
        stats=stats # Pass stats to normalize input data same as training
    )
    
    # Get dimensions
    sample_batch = dataset[0]
    Y1_dim = sample_batch["Y1"].shape[0]
    Y2_dim = sample_batch["Y2"].shape[0]
    N = len(dataset)
    
    # 2. Reconstruct Model Structure
    model = SharedVariationalGPLVM(
        n_data=N, 
        latent_dim=args.latent_dim, 
        n_inducing=args.n_inducing,
        Y1_dim=Y1_dim,
        Y2_dim=Y2_dim
    ).to(device)
    
    likelihood_img, likelihood_tab = create_multimodal_likelihoods(Y1_dim, Y2_dim)
    likelihood_img = likelihood_img.to(device)
    likelihood_tab = likelihood_tab.to(device)
    
    # 3. Load Trained Weights
    model_path = os.path.join(args.model_dir, "shared_gplvm_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find trained model at {model_path}. Run train_gplvm.py first.")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    likelihood_img.load_state_dict(torch.load(os.path.join(args.model_dir, "likelihood_img.pth"), map_location=device))
    likelihood_tab.load_state_dict(torch.load(os.path.join(args.model_dir, "likelihood_tab.pth"), map_location=device))
    
    model.eval()
    likelihood_img.eval()
    likelihood_tab.eval()
    
    # 4. Evaluation Logistics
    # We want to see if the latent space X, given the Image embeddings Y1, 
    # can accurately predict the Tabular outcomes Y2.
    
    print(f"\nEvaluating on {N} samples using mode: {args.inference_mode}")
    if args.inference_mode == "image_only":
        print("  -> Ignoring trained latent X. Re-inferring X from Image (Y1) only for each sample.")
    else:
        print("  -> Using trained latent X (Reconstruction).")
    
    mse_total = 0.0
    all_preds = []
    all_truth = []
    all_latents = []
    all_indices = []
    
    with torch.no_grad():
        for i in range(N):
            batch = dataset[i]
            y1_img = batch["Y1"].to(device).unsqueeze(0) # [1, D1]
            y_true_tab = batch["Y2"].to(device)
            
            if args.inference_mode == "image_only":
                # Optimization step to find X given Y1
                x_latent = infer_latent_map(model, likelihood_img, y1_img)
            else:
                # Use the X learned during training (saw both Y1 and Y2)
                x_latent = model.X[i].unsqueeze(0)
            
            all_latents.append(x_latent.cpu().numpy())
            
            # Predict the tabular modality
            _, pred_tab_dist = model(x_latent)
            
            # Pass through likelihood to get actual observation variance
            observed_pred_tab = likelihood_tab(pred_tab_dist)
            
            y_pred_mean = observed_pred_tab.mean.squeeze(0)
            
            # Calculate Mean Squared Error for this sample
            mse = torch.nn.functional.mse_loss(y_pred_mean, y_true_tab).item()
            mse_total += mse
            
            all_indices.append(i)
            all_preds.append(y_pred_mean.cpu().numpy())
            all_truth.append(y_true_tab.cpu().numpy())
            
                
    avg_mse = mse_total / N
    print(f"\nFinal Evaluation:")
    print(f"Average MSE across all generalized Tabular predictions: {avg_mse:.4f}")
    
    # --- Un-normalize and Bucket Evaluation ---
    print("\n--- Clinical Bucket Accuracy ---")
    
    # Reconstruct full Y2 matrix (preds and truth)
    Y_pred_norm = np.vstack(all_preds)
    Y_true_norm = np.vstack(all_truth)
    
    # Split back into X_tab and Y_target components to un-normalize
    # Y2 was constructed as cat(X_tab, Y_target)
    n_xtab = len(stats['xtab_cols'])
    n_ytarget = len(stats['ytarget_cols'])
    
    # Extract just the Y_target part (clinical outcomes)
    Y_pred_target_norm = Y_pred_norm[:, n_xtab:]
    Y_true_target_norm = Y_true_norm[:, n_xtab:]
    
    # Un-normalize
    Y_pred_target = Y_pred_target_norm * stats['ytarget_std'] + stats['ytarget_mean']
    Y_true_target = Y_true_target_norm * stats['ytarget_std'] + stats['ytarget_mean']
    
    # Check buckets for specific columns
    for idx, col_name in enumerate(stats['ytarget_cols']):
        if col_name in ['V_SCORE', 'AT_THICK', 'AVG_MIN']:
            correct = 0
            total = 0
            for i in range(N):
                # Skip if ground truth was NaN (masked)
                # In dataset, NaNs are preserved.
                if np.isnan(Y_true_target[i, idx]):
                    continue
                    
                true_bucket = get_bucket(Y_true_target[i, idx], col_name)
                pred_bucket = get_bucket(Y_pred_target[i, idx], col_name)
                
                if true_bucket != -1:
                    total += 1
                    if true_bucket == pred_bucket:
                        correct += 1
            
            if total > 0:
                print(f"  {col_name} Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

    # --- Attribution / Correlation Analysis ---
    print("\n--- Attribution Analysis: Latent vs Tabular Correlations ---")
    print("Which latent dimensions drive which clinical outputs?")
    
    # Stack arrays: [N, Latent_Dim] and [N, Tabular_Dim]
    X_matrix = np.vstack(all_latents).squeeze()
    Y_pred_matrix = np.vstack(all_preds)
    
    # Use the columns from stats
    cols = stats['xtab_cols'] + stats['ytarget_cols']
    
    # Compute correlation between each Latent Dim and each Tabular Output
    # This tells us: "Latent Dim 3 is highly correlated with VISA Score"
    for dim_i in range(min(5, X_matrix.shape[1])): # Check first 5 latent dims
        print(f"\nLatent Dimension {dim_i}:")
        corrs = []
        for col_j, col_name in enumerate(cols):
            if col_j < Y_pred_matrix.shape[1]:
                # Pearson correlation
                corr = np.corrcoef(X_matrix[:, dim_i], Y_pred_matrix[:, col_j])[0, 1]
                corrs.append((col_name, corr))
        
        # Sort by absolute correlation
        corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        for name, r in corrs[:3]: # Top 3 correlated features
            print(f"  - {name}: {r:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Shared Variational GPLVM Imputation")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv")
    parser.add_argument("--encoder", type=str, default="dinov3")
    parser.add_argument("--model_dir", type=str, default="models/checkpoints")
    parser.add_argument("--inference_mode", type=str, choices=["reconstruction", "image_only"], default="reconstruction", 
                        help="reconstruction: use trained X. image_only: infer X from image Y1 (simulates missing Y2).")
    
    parser.add_argument("--latent_dim", type=int, default=12)
    parser.add_argument("--n_inducing", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    evaluate_imputation(args)
