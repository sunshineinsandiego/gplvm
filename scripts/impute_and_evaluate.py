import os
import argparse
import numpy as np
import pandas as pd
import torch
import gpytorch

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.shared_gplvm import SharedVariationalGPLVM, create_multimodal_likelihoods
from scripts.train_gplvm import MultimodalDataset

def evaluate_imputation(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    print(f"Loading datasets for evaluation: {args.manifest}")
    dataset = MultimodalDataset(
        args.manifest, args.xtab, args.ytarget, 
        embeddings_dir=args.embeddings_dir, 
        encoder=args.encoder
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
    
    # For a robust test, we take the learned latent probability distributions 
    # and pass them through the Tabular Likelihood.
    
    print(f"\nEvaluating on {N} samples...")
    
    mse_total = 0.0
    with torch.no_grad():
        for i in range(N):
            batch = dataset[i]
            y_true_tab = batch["Y2"].to(device)
            
            # Get the optimized latent variable for this specific sample
            x_latent = model.X[i].unsqueeze(0)
            
            # Predict the tabular modality
            _, pred_tab_dist = model(x_latent)
            
            # Pass through likelihood to get actual observation variance
            observed_pred_tab = likelihood_tab(pred_tab_dist)
            
            y_pred_mean = observed_pred_tab.mean.squeeze(0)
            
            # Calculate Mean Squared Error for this sample
            mse = torch.nn.functional.mse_loss(y_pred_mean, y_true_tab).item()
            mse_total += mse
            
            if i < 5: # Print a few examples
                print(f"Sample {i}:")
                print(f"  True Tabular: {y_true_tab.cpu().numpy()[:5]}...")
                print(f"  Pred Tabular: {y_pred_mean.cpu().numpy()[:5]}...\n")
                
    avg_mse = mse_total / N
    print(f"\nFinal Evaluation:")
    print(f"Average MSE across all generalized Tabular predictions: {avg_mse:.4f}")
    
    # You could extend this script to iteratively drop elements of Y2, rebuild the latent point
    # strictly from Y1 using variational inference, and fully impute missing data.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Shared Variational GPLVM Imputation")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv")
    parser.add_argument("--xtab", type=str, default="data/X_tab.npy")
    parser.add_argument("--ytarget", type=str, default="data/Y_target.npy")
    parser.add_argument("--embeddings_dir", type=str, default="scan_output")
    parser.add_argument("--encoder", type=str, default="dinov3")
    parser.add_argument("--model_dir", type=str, default="models/checkpoints")
    
    parser.add_argument("--latent_dim", type=int, default=12)
    parser.add_argument("--n_inducing", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    evaluate_imputation(args)
