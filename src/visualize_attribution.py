"""
src/visualize_attribution.py
Generates spatial heatmaps showing which parts of the ultrasound image 
change when traversing specific latent dimensions (e.g., the 'Injury' dimension).
"""
import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add src to path to import sibling modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from shared_gplvm import SharedVariationalGPLVM
from train_gplvm import MultimodalDataset

def visualize_attribution(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Load Stats & Data
    stats_path = os.path.join(args.model_dir, "y_stats.json")
    if not os.path.exists(stats_path):
        raise FileNotFoundError("Stats file not found. Run train_gplvm.py first.")
        
    with open(stats_path, 'r') as f:
        stats = json.load(f)
        # Convert lists to numpy for the dataset class
        for k in ['xtab_mean', 'xtab_std', 'ytarget_mean', 'ytarget_std']:
            stats[k] = np.array(stats[k], dtype=np.float32)

    print(f"Loading dataset: {args.manifest}")
    dataset = MultimodalDataset(args.manifest, encoder=args.encoder, device=args.device, stats=stats)
    
    # 2. Load Model
    sample = dataset[0]
    Y1_dim = sample["Y1"].shape[0]
    Y2_dim = sample["Y2"].shape[0]
    
    model = SharedVariationalGPLVM(
        n_data=len(dataset), 
        latent_dim=args.latent_dim, 
        n_inducing=args.n_inducing, 
        Y1_dim=Y1_dim, 
        Y2_dim=Y2_dim
    ).to(device)
    
    model.load_state_dict(torch.load(os.path.join(args.model_dir, "shared_gplvm_model.pth"), map_location=device))
    model.eval()
    
    # 3. Identify "Clinical" Dimensions
    # We correlate the learned X with the clinical targets to find which dim tracks what.
    print("Correlating Latent Space with Clinical Variables...")
    X_all = model.X.detach().cpu().numpy()
    Y_target_all = dataset.Y_target # Normalized
    
    # Un-normalize targets for readability
    Y_target_raw = Y_target_all * stats['ytarget_std'] + stats['ytarget_mean']
    target_names = stats['ytarget_cols']
    
    correlations = np.zeros((args.latent_dim, len(target_names)))
    for i in range(args.latent_dim):
        for j in range(len(target_names)):
            # Handle NaNs in target
            valid = ~np.isnan(Y_target_raw[:, j])
            if valid.sum() > 5:
                correlations[i, j] = np.corrcoef(X_all[valid, i], Y_target_raw[valid, j])[0, 1]
    
    # Find top dimension for specific targets
    targets_of_interest = ['V_SCORE', 'AT_THICK']
    dims_to_viz = {}
    
    for target in targets_of_interest:
        if target in target_names:
            idx = target_names.index(target)
            # Find dim with max absolute correlation
            best_dim = np.argmax(np.abs(correlations[:, idx]))
            corr_val = correlations[best_dim, idx]
            dims_to_viz[target] = (best_dim, corr_val)
            print(f"  {target} tracks Latent Dim {best_dim} (Corr: {corr_val:.2f})")
            
    # 4. Generate Heatmaps
    # We use Finite Differences: How does the Image change if we move along this latent dim?
    # Delta_Image = GP_mean(X + epsilon) - GP_mean(X - epsilon)
    
    # We need to know the spatial shape. DINOv3 ViT-S: 384 channels.
    n_channels = args.channels
    n_patches = Y1_dim // n_channels
    grid_size = int(np.sqrt(n_patches))
    print(f"Inferred Image Grid: {grid_size}x{grid_size} patches (from {Y1_dim} dims, {n_channels} ch)")
    
    # Create a "Probe" latent point (e.g., the mean of all patients)
    X_probe = torch.mean(model.X, dim=0).unsqueeze(0) # [1, latent_dim]
    
    for target, (dim_idx, corr_val) in dims_to_viz.items():
        print(f"\nGenerating attribution for {target} (Dim {dim_idx})...")
        
        # Perturb X along the dimension of interest
        epsilon = 2.0 # Move 2 std devs
        X_plus = X_probe.clone(); X_plus[0, dim_idx] += epsilon
        X_minus = X_probe.clone(); X_minus[0, dim_idx] -= epsilon
        
        with torch.no_grad():
            # Get mean image embeddings from the GP
            dist_plus, _ = model(X_plus)
            dist_minus, _ = model(X_minus)
            
            Y_plus = dist_plus.mean # [1, Y1_dim]
            Y_minus = dist_minus.mean
            
            # Difference vector
            delta = (Y_plus - Y_minus).squeeze(0).cpu().numpy() # [Y1_dim]
            
        # Reshape to [C, H, W] -> [384, grid, grid]
        # Note: PyTorch flatten is row-major (C, H, W)
        delta_map = delta.reshape(n_channels, grid_size, grid_size)
        
        # Aggregate magnitude (L2 norm across channels)
        # This tells us "How much did the features at this patch change?"
        heatmap = np.linalg.norm(delta_map, axis=0) # [grid, grid]
        
        # Normalize heatmap 0-1 for plotting
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Plot
        plt.figure(figsize=(6, 5))
        sns.heatmap(heatmap, cmap="magma", xticklabels=False, yticklabels=False)
        plt.title(f"Attribution: {target}\n(Latent Dim {dim_idx}, Corr {corr_val:.2f})")
        
        out_file = os.path.join(args.model_dir, f"attribution_{target}.png")
        plt.savefig(out_file)
        plt.close()
        print(f"Saved heatmap to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--model_dir", default="models/checkpoints")
    parser.add_argument("--encoder", default="dinov3")
    parser.add_argument("--latent_dim", type=int, default=12)
    parser.add_argument("--n_inducing", type=int, default=50)
    parser.add_argument("--channels", type=int, default=384, help="Embedding channels (384 for ViT-S, 768 for ViT-B)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    visualize_attribution(args)