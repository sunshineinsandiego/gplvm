import os
import argparse
import numpy as np
import pandas as pd
import torch
import gpytorch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.shared_gplvm import SharedVariationalGPLVM, create_multimodal_likelihoods

class MultimodalDataset(Dataset):
    """
    On-the-fly dataset loading for Shared GPLVM training.
    Loads the low-dimensional X_tab and Y_target directly into memory.
    Lazy-loads the high-dimensional DINOv3 features using the manifest.csv 
    to avoid OOM errors.
    """
    def __init__(self, manifest_path, xtab_path, ytarget_path, embeddings_dir="scan_output", encoder="dinov3"):
        self.manifest = pd.read_csv(manifest_path)
        self.X_tab = np.load(xtab_path)       # Static Predictors
        self.Y_target = np.load(ytarget_path) # Clinical Outcomes (the things with missingness)
        self.embeddings_dir = embeddings_dir
        self.encoder = encoder
        
    def __len__(self):
        return len(self.manifest)
        
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        subject_id = f"T{row['team']}-{row['player']}.{row['timepoint']}"
        knee_id = row['knee']
        frame_id = row['frame_id']
        
        # We need to find the subset directory from the manifest or dynamically.
        # Assuming frame_id contains the _fXXX suffix from output_stem.
        encoder_dir = os.path.join(self.embeddings_dir, subject_id, knee_id, self.encoder)
        
        # Find the correct npy file. The subset logic might vary so we search.
        feat_path = None
        for root, _, files in os.walk(encoder_dir):
            for file in files:
                if frame_id in file and file.endswith(".npy") and "pre_encoder" not in file:
                    feat_path = os.path.join(root, file)
                    break
            if feat_path: break
            
        if not feat_path:
            raise FileNotFoundError(f"Feature vector for {frame_id} not found in {encoder_dir}")
            
        # 1. Image Embeddings (Y1)
        # Assuming shape is (D1,) where D1 is e.g. 768 or 1024
        img_feat = torch.from_numpy(np.load(feat_path)).float()
        
        # 2. Tabular Features (Y2)
        # Concatenate X_tab and Y_target for the Shared observation space?
        # Actually, standard GPLVM usually treats *all* observed data as Y.
        # Let's concatenate predictors and targets into a single Y2 vector.
        tab_feat = torch.cat([
            torch.from_numpy(self.X_tab[idx]).float(),
            torch.from_numpy(self.Y_target[idx]).float()
        ], dim=-1)
        
        return {
            "index": idx,
            "Y1": img_feat,
            "Y2": tab_feat
        }

def train_shared_gplvm(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 1. Load Dataset
    print(f"Loading datasets from index: {args.manifest}")
    dataset = MultimodalDataset(
        args.manifest, args.xtab, args.ytarget, 
        embeddings_dir=args.embeddings_dir, 
        encoder=args.encoder
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    N = len(dataset)
    
    # Determine dimensions
    sample_batch = next(iter(dataloader))
    Y1_dim = sample_batch["Y1"].shape[1]
    Y2_dim = sample_batch["Y2"].shape[1]
    print(f"Detected N={N} frames. Y1 (Image) Dim={Y1_dim}. Y2 (Tabular) Dim={Y2_dim}.")
    
    # 2. Initialize Model
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
    
    # 3. Setup Optimizers
    # We optimize the Variational Parameters (inducing points), the Latent X prior parameters,
    # the Kernels, and the Likelihood noises.
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood_img.parameters()},
        {'params': likelihood_tab.parameters()}
    ], lr=args.lr)

    # 4. Objective Function (Variational ELBO)
    # We sum the Evidence Lower Bounds of both modalities.
    # Note: VariationalELBO needs to know the total N to scale the KL divergence term properly when batching.
    mll_img = gpytorch.mlls.VariationalELBO(likelihood_img, model, num_data=N)
    mll_tab = gpytorch.mlls.VariationalELBO(likelihood_tab, model, num_data=N)
    
    # 5. Training Loop
    model.train()
    likelihood_img.train()
    likelihood_tab.train()
    
    print("\nStarting ELBO Optimization...")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            indices = batch["index"].to(device)
            Y1 = batch["Y1"].to(device)
            Y2 = batch["Y2"].to(device)
            
            optimizer.zero_grad()
            
            # Retrieve the batch's specific latent coordinates X_batch from the model
            X_batch = model.X[indices]
            
            # Forward pass: mapped representations
            pred_img, pred_tab = model(X_batch)
            
            # ELBO computes: -E[log_prob(Y|X)] + KL(q(X) || p(X))
            # We negate it because PyTorch optimizers *minimize* loss
            loss_img = -mll_img(pred_img, Y1)
            loss_tab = -mll_tab(pred_tab, Y2)
            
            loss = loss_img + loss_tab
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % args.log_interval == 0:
            print(f"Epoch {epoch+1:03d}/{args.epochs:03d} - Total ELBO Loss: {epoch_loss/len(dataloader):.4f}")
            
    # 6. Save Model State
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "shared_gplvm_model.pth"))
    torch.save(likelihood_img.state_dict(), os.path.join(args.out_dir, "likelihood_img.pth"))
    torch.save(likelihood_tab.state_dict(), os.path.join(args.out_dir, "likelihood_tab.pth"))
    
    # Also save the optimized latent space X explicitly so we can plot it via UMAP/PCA later
    np.save(os.path.join(args.out_dir, "optimized_latent_X.npy"), model.X.detach().cpu().numpy())
    
    print(f"\nTraining Complete. Models and Latent Space saved to {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Shared Variational GPLVM")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv")
    parser.add_argument("--xtab", type=str, default="data/X_tab.npy")
    parser.add_argument("--ytarget", type=str, default="data/Y_target.npy")
    parser.add_argument("--embeddings_dir", type=str, default="scan_output")
    parser.add_argument("--encoder", type=str, default="dinov3")
    parser.add_argument("--out_dir", type=str, default="models/checkpoints")
    
    parser.add_argument("--latent_dim", type=int, default=12, help="Dimensions of shared latent space X")
    parser.add_argument("--n_inducing", type=int, default=50, help="Number of inducing points for variational approx")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_interval", type=int, default=50)
    
    args = parser.parse_args()
    train_shared_gplvm(args)
