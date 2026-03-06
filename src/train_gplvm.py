import os
import argparse
import numpy as np
import pandas as pd
import torch
import gpytorch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared_gplvm import SharedVariationalGPLVM, create_multimodal_likelihoods
from sequence_frame_loader import SequenceSource, load_selected_rgb_frame, source_frame_path
# We import private functions from features_dinov3 to reuse preprocessing logic
# Assuming src is in path
from features_dinov3 import _preprocess, _extract_features

class MultimodalDataset(Dataset):
    """
    On-the-fly dataset loading for Shared GPLVM training.
    Loads the low-dimensional X_tab and Y_target directly from the manifest CSV.
    Computes high-dimensional DINOv3 features on-the-fly from raw images/videos
    to avoid saving massive embedding files.
    """
    def __init__(self, manifest_path, encoder="dinov3", device="cuda"):
        self.manifest = pd.read_csv(manifest_path)
        
        # Define columns to extract (matching compile_dataset.py)
        static_cols = ['HEIGHT', 'WEIGHT', 'BMI', 'AGE', 'POS', 'YEARS']
        game_cols = ['AVG_MIN', 'AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_PLUS_MINUS', 'TOT_MIN', 'TOT_PTS', 'TOT_REB', 'TOT_AST', 'TOT_PLUS_MINUS', 'GP']
        clinical_cols = ['V_SCORE', 'PT_TEND', 'TT_TEND', 'HE', 'AT_THICK', 'SYMP', 'HEPRE', 'HEMID', 'HEPOST']
        
        # Extract X_tab (Static + Game) and Y_target (Clinical)
        xtab_cols = [c for c in static_cols + game_cols if c in self.manifest.columns]
        ytarget_cols = [c for c in clinical_cols if c in self.manifest.columns]
        
        self.X_tab = self.manifest[xtab_cols].values.astype(np.float32)
        self.Y_target = self.manifest[ytarget_cols].values.astype(np.float32)
        
        self.encoder = encoder
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load Frozen Encoder
        print(f"Loading {encoder} model to {self.device}...")
        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dinov3"))
        # Assuming weights are in standard location or env var
        weights_path = os.environ.get("DINOV3_WEIGHTS", "/workspace/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
        
        self.model = torch.hub.load(
            repo_dir, 
            "dinov3_vits16", 
            source="local", 
            weights=weights_path
        )
        self.model.eval().to(self.device)
        
    def __len__(self):
        return len(self.manifest)
        
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        frame_id = row['frame_id']
        scan_path = row['scan_path']
        
        # 1. Load Raw Frame
        # Determine source type based on extension or directory structure
        # SequenceSource expects a Path object
        # We need to reconstruct the SequenceSource object to use load_selected_rgb_frame
        # This is a bit hacky, ideally we'd persist source info, but we can infer it.
        
        # Parse frame index from frame_id (e.g., "vid_f0045" -> 45)
        try:
            frame_idx_int = int(frame_id.split('_f')[-1])
        except (ValueError, IndexError):
            frame_idx_int = 0
            
        # Construct a temporary source object
        # We assume scan_path points to the directory containing the media
        source = SequenceSource(Path(scan_path), "unknown", 0, 0.0) # Type/count don't matter for loading specific frame if path is dir
        
        # Load and Preprocess
        frame_rgb = load_selected_rgb_frame(source, frame_idx_int)
        image_t = _preprocess(frame_rgb, 16, 0, 0, 0, 0, False) # No offsets, no imagenet norm (DINOv3 handles it?) 
        # Note: features_dinov3.py _preprocess handles normalization if imagenet=True. 
        # Default in features_dinov3.py is False.
        
        # Extract Features
        # _extract_features returns [C, H, W]. We likely need to flatten it for GPLVM 
        # or pool it. The user said "Each embedding is ~300MB", implying full map.
        # However, standard GPLVM expects a vector. 
        # For now, we return the flattened vector.
        feat_map = _extract_features(self.model, image_t, self.device)
        img_feat = feat_map.flatten().float()
        
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
        args.manifest, 
        encoder=args.encoder,
        device=args.device
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
    mll_img = gpytorch.mlls.VariationalELBO(likelihood_img, model.gp_img, num_data=N)
    mll_tab = gpytorch.mlls.VariationalELBO(likelihood_tab, model.gp_tab, num_data=N)
    
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
            
            # Add regularization for the latent space X (MAP estimation)
            # This constrains X to stay close to the prior N(0, I)
            loss_prior = model.latent_prior_loss()
            
            loss = loss_img + loss_tab + loss_prior
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
