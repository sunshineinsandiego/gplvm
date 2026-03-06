# Variational Shared GPLVM 

## Deep Learning Medical Imaging Pipeline

The preprocessing and feature extraction pipeline handles complex multi-modal imaging sources uniformly through the following complete pipeline:

1. **Loading and Channel Correction (`sequence_frame_loader.py`)**
   - DICOM images, native images (`.jpg`/`.png`), and videos (`.mp4`) are loaded frame by frame.
   - If a grayscale DICOM slice is encountered `[1, H, W]`, it is replicated across 3 channels to simulate RGB since deep learning generic vision encoders (DINOv3, MedSAM2) expect standard image inputs.
   - If the source is native RGB or 4-channel RGBA, the alpha channel is dropped, reducing the arrays strictly to `[3, H, W]`.

2. **DINOv3 Feature Extraction (`features_dinov3.py`)**
   - Each individual frame is passed through the `dinov3_vits16` backbone.
   - We extract 4 intermediate layers from the transformer blocks.
   - For each layer, we concatenate both the patch map embeddings *and* the broadcasted global `CLS` token along the channel dimension.
   - The final output generated per frame is a dense, high-dimensional tensor in the shape of `[C, H, W]` (specifically `[3072, H, W]`).

3. **In-Memory DPP Selection for Sequences (`features_dinov3.py`)**
   - Video files and multi-frame/multi-slice DICOM sequence sources require summarization before statistical modeling. Because `[C, H, W]` acts as a 3D feature representation, we flatten it out into a massive 1D vector (`[C * H * W]`) for each frame in the sequence natively in memory.
   - We compute pairwise cosine similarities between all frames belonging to the dynamic video or DICOM sequence using these flattened vectors.
   - We then apply the **Determinantal Point Process (DPP)** algorithm (`dpp_map_greedy_logdet`), which inherently balances *quality* (magnitude) and *diversity* (low similarity to other selected frames) to identify a requested number of unique and representative keyframes (e.g., 10 frames).

4. **Dataset Compilation (`compile_dataset.py`)**
   - We bundle all the frames logically required: **Every static single-image scan** + **Only the selected DPP keyframes from dynamic video scans**.
   - The compiler cross-references `selected_keyframes_dpp.json` to find these IDs, maps them directly to the associated tabular patient data (`PPTTUS.csv` and game stats), and emits them row-by-row into a single unified `manifest.csv`.
   - The imaging features themselves stay saved on disk as `.npy` files to prevent RAM overflow, while `compile_dataset.py` builds the structured `X_tab.npy` and `Y_target.npy`. The GPLVM dataloader can seamlessly fetch the correct `.npy` frames during training using the sequence index provided by the manifest.
