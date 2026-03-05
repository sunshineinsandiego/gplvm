# Multimodal Shared Gaussian Process Latent Variable Model (Shared GPLVM)

This repository implements a **Shared Gaussian Process Latent Variable Model** to fuse high-dimensional medical imaging (ultrasound scans of patellar tendons) with low-dimensional tabular data (clinical health measurements, VISA scores, and athlete game statistics). Our goal is to uncover nonlinear correlations between physical knee structures and athletic outcomes, specifically accounting for partially missing clinical/game data.

---

## 1. Background: What is a Shared GPLVM? (Undergraduate Level)

Imagine you have two different "views" of a patient:
1. **View 1 (The Image):** A highly detailed, complex ultrasound scan of their knee.
2. **View 2 (The Spreadsheet):** A neat row of numbers containing their clinical health (like a tendon thickness measurement) and their basketball performance (points, rebounds).

It is very hard to directly draw a line showing how pixel #402 in an ultrasound affects a player's free-throw percentage. Instead of trying to connect them directly, a **Shared Gaussian Process Latent Variable Model (Shared GPLVM)** takes a different approach.

It assumes that both the complex image and the neat spreadsheet are products of a hidden, underlying "state" specific to that patient—a low-dimensional **Latent Space**. 
Think of this latent space as the "true" overall physical health of the player's knee. We can't measure this "true health" directly. But we know that this hidden state casts two shadows:
- It generates the structural shapes we see in the **ultrasound**.
- It generates the pain scores and performance metrics we record in the **spreadsheet**.

Because we use **Gaussian Processes (GPs)**—a statistical tool that excels at modeling smooth, non-linear relationships with built-in uncertainty—we can mathematically reverse-engineer this hidden state! 

Even better, if a player is missing their spreadsheet data (maybe they didn't fill out the clinical survey), the model uses their ultrasound image to find their location in the hidden "true health" space. Once we know where they are in that hidden space, the model can reliably predict what their survey answers *would have been*, along with a confidence interval (uncertainty) telling us how sure it is about the prediction.

---

## 2. Specific Considerations for this Repository

Our dataset presents a unique set of challenges and opportunities:

*   **Low Patient Count ($N$):** We currently have around 46 to 92 ultrasound images. Deep learning models (like VAEs or multi-modal autoencoders) generally fail with such a small $N$ because they overfit. GPs are uniquely suited for low-$N$ regimes because they gracefully fall back on their priors and maintain robust uncertainty bounds.
*   **High-Dimensional Imaging ($D_1$):** Raw ultrasound videos are impossibly huge. We solve this by passing the scans through frozen Vision Foundation Models (**DINOv3** and **MedSAM2**). This reduces the images to dense feature embeddings (e.g., 768 or 1024 dimensions). While smaller than raw pixels, this is still high-dimensional, making a dimensionality-reduction technique like GPLVM ideal.
*   **Low-Dimensional Tabular Data ($D_2$):** Features extracted from `PPTTUS.csv` and `game_stats_CU.csv` range from categorical (Position) to continuous (Tendon Thickness, VISA score, Points Per Game). These features must be carefully normalized so they contribute appropriately to the joint marginal likelihood alongside the high-dimensional image embeddings.
*   **Missingness:** In our clinical dataset, missing values will occur naturally (e.g., athletes missing surveys or non-players missing game stats). The Shared GPLVM handles this naturally by "anchoring" the patient in the latent space using their available modality.

---

## 3. Implementation Options and Considerations

When building a Shared GPLVM, we have several architectural and framework choices:

### Architecture Options
1.  **Exact Shared GPLVM:** Inverts the full $N \times N$ covariance matrix.
    *   *Pros:* Mathematically exact point estimations, no approximations.
    *   *Cons:* Computes distances between *every single data point*, scaling cubically at $\mathcal{O}(N^3)$.
    *   *Verdict for us:* If we only had 92 still images ($N=92$), an Exact GPLVM would be viable. However, to capture the full structural variance of the tendon, we treat *every video frame* independently. This causes $N$ to skyrocket into tens of thousands of frames. A standard GPLVM is computationally impossible in this regime because inverting a $30,000 \times 30,000$ matrix every training step would exhaust memory and compute limits.
2.  **Variational Shared GPLVM (vGPLVM):** Introduces a small set of "inducing points" (e.g., 50-100 anchor points) to approximate the posterior distribution.
    *   *Pros:* Drops the computational complexity to $\mathcal{O}(NM^2)$, making it easily scalable to tens of thousands of video frames on a GPU. Furthermore, instead of optimizing exact points, it places a **Gaussian probability distribution** (Bayesian prior) over the latent space $X$. This mathematical prior gives us a rigorous, built-in mechanism to handle missing clinical tabular data by falling back on the image likelihood to anchor the probability coordinates.
    *   *Cons:* Slightly more complex to train and tune bounds (ELBO).

### Framework Options
1.  **GPy / paramz:** The historic gold standard for GPLVMs in Python.
    *   *Downside:* Built on NumPy. It cannot seamlessly backpropagate gradients into PyTorch models or leverage GPU acceleration effectively alongside neural networks.
2.  **Pyro / NumPyro:** Highly flexible Probabilistic Programming Languages.
    *   *Downside:* Steep learning curve to encode custom cross-modal likelihoods manually.
3.  **GPyTorch:** A highly optimized GP library built directly on PyTorch.
    *   *Upside:* Excellent GPU support, seamlessly integrates with PyTorch-based frozen encoders (DINOv3/MedSAM2), and has natively implemented `gpytorch.models.gplvm` modules that we can extend for Shared/Multimodal setups.

---

## 4. Optimal Solution

### The Recommendation: **GPyTorch Variational Shared GPLVM**

Although an Exact GP is mathematically affordable for $N=92$, a **Variational GPLVM implemented in GPyTorch** is the most robust and forward-looking solution for this repository. 

**Why?**
1.  **End-to-End PyTorch Ecosystem:** Both DINOv3 and MedSAM2 run in PyTorch. Using GPyTorch keeps the entire pipeline (from image loading to GP inference) in a single, differentiable GPU ecosystem.
2.  **Handling Missingness as a Prior:** Variational inference allows us to place a standard Normal prior over the latent space $X$. For patients with missing tabular data, the variational distribution for their latent point $q(x_i)$ will naturally rely entirely on the image data likelihood and the prior, making the cross-modal imputation statistically sound out-of-the-box.
3.  **Future-Proofing for Video Tracking:** As mentioned in `legend.txt` ("UMAP 3D spatial tracker deformation during video"), we may eventually want to embed *every frame* of an ultrasound sweep rather than a single summarized image. This will explode $N$ from 92 to tens of thousands. A Variational GP with inducing points easily scales to this regime without requiring a complete rewrite of the codebase.

### Next Steps / Implementation Plan
1.  **DINOv3/MedSAM2 Feature Extraction:** Write a PyTorch script (using the Docker environments) to pass the `scan/` files through the encoders and extract $\mathbf{Y}^{(1)}$ (the $N \times D_1$ embedding matrix).
2.  **Dataset Curation:** Preprocess `legend.txt` targets, `PPTTUS.csv`, and `game_stats_CU.csv` to build a clean $\mathbf{Y}^{(2)}$ matrix (the $N \times D_2$ tabular matrix) representing the exact same athletes.
3.  **Model Construction:** Extend GPyTorch's `BayesianGPLVM` to accept two likelihood functions (one for $\mathbf{Y}^{(1)}$, one for $\mathbf{Y}^{(2)}$) tied to a single variational latent variable $\mathbf{X}$.
4.  **Training & Imputation:** Train using a variational lower bound (ELBO) and validate cross-modal imputation by artificially masking out known VISA scores/tendon thickness and predicting them from the images alone.

---

# Attribution Clustering Plan

This plan details how to group the dataset into meaningful clusters. These assignments define what we are comparing when running attribution, enabling us to answer specific clinical and biomechanical questions. We will create a script `src/create_attribution_cohorts.py` that builds these definitions for the feature extractors (MedSAM2/DINOv3).

## Core Definitions

To map a clinical question into an attribution task, we need the following components for each assignment:

- **Target / Cluster A**: The group of interest (e.g., Injured Knees, Post-season scans).
- **Rival / Cluster B**: The baseline or contrasting group (e.g., Healthy Knees, Pre-season scans).
- **Centroids (`mu_pos`, `mu_neg`)**: The mathematical average embedding vector of all images/frames belonging to the Target and Rival clusters, respectively.
- **Positives**: The frames inside the Target cluster. When analyzing a specific frame, these are its nearest neighbors within the same Target cluster. They teach the model *cohesion* (what this condition generally looks like).
- **Hard Negatives**: The frames from the Rival cluster that look most structurally similar to the Target frame. They teach the model *separation* (what differentiates the target from a closely related but functionally different scan).

### Addressing the Low Sample Size Issue
*Note on Small N:* Comparing a specific player's specific knee at a single timepoint will only have ~2 still images and one DICOM video. Clustering usually requires larger distributions to find robust centroids. 
*Solution:* For questions comparing "Player X vs. Player Y", we will compute cluster centroids at the **cohort level** (e.g., all healthy vs. all injured) and compute the mathematical margin for Player X's specific frame against those global or team-level centroids. If we truly want to isolate a single knee's temporal evolution (e.g., Player X Pre vs Player X Post), we will pool individual frames extracted from the *video* scan alongside the still images to artificially inject $N$, ensuring we have enough sample points to compute a stable centroid for that specific joint at that timepoint.

## Proposed Cohort Groupings

Below are the expanded assignment groupings incorporating both the original clinical questions and the new combinatorial requirements.

### 1. Temporal Evolution (How does an image evolve over the season?)
* **Longitudinal Cohort Progression**: Target: POST (.3) L Knees. Rival: PRE (.1) L Knees. (Repeat for R knees).
* **Specific Player Progression**: Target: Player X R Knee POST. Rival: Player X R Knee PRE. Positives: Frames from Player X's POST video. Hard Negatives: Frames from Player X's PRE video. Centroids: Average embeddings of Player X POST vs Player X PRE frames.
* **Decreasing VISA-P Progression**: Target: PRE/MID/POST scans of players with dropping VISA scores.
* **Stable/Increasing VISA-P Progression**: Target: PRE/MID/POST scans of players with stable or increasing VISA scores.

### 2. Symptom Presence & Transition
* **Symptomatic at Time T**: Target: Symptomatic (SYMP=1) vs Rival: Non-symptomatic (SYMP=0). Run this definition isolated at PRE, MID, POST, and pooled across ALL timepoints.
* **Symptom Transition (Pre to Mid)**: 
  * Target: Non-symptomatic at PRE vs Rival: Symptomatic at MID. 
  * Target: Symptomatic at PRE vs Rival: Non-symptomatic at MID.
* **Symptom Transition (Mid to Post)**: 
  * Target: Non-symptomatic at MID vs Rival: Symptomatic at POST. 
  * Target: Symptomatic at MID vs Rival: Non-symptomatic at POST.

### 3. Asymmetry & Anomalies
* **Inter-subject Asymmetry (Side-matched)**: Target: Player Y R knee (Injured). Rival: Player X R knee (Healthy).
* **Intra-subject Asymmetry (Gold Standard)**: Target: Player X L knee (Injured/Target). Rival: Player X R knee (Healthy).
* **Subject vs. Team Anomaly**: Target: Player X R knee. Rival: All Team 1 R knees @ same timepoint.
* **Subject vs. Healthy Baseline Anomaly**: Target: Player X Injured knee. Rival: All Healthy knees across entire study @ same timepoint.

### 4. Structural Pathology & Predictive Interactions
* **Silent Pathology Assignment**: Target: Hypoechoic (HE=1) AND Symptomatic (SYMP=1). Rival: Hypoechoic (HE=1) AND Non-symptomatic (SYMP=0). 
* **The Predictive Assignment**: Target: PRE scans of athletes who got injured during the season (TLPT=1 or TTR=1). Rival: PRE scans of athletes who stayed healthy.
* **Clinical Crash Predictor**: Target: PRE scans where DV13=1 (Future drop in VISA >= 13). Rival: PRE scans where DV13=0 (Stable).
* **Severity Gradient**: Target: Low VISA (<= 70) across all knees/timepoints. Rival: High VISA (>= 90) across all knees/timepoints. 
* **Osgood-Schlatter Fingerprint**: Target: Osgood-Schlatter history (OS=1). Rival: No history (OS=0). 

### 5. Load Management, Adaptation & Resilience
* **Healthy Adaptation**: Target: FULLHEALTH=1 AND Minutes > 25 (High Min). Rival: FULLHEALTH=1 AND Minutes < 10 (Low Min).
* **Injury Resilience (Unhealthy vs Healthy Load)**: Target: Unhealthy (SYMP=1 or injured) AND Minutes > 25. Rival: FULLHEALTH=1 AND Minutes > 25.
* **Positional Specificity (Mechanics)**: Target: Guards (POS=1) vs Rival: Forwards/Centers (POS=2).
* **Positional Progression**: Target: Guard Knees PRE vs Rival: Guard Knees POST. (Repeat identical logic for Forwards).

## Implementation Steps

### `src/create_attribution_cohorts.py`
#### [NEW] `src/create_attribution_cohorts.py`
Create a new script mapping the structured requirements:
1. Merge `PPTTUS.csv` data with `Min` fields from the game stats arrays.
2. Build JSON/CSV dictionary structures defining the target, rival, and match logic for every assignment listed above.
3. Identify relevant subset IDs (`scan/T#-#.#/<L|R>`) to filter video/scan directories.
4. Handle the "N=2" fallback mechanic by instructing the downstream embedding pipeline to extract video frames instead of exclusively relying on static images.

## Verification Plan
- Unit tests mapping edge cases (ensuring a Player with dropping VISA maps correctly into the prediction clusters).
- Ensure generated files output `Target N=...` and `Rival N=...` before triggering deep learning embeddings.

## COPIED IMPLEMENTATION PLAN
Shared GPLVM Implementation Plan
Overview
We will implement a Variational Shared GPLVM using GPyTorch to map high-dimensional ultrasound feature vectors (extracted via DINOv3 or MedSAM2) and low-dimensional tabular data (
PPTTUS.csv
, game_stats) into a shared latent space. This will allow for cross-modal imputation and discovery of non-linear correlations between structural knee health and athletic performance.

Key Decisions from Discovery
Data Mapping:
Scans: T[Team]-[Player].[Timepoint]/[Knee] (e.g., T1-23.3/R).
Tabular: 
PPTTUS.csv
 contains rows mapping explicitly to the {Team}-{Player}{Knee} format (e.g., T1-23R).
Games Stats: Will need to be aggregated/mapped by Player and corresponding timepoints.
Frame Strategy (Keyframes via DPP):
Approach: Raw ultrasound sequences contain huge frame counts. Rather than processing every single frame, we will select N spatially diverse keyframes per sweep using Deterministic Point Process (DPP) via the provided script (
select_keyframes_dpp.py
). In the Shared GPLVM, each of these keyframes will act as an independent data point $N$ tied back to the identical tabular patient record.
Why: This ensures we capture structural variation of the tendon across the whole sweep without bloating $N$ to unreasonable sizes (e.g., hundreds of nearly identical adjacent frames per patient).
Framework: GPyTorch, utilizing the new Docker environment at /docker/gpytorch/.
Proposed Changes
1. File Ingestion & Feature Extraction
To handle file parsing and feature extraction natively using Docker, we will reuse robust scripts copied from the PTT repository:

sequence_frame_loader.py (Strict multi-frame/single-frame DICOM input logic)
features_dinov3.py (Feature extraction matching the DINOv3 pipeline)
features_medsam2.py (Feature extraction matching the MedSAM2 pipeline)
select_keyframes_dpp.py (Select diverse keyframe subsets from extracted frame embeddings using DPP)
[NEW] scripts/compile_dataset.py
This script acts as the master join between unstructured keyframes and structured tabular outcomes. It will accept an argument --encoder {dinov3|medsam2} to dictate which set of embeddings to aggregate.

1. Data Parsing Strategy

Parses the scan/T[Team]-[Player].[Timepoint]/[Knee] paths to extract identifying fields.
Generates a unique key string: T{Team}-{Player}{Knee} (e.g. T1-20R). This key matches the exact format found in the Number column of PPTTUS.csv.
2. Table Merging Strategy

Loads data/PPTTUS.csv.
Filters only for FULLDATA=1 (or logs exceptions if missing).
Extracts static patient information: HEIGHT, WEIGHT, BMI, AGE, POS, YEARS.
Extracts dynamic target block based on the Timepoint found in the scan path. For instance, if the scan is T1-20.1/R (Preseason), it extracts: VPRE (VISA Score), PTPRE, TTPRE, HEPRE, ATPRE (Thickness). If the scan is .2, it extracts the MID equivalents.
Game Stats Integration (Temporal Slicing):
Loads game_stats_CU.csv (Team 1) and game_stats_FD.csv (Team 2).
Matches the Team and Player.
Instead of flat season averages, it dynamically aggregates game stats based on the Timepoint:
Timepoint 1 (PRE): Preseason (Stats = 0, as games haven't started).
Timepoint 2 (MID): End of January. Aggregates only games with a date before 2/1/2022.
Timepoint 3 (POST): Postseason. Aggregates all games for the player.
Computes the average (and possibly sums, to capture total mechanical load) for key metrics: Min, Pts, FG%, Tot Reb, Ast, and +/- over that specific temporal slice.
3. Alignment & Flattening (No Image Caching)

CRITICAL CONSTRAINT: A single DINOv3 generic features array can be ~300MB. Attempting to save thousands of these frames into an X_img.npy array will instantly exhaust disk space and memory.
Therefore, we will NOT save the image embedding tensors during this compilation step.
Instead, compile_dataset.py cross-references the selected_keyframes_dpp.json outputs from the DPP step to figure out exactly which frames we intend to use.
Emits a synchronized dataset:
X_tab.npy: A tensor array of size $[N \times D_{static}]$ containing static predictors (Age, BMI, Height, Game Stats avg).
Y_target.npy: A tensor array $[N \times D_{clinical}]$ containing clinical outcomes to impute/predict (Thickness, VISA score).
manifest.csv: The critical master index. Row $i$ maps to Row $i$ of X_tab and Y_target and contains the absolute path to the raw .dicom or .mp4 and the frame_id to extract.
IMPORTANT

The actual $N \times D_1$ image embeddings are never fully serialized to disk. The GPyTorch Dataloader will use manifest.csv to load the raw image/video source, crop out the specific frame_id, and run it through the frozen VFM encoder strictly on-the-fly during the Forward pass.

Example Merged Row in manifest.csv: If compile_dataset.py processes image scan/T1-20.2/R and selects frame_0045, the resulting row matched to T1-20R at the .2 (Midseason) timepoint might look like:

csv
frame_id,team,player,knee,timepoint,HEIGHT,WEIGHT,BMI,AGE,POS,YEARS,MID_AVG_MIN,MID_AVG_PTS,MID_AVG_REB,V_SCORE,PT_TEND,TT_TEND,HE,AT_THICK
T1-20.2_R_f0045,1,20,R,2,80,220,24.16,19,2,12,18.5,8.3,2.2,93,0,0,0,3.6
(where V_SCORE comes from VPRE, PT_TEND from PTPRE, and so on. X_img.npy[i] will hold the exact DINOv3 embedding for f0045.)

2. GPyTorch Model Implementation
The core machine learning architecture.

[NEW] src/models/shared_gplvm.py
Defines the SharedVariationalGPLVM class extending gpytorch.models.ApproximateGP.
Latent Space: Initializes gpytorch.priors.NormalPrior for the shared latent variables $\mathbf{X}$.
Likelihoods:
likelihood_img: For the $D_1$ image embeddings (e.g., Gaussian likelihood).
likelihood_tab: For the $D_2$ tabular features.
Kernels: RBF or Matern kernels for both modalities operating on $\mathbf{X}$.
3. Training & Inference
[NEW] scripts/train_gplvm.py
Runs inside the gpytorch Docker environment.
Loads the extracted embeddings $\mathbf{Y}^{(1)}$ and tabular targets $\mathbf{Y}^{(2)}$.
Identifies missing $\mathbf{Y}^{(2)}$ (e.g., missing survey scores) and sets up masking.
Optimizes the ELBO (Evidence Lower Bound) utilizing gpytorch.mlls.VariationalELBO.
Saves the optimized latent space $\mathbf{X}$ and the model checkpoints.
[NEW] scripts/impute_and_evaluate.py
Evaluates the trained model.
Uses the learned mappings to predict missing tabular values (like future VISA scores or tendon thickness) based purely on the image embeddings.
Will calculate metrics (MSE, Confidence Intervals) on a held-out validation set of tabular variables to prove the Shared GPLVM is correctly learning the correlation.
Verification Plan
Data Parsing Unit Tests: Verify that T1-23.3/R correctly joins against PPTTUS.csv row T1-23R and the game_stats_CU.csv for Player 23.
Latent Space Visualization: Once trained, we will generate a 2D PCA or UMAP plot of the latent space $\mathbf{X}$. We expect frames from the same patient to cluster together, and we expect gradients along the space corresponding to continuous tabular variables (like VISA scores).
Cross-Modal Prediction: Mask 20% of the known PPTTUS.csv data during training. Use the trained Shared GPLVM to predict those masked values using only the frame embeddings and evaluate the accuracy against the ground truth.