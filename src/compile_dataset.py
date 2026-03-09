import os
import glob
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime
import subprocess
import tempfile
import argparse

def parse_scan_directories(scan_dir):
    """
    Parses the scan directories to extract Team, Player, Timepoint, Knee.
    Returns a pandas DataFrame of available scans.
    """
    scans = []
    
    # Expected format: T[Team]-[Player].[Timepoint]/[Knee]
    # e.g., scan/T1-20.1/R
    scan_paths = glob.glob(os.path.join(scan_dir, "T*-*.*", "*"))
    for path in scan_paths:
        if not os.path.isdir(path):
            continue
            
        parts = path.split(os.sep)
        knee = parts[-1]
        folder = parts[-2]
        
        match = re.match(r"T(\d+)-(\d+)\.(\d+)", folder)
        if match:
            team = int(match.group(1))
            player = int(match.group(2))
            timepoint = int(match.group(3))
            
            scans.append({
                "scan_path": path,
                "team": team,
                "player": player,
                "timepoint": timepoint,
                "knee": knee,
                "ppt_key": f"T{team}-{player}{knee}"
            })
            
    return pd.DataFrame(scans)

def load_and_filter_patient_data(ppt_csv_path):
    """
    Loads PPTTUS.csv and returns a dataframe filtered for FULLDATA=1.
    """
    df = pd.read_csv(ppt_csv_path)
    if "FULLDATA" in df.columns:
        df = df[df["FULLDATA"] == 1]
    return df

def aggregate_game_stats(team, player, timepoint, game_stats_df):
    """
    Aggregates game stats based on timepoint.
    Timepoint 1: 0 games (preseason)
    Timepoint 2: Games before 2/1/2022 (midseason)
    Timepoint 3: All games (postseason)
    """
    base_stats = {
        "AVG_MIN": 0.0,
        "AVG_PTS": 0.0,
        "AVG_REB": 0.0,
        "AVG_AST": 0.0,
        "AVG_PLUS_MINUS": 0.0,
        "TOT_MIN": 0.0,
        "TOT_PTS": 0.0,
        "TOT_REB": 0.0,
        "TOT_AST": 0.0,
        "TOT_PLUS_MINUS": 0.0,
        "GP": 0 # Games played
    }
    
    if timepoint == 1:
        return base_stats
        
    cutoff_date = None
    if timepoint == 2:
        cutoff_date = datetime.strptime("2/1/2022", "%m/%d/%Y")
        
    # Filter by player ID
    player_games = game_stats_df[game_stats_df["player #"] == player].copy()
    
    if len(player_games) == 0:
        return base_stats
        
    # Filter by date
    player_games['date_parsed'] = pd.to_datetime(player_games['date'], errors='coerce')
    
    if cutoff_date:
        player_games = player_games[player_games['date_parsed'] < cutoff_date]
        
    if len(player_games) == 0:
        return base_stats
        
    # Parse Minutes from MM:SS or raw numeric
    def parse_min(m):
        if pd.isna(m): return 0.0
        m_str = str(m)
        if ":" in m_str:
            parts = m_str.split(":")
            if len(parts) >= 2:
                try:
                    return float(parts[0]) + float(parts[1])/60.0
                except ValueError:
                    return 0.0
        try:
            return float(m)
        except ValueError:
            return 0.0
            
    player_games['Min_float'] = player_games['Min'].apply(parse_min)
    
    # Calculate averages
    avg_min = player_games['Min_float'].mean()
    avg_pts = pd.to_numeric(player_games['Pts'], errors='coerce').mean()
    avg_reb = pd.to_numeric(player_games['Tot Reb'], errors='coerce').mean()
    avg_ast = pd.to_numeric(player_games['Ast'], errors='coerce').mean()
    avg_pm = pd.to_numeric(player_games['+/-'], errors='coerce').mean()
    
    tot_min = player_games['Min_float'].sum()
    tot_pts = pd.to_numeric(player_games['Pts'], errors='coerce').sum()
    tot_reb = pd.to_numeric(player_games['Tot Reb'], errors='coerce').sum()
    tot_ast = pd.to_numeric(player_games['Ast'], errors='coerce').sum()
    tot_pm = pd.to_numeric(player_games['+/-'], errors='coerce').sum()
    
    return {
        "AVG_MIN": float(avg_min) if not pd.isna(avg_min) else 0.0,
        "AVG_PTS": float(avg_pts) if not pd.isna(avg_pts) else 0.0,
        "AVG_REB": float(avg_reb) if not pd.isna(avg_reb) else 0.0,
        "AVG_AST": float(avg_ast) if not pd.isna(avg_ast) else 0.0,
        "AVG_PLUS_MINUS": float(avg_pm) if not pd.isna(avg_pm) else 0.0,
        "TOT_MIN": float(tot_min),
        "TOT_PTS": float(tot_pts),
        "TOT_REB": float(tot_reb),
        "TOT_AST": float(tot_ast),
        "TOT_PLUS_MINUS": float(tot_pm),
        "GP": len(player_games)
    }

def compile_tabular_data(scan_dir, ppt_csv, game_stats_cu, game_stats_fd, encoder, n_keyframes, off_top=0, off_bottom=0, off_left=0, off_right=0):
    """
    Executes Steps 1 and 2 of the dataset compilation pipeline.
    Parses scans, extracts static patient info, targets based on timepoint, 
    and averaged game stats up to the timepoint cutoff.
    Cross-references selected_keyframes_dpp.json to emit one row per selected frame.
    """
    scans_df = parse_scan_directories(scan_dir)
    print(f"Found {len(scans_df)} scan directories across {scans_df['timepoint'].nunique()} timepoints.")
    
    if len(scans_df) == 0:
        print("No scan directories found. Returning empty dataframe.")
        return pd.DataFrame()

    ppt_df = load_and_filter_patient_data(ppt_csv)
    print(f"Loaded {len(ppt_df)} patients from FULLDATA PPTTUS.")
    
    cu_stats = pd.read_csv(game_stats_cu)
    fd_stats = pd.read_csv(game_stats_fd)
    
    merged_rows = []
    skipped_patient = 0
    
    for _, scan in scans_df.iterrows():
        ppt_key = scan['ppt_key']
        # e.g., 'T1-20R'
        patient_info = ppt_df[ppt_df["Number"] == ppt_key]
        
        if len(patient_info) == 0:
            skipped_patient += 1
            continue
            
        patient_info = patient_info.iloc[0]
        
        # Base row identifiers
        row = {
            "ppt_key": ppt_key,
            "scan_path": scan['scan_path'],
            "team": scan['team'],
            "player": scan['player'],
            "timepoint": scan['timepoint'],
            "knee": scan['knee']
        }
        
        # 1. Static predictors
        static_cols = ['HEIGHT', 'WEIGHT', 'BMI', 'AGE', 'POS', 'YEARS']
        for col in static_cols:
            row[col] = patient_info.get(col, np.nan)
                
        # 2. Dynamic targets based on timepoint (PRE, MID, POST)
        prefix_map = {1: 'PRE', 2: 'MID', 3: 'POST'}
        time_prefix = prefix_map.get(scan['timepoint'])
        
        if time_prefix:
            # Map dynamic clinical outcomes to generic names
            row['V_SCORE'] = patient_info.get(f'V{time_prefix}', np.nan)
            row['PT_TEND'] = patient_info.get(f'PT{time_prefix}', np.nan)
            row['TT_TEND'] = patient_info.get(f'TT{time_prefix}', np.nan)
            row['HE'] = patient_info.get(f'HE{time_prefix}', np.nan)
            row['AT_THICK'] = patient_info.get(f'AT{time_prefix}', np.nan)
            
            # Include specific requested variables
            row['SYMP'] = patient_info.get('SYMP', np.nan)
            row['HEPRE'] = patient_info.get('HEPRE', np.nan)
            row['HEMID'] = patient_info.get('HEMID', np.nan)
            row['HEPOST'] = patient_info.get('HEPOST', np.nan)
        else:
            # Fallback if unknown timepoint
            row['V_SCORE'] = np.nan
            row['PT_TEND'] = np.nan
            row['TT_TEND'] = np.nan
            row['HE'] = np.nan
            row['AT_THICK'] = np.nan
            row['SYMP'] = np.nan
            row['HEPRE'] = np.nan
            row['HEMID'] = np.nan
            row['HEPOST'] = np.nan
            
        # 3. Game stats aggregation
        # Team 1 = CU (game_stats_CU.csv), Team 2 = FD (game_stats_FD.csv)
        stats_df = cu_stats if scan['team'] == 1 else fd_stats
        
        game_aggs = aggregate_game_stats(scan['team'], scan['player'], scan['timepoint'], stats_df)
        
        for k, v in game_aggs.items():
            row[k] = v
            
        # 4. Run Feature Extraction & DPP Selection on the fly
        # We use a temporary directory to capture the JSON output from the feature extractor
        selected_frames = []
        
        import sys
        import torch
        workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        src_dir = os.path.join(workspace_dir, "src")
        if src_dir not in sys.path:
            sys.path.insert(0, src_dir)
            
        from sequence_frame_loader import detect_all_sources, load_selected_rgb_frame
        from pathlib import Path
        
        try:
            sources = detect_all_sources(Path(scan['scan_path']))
        except Exception as e:
            print(f"Error extracting sequence sources from {scan['scan_path']}: {e}")
            sources = []
            
        if n_keyframes == "all":
            for source in sources:
                for f_idx in range(source.frame_count):
                    frame_id = f"{source.source_stem}_f{f_idx:04d}"
                    frame_row = row.copy()
                    frame_row['frame_id'] = frame_id
                    frame_row['scan_path'] = str(source.primary_path or source.sequence_paths[f_idx])
                    merged_rows.append(frame_row)
            continue
            
        else:
            try:
                n_select_target = int(n_keyframes)
            except ValueError:
                n_select_target = 0
                
            if n_select_target > 0:
                from select_keyframes_dpp import build_cosine_matrices, greedy_dpp_map_order
                
                # Lazy load model if not already loaded globally
                global _LAZY_MODEL, _LAZY_DEVICE, _LAZY_ENCODER_NAME
                if '_LAZY_MODEL' not in globals() or _LAZY_ENCODER_NAME != encoder:
                    print(f"Loading {encoder} model for on-the-fly DPP feature extraction...")
                    _LAZY_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    _LAZY_ENCODER_NAME = encoder
                    if encoder == "dinov3":
                        import dinov3
                        model_name = "dinov3_vits16"
                        weights_path = os.environ.get("DINOV3_WEIGHTS", "/workspace/models/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth")
                        print(f"Loading {model_name} from local path: {weights_path}")
                        _LAZY_MODEL = torch.hub.load(
                            "/app/dinov3", 
                            model_name, 
                            source="local", 
                            pretrained=False
                        )
                        _LAZY_MODEL.load_state_dict(torch.load(weights_path, map_location=_LAZY_DEVICE, weights_only=True))
                    elif encoder == "medsam2":
                        # Import specific to medsam2 if implemented
                        pass
                    
                    if getattr(sys.modules[__name__], '_LAZY_MODEL', None):
                        _LAZY_MODEL.eval().to(_LAZY_DEVICE)
                        
                for source in sources:
                    is_static = source.source_type in ["image_single", "dicom_single"]
                    if is_static or source.frame_count <= n_select_target:
                        # Automatically select all frames if single-frame or fewer frames than requested subset
                        for f_idx in range(source.frame_count):
                            frame_id = f"{source.source_stem}_f{f_idx:04d}"
                            frame_row = row.copy()
                            frame_row['frame_id'] = frame_id
                            frame_row['scan_path'] = str(source.primary_path or source.sequence_paths[f_idx])
                            merged_rows.append(frame_row)
                    else:
                        print(f"Performing on-the-fly DPP selection ({n_select_target} from {source.frame_count}) for {source.source_stem}...")
                        in_memory_feats = []
                        in_memory_ids = []
                        in_memory_paths = []
                        
                        if encoder == "dinov3":
                            from features_dinov3 import _preprocess, _extract_features
                            for f_idx in range(source.frame_count):
                                base_name = f"{source.source_stem}_f{f_idx:04d}"
                                frame_rgb_u8 = load_selected_rgb_frame(source, f_idx)
                                image_t = _preprocess(frame_rgb_u8, patch_size=16, off_top=off_top, off_bottom=off_bottom, off_left=off_left, off_right=off_right, imagenet=False)
                                feats = _extract_features(_LAZY_MODEL, image_t, _LAZY_DEVICE)
                                in_memory_feats.append(feats.numpy())
                                in_memory_ids.append(base_name)
                                in_memory_paths.append(str(source.primary_path or source.sequence_paths[f_idx]))
                        else:
                            # Fallback if other encoders requested
                            pass
                            
                        if in_memory_feats:
                            embeddings = np.stack(in_memory_feats, axis=0)
                            if embeddings.ndim > 2:
                                embeddings = embeddings.reshape(embeddings.shape[0], -1)
                            
                            coverage_sim, kernel = build_cosine_matrices(embeddings)
                            order = greedy_dpp_map_order(kernel, k=n_select_target + 5)
                            selected_indices = order[:n_select_target]
                            
                            for idx in selected_indices:
                                frame_row = row.copy()
                                frame_row['frame_id'] = in_memory_ids[idx]
                                frame_row['scan_path'] = in_memory_paths[idx]
                                merged_rows.append(frame_row)
            
    final_df = pd.DataFrame(merged_rows)
    print(f"Successfully joined {len(final_df)} scan records to tabular data. Skipped {skipped_patient} missing tabulars.")
    return final_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile intermediate tabular records for Shared GPLVM")
    parser.add_argument("--scan_dir", type=str, default="scan", help="Path to 'scan' directory")
    parser.add_argument("--ppt_csv", type=str, default="data/PPTTUS.csv", help="Path to PPTTUS.csv")
    parser.add_argument("--cu_csv", type=str, default="data/game_stats_CU.csv", help="Path to game_stats_CU.csv")
    parser.add_argument("--fd_csv", type=str, default="data/game_stats_FD.csv", help="Path to game_stats_FD.csv")
    parser.add_argument("--encoder", type=str, choices=["dinov3", "medsam2"], default="dinov3", help="Encoder used (dinov3 or medsam2)")
    parser.add_argument("--n_keyframes", type=str, default="all", help="Number of keyframes to select per video (default: 'all', or pass an integer)")
    parser.add_argument("--out_manifest", type=str, default="data/manifest.csv", help="Output path for manifest CSV")
    
    parser.add_argument("--off_top", type=int, default=0, help="Pixels to crop from top of frame")
    parser.add_argument("--off_bottom", type=int, default=0, help="Pixels to crop from bottom of frame")
    parser.add_argument("--off_left", type=int, default=0, help="Pixels to crop from left of frame")
    parser.add_argument("--off_right", type=int, default=0, help="Pixels to crop from right of frame")
    
    args = parser.parse_args()
    
    df = compile_tabular_data(
        args.scan_dir, args.ppt_csv, args.cu_csv, args.fd_csv, args.encoder, args.n_keyframes,
        off_top=args.off_top, off_bottom=args.off_bottom, off_left=args.off_left, off_right=args.off_right
    )
    
    if not df.empty:
        static = ['HEIGHT', 'WEIGHT', 'BMI', 'AGE', 'POS', 'YEARS']
        game = ['AVG_MIN', 'AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_PLUS_MINUS', 'TOT_MIN', 'TOT_PTS', 'TOT_REB', 'TOT_AST', 'TOT_PLUS_MINUS', 'GP']
        clinical = ['V_SCORE', 'PT_TEND', 'TT_TEND', 'HE', 'AT_THICK', 'SYMP', 'HEPRE', 'HEMID', 'HEPOST']
        base = ['frame_id', 'scan_path', 'ppt_key', 'timepoint', 'team', 'player', 'knee']
        
        # Verify columns exist
        existing_cols = [c for c in base + static + game + clinical if c in df.columns]
        df = df[existing_cols]
        
        print("\nPreview of Compiled Tabular Row (idx 0):")
        print(df.iloc[0].to_dict())
        
        df.to_csv(args.out_manifest, index=False)
        print(f"\nSaved manifest dataset to {args.out_manifest}")
    else:
        print("Failed to compile dataset. DataFrame is empty.")
