import os
import glob
import re
import json
import pandas as pd
import numpy as np
from datetime import datetime
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
    
    return {
        "AVG_MIN": float(avg_min) if not pd.isna(avg_min) else 0.0,
        "AVG_PTS": float(avg_pts) if not pd.isna(avg_pts) else 0.0,
        "AVG_REB": float(avg_reb) if not pd.isna(avg_reb) else 0.0,
        "AVG_AST": float(avg_ast) if not pd.isna(avg_ast) else 0.0,
        "AVG_PLUS_MINUS": float(avg_pm) if not pd.isna(avg_pm) else 0.0,
        "GP": len(player_games)
    }

def compile_tabular_data(scan_dir, ppt_csv, game_stats_cu, game_stats_fd, encoder, embeddings_dir):
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
    skipped_no_dpp = 0
    
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
        else:
            # Fallback if unknown timepoint
            row['V_SCORE'] = np.nan
            row['PT_TEND'] = np.nan
            row['TT_TEND'] = np.nan
            row['HE'] = np.nan
            row['AT_THICK'] = np.nan
            
        # 3. Game stats aggregation
        # Team 1 = CU (game_stats_CU.csv), Team 2 = FD (game_stats_FD.csv)
        stats_df = cu_stats if scan['team'] == 1 else fd_stats
        
        game_aggs = aggregate_game_stats(scan['team'], scan['player'], scan['timepoint'], stats_df)
        
        for k, v in game_aggs.items():
            row[k] = v
            
        # 4. Filter by DPP keyframes
        subject_id = f"T{scan['team']}-{scan['player']}.{scan['timepoint']}"
        knee_id = scan['knee']
        search_path = os.path.join(embeddings_dir, subject_id, knee_id, encoder, "*", "selected_keyframes_dpp.json")
        json_files = glob.glob(search_path)
        
        if not json_files:
            # Check if features are extracted at all
            encoder_dir = os.path.join(embeddings_dir, subject_id, knee_id, encoder)
            if not os.path.exists(encoder_dir) or not os.listdir(encoder_dir):
                print(f"Extracting {encoder} features for {scan['scan_path']} on the fly...")
                workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                rel_scan_path = os.path.relpath(scan['scan_path'], workspace_dir)
                import subprocess
                # If it's a video/multiframe, ask for 10 keyframes. If it's a single image, features_dinov3 
                # will just return 1 frame and DPP will gracefully accept it.
                import sys
                cmd = [
                    sys.executable, f"{workspace_dir}/src/features_{encoder}.py",
                    "--input", f"{workspace_dir}/{rel_scan_path}",
                    "--output-root", f"{workspace_dir}/{embeddings_dir}",
                    "--save-encodings", "false",
                    "--frame-index", "all",
                    "--dpp-keyframes", "10"
                ]
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    json_files = glob.glob(search_path)
                except subprocess.CalledProcessError as e:
                    print(f"Error extracting features: {e.stderr}")

            # If JSON is still not found, try generating it on the fly via the old method
            if not json_files and os.path.exists(encoder_dir):
                # We need to find the specific subset directory (e.g., 'all' or 'idx0')
                subsets = [d for d in os.listdir(encoder_dir) if os.path.isdir(os.path.join(encoder_dir, d))]
                
                if subsets:
                    subset_dir = os.path.join(encoder_dir, subsets[0])
                    print(f"Generating DPP keyframes on the fly for {subset_dir}...")
                    
                    # Convert absolute paths to workspace relative paths
                    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
                    rel_subset_dir = os.path.relpath(subset_dir, workspace_dir)
                    
                    # Run python command
                    import subprocess
                    import sys
                    cmd = [
                        sys.executable, f"{workspace_dir}/src/select_keyframes_dpp.py",
                        "--input", f"{workspace_dir}/{rel_subset_dir}"
                    ]
                    
                    try:
                        subprocess.run(cmd, check=True, capture_output=True, text=True)
                        json_files = glob.glob(search_path)
                        if not json_files:
                            print(f"Warning: Script succeeded but {search_path} still not found.")
                    except subprocess.CalledProcessError as e:
                        print(f"Error generating keyframes for {subset_dir}: {e.stderr}")
            
        if not json_files:
            skipped_no_dpp += 1
            # Still output the row without a frame_id just in case, but typically we want per-frame
            # Let's emit a single row to permit testing functionality when DP isn't run
            frame_row = row.copy()
            frame_row['frame_id'] = None
            frame_row['embedding_dir'] = None
            merged_rows.append(frame_row)
            continue
            
        with open(json_files[0], 'r') as f:
            dpp_data = json.load(f)
            
        selected_frames = dpp_data.get("selected_frame_ids", [])
        
        for frame_id in selected_frames:
            frame_row = row.copy()
            frame_row['frame_id'] = frame_id
            frame_row['embedding_dir'] = os.path.dirname(json_files[0])
            merged_rows.append(frame_row)
            
    final_df = pd.DataFrame(merged_rows)
    print(f"Successfully joined {len(final_df)} scan records to tabular data. Skipped {skipped_patient} missing tabulars. Scans missing DPP JSON tracking: {skipped_no_dpp}")
    return final_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile intermediate tabular records for Shared GPLVM")
    parser.add_argument("--scan_dir", type=str, default="scan", help="Path to 'scan' directory")
    parser.add_argument("--ppt_csv", type=str, default="data/PPTTUS.csv", help="Path to PPTTUS.csv")
    parser.add_argument("--cu_csv", type=str, default="data/game_stats_CU.csv", help="Path to game_stats_CU.csv")
    parser.add_argument("--fd_csv", type=str, default="data/game_stats_FD.csv", help="Path to game_stats_FD.csv")
    parser.add_argument("--encoder", type=str, choices=["dinov3", "medsam2"], default="dinov3", help="Encoder used (dinov3 or medsam2)")
    parser.add_argument("--embeddings_dir", type=str, default="scan_output", help="Directory where embeddings are saved")
    parser.add_argument("--out_manifest", type=str, default="data/manifest.csv", help="Output path for manifest CSV")
    parser.add_argument("--out_xtab", type=str, default="data/X_tab.npy", help="Output path for X_tab tensor")
    parser.add_argument("--out_ytarget", type=str, default="data/Y_target.npy", help="Output path for Y_target tensor")
    
    args = parser.parse_args()
    
    df = compile_tabular_data(args.scan_dir, args.ppt_csv, args.cu_csv, args.fd_csv, args.encoder, args.embeddings_dir)
    
    if not df.empty:
        static = ['HEIGHT', 'WEIGHT', 'BMI', 'AGE', 'POS', 'YEARS']
        game = ['AVG_MIN', 'AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_PLUS_MINUS', 'GP']
        clinical = ['V_SCORE', 'PT_TEND', 'TT_TEND', 'HE', 'AT_THICK']
        base = ['frame_id', 'scan_path', 'ppt_key', 'timepoint', 'team', 'player', 'knee', 'embedding_dir']
        
        # Verify columns exist
        existing_cols = [c for c in base + static + game + clinical if c in df.columns]
        df = df[existing_cols]
        
        print("\nPreview of Compiled Tabular Row (idx 0):")
        print(df.iloc[0].to_dict())
        
        # Form X_tab and Y_target
        X_tab = df[[c for c in static + game if c in df.columns]].values.astype(np.float32)
        Y_target = df[[c for c in clinical if c in df.columns]].values.astype(np.float32)
        
        np.save(args.out_xtab, X_tab)
        np.save(args.out_ytarget, Y_target)
        df.to_csv(args.out_manifest, index=False)
        
        print(f"\nSaved X_tab of shape {X_tab.shape} to {args.out_xtab}")
        print(f"Saved Y_target of shape {Y_target.shape} to {args.out_ytarget}")
        print(f"Saved manifest dataset to {args.out_manifest}")
    else:
        print("Failed to compile dataset. DataFrame is empty.")
