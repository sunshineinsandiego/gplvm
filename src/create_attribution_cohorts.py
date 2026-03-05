import os
import json
import pandas as pd
import glob
import re
from datetime import datetime
import argparse

def load_data(ppt_path, cu_path, fd_path, scan_dir):
    """Loads and returns combined patient, stats, and scan info."""
    ppt_df = pd.read_csv(ppt_path)
    if "FULLDATA" in ppt_df.columns:
        ppt_df = ppt_df[ppt_df["FULLDATA"] == 1]
    
    cu_df = pd.read_csv(cu_path)
    fd_df = pd.read_csv(fd_path)
    
    # Load available scans
    scans = []
    scan_paths = glob.glob(os.path.join(scan_dir, "T*-*.*", "*"))
    for path in scan_paths:
        if not os.path.isdir(path): continue
        parts = path.split(os.sep)
        knee = parts[-1]
        folder = parts[-2]
        match = re.match(r"T(\d+)-(\d+)\.(\d+)", folder)
        if match:
            scans.append({
                "scan_id": f"{folder}/{knee}",
                "scan_path": path,
                "team": int(match.group(1)),
                "player": int(match.group(2)),
                "timepoint": int(match.group(3)),
                "knee": knee,
                "ppt_key": f"T{match.group(1)}-{match.group(2)}{knee}"
            })
            
    return ppt_df, cu_df, fd_df, pd.DataFrame(scans)

def parse_minutes(m_val):
    if pd.isna(m_val): return 0.0
    s = str(m_val)
    if ":" in s:
        parts = s.split(":")
        try: return float(parts[0]) + float(parts[1])/60.0
        except: return 0.0
    try: return float(s)
    except: return 0.0

def calculate_average_minutes(team, player, timepoint, cu_df, fd_df):
    stats_df = cu_df if team == 1 else fd_df
    player_games = stats_df[stats_df["player #"] == player].copy()
    if len(player_games) == 0 or timepoint == 1:
        return 0.0
        
    player_games['date_parsed'] = pd.to_datetime(player_games['date'], errors='coerce')
    if timepoint == 2:
        cutoff = datetime.strptime("2/1/2022", "%m/%d/%Y")
        player_games = player_games[player_games['date_parsed'] < cutoff]
        
    player_games['Min_float'] = player_games['Min'].apply(parse_minutes)
    return player_games['Min_float'].mean() if len(player_games) > 0 else 0.0

def build_cohort_assignments(ppt_df, scans_df, cu_df, fd_df):
    """Evaluates each scan to determine which cohorts it belongs to."""
    cohorts = {
        "1_longitudinal_post_v_pre_L": {"target": [], "rival": []},
        "1_longitudinal_post_v_pre_R": {"target": [], "rival": []},
        "2_symptomatic_all": {"target": [], "rival": []},
        "2_symptomatic_pre": {"target": [], "rival": []},
        "2_symptomatic_mid": {"target": [], "rival": []},
        "2_symptomatic_post": {"target": [], "rival": []},
        "3_intra_subject_asymmetry": {"target": [], "rival": []},
        "4_silent_pathology": {"target": [], "rival": []},
        "4_predictive_injury": {"target": [], "rival": []},
        "4_clinical_crash_predictor": {"target": [], "rival": []},
        "4_severity_gradient": {"target": [], "rival": []},
        "4_osgood_schlatter": {"target": [], "rival": []},
        "5_healthy_adaptation": {"target": [], "rival": []},
        "5_injury_resilience": {"target": [], "rival": []},
        "5_positional_guards_v_bigs": {"target": [], "rival": []}
    }
    
    # Pre-compute useful aggregated fields for scans
    scan_meta = []
    for _, scan in scans_df.iterrows():
        ppt_matches = ppt_df[ppt_df["Number"] == scan["ppt_key"]]
        if len(ppt_matches) == 0: continue
        patient = ppt_matches.iloc[0]
        
        tp_prefix = {1: 'PRE', 2: 'MID', 3: 'POST'}.get(scan['timepoint'], None)
        if not tp_prefix: continue
        
        avg_min = calculate_average_minutes(scan['team'], scan['player'], scan['timepoint'], cu_df, fd_df)
        
        meta = {
            "scan_id": scan["scan_id"],
            "scan_path": scan["scan_path"],
            "team": scan["team"],
            "player": scan["player"],
            "timepoint": scan["timepoint"],
            "knee": scan["knee"],
            "pos": patient.get("POS", 0),
            "os": patient.get("OS", 0), # Osgood-Schlatter
            "full_health": patient.get("FULLHEALTH", 0),
            "symp": patient.get(f"SYMP{tp_prefix}", 0),
            "visa": patient.get(f"V{tp_prefix}", 100),
            "he": patient.get(f"HE{tp_prefix}", 0),
            "tlpt": patient.get("TLPT", 0), # Time lost
            "ttr": patient.get("TTR", 0),   # Time to return (injury markers)
            "dv13": patient.get("DV13", 0), # Future drop in VISA >= 13
            "avg_min": avg_min,
            # We track "require_video_frames" to tell downstream embedding scripts to
            # extract specific frames if it's a tight 1v1 N=2 comparison
            "require_video_frames": False 
        }
        scan_meta.append(meta)
        
    scan_df_meta = pd.DataFrame(scan_meta)
    
    if len(scan_df_meta) == 0:
        print("No matches found between scans and tabular data.")
        return cohorts
        
    def add_to_cohort(cohort_name, target_condition, rival_condition, require_video=False):
        targets = scan_df_meta[target_condition]["scan_id"].tolist()
        rivals = scan_df_meta[rival_condition]["scan_id"].tolist()
        
        cohorts[cohort_name]["target"].extend(targets)
        cohorts[cohort_name]["rival"].extend(rivals)
        cohorts[cohort_name]["require_video_frames"] = require_video
        
    # --- 1. Temporal Evolution ---
    # Target: POST L Knees. Rival: PRE L Knees
    add_to_cohort("1_longitudinal_post_v_pre_L", 
                 (scan_df_meta['knee'] == 'L') & (scan_df_meta['timepoint'] == 3),
                 (scan_df_meta['knee'] == 'L') & (scan_df_meta['timepoint'] == 1))
    
    # Target: POST R Knees. Rival: PRE R Knees
    add_to_cohort("1_longitudinal_post_v_pre_R", 
                 (scan_df_meta['knee'] == 'R') & (scan_df_meta['timepoint'] == 3),
                 (scan_df_meta['knee'] == 'R') & (scan_df_meta['timepoint'] == 1))
                 
    # Specific Player Progression (e.g., Player 20 R Knee POST vs PRE). Handled dynamically during runtime 
    # but we will set require_video_frames=True for this kind of logic.
                 
    # --- 2. Symptom Presence ---
    add_to_cohort("2_symptomatic_all", scan_df_meta['symp'] == 1, scan_df_meta['symp'] == 0)
    add_to_cohort("2_symptomatic_pre", (scan_df_meta['symp'] == 1) & (scan_df_meta['timepoint'] == 1), 
                                       (scan_df_meta['symp'] == 0) & (scan_df_meta['timepoint'] == 1))
    add_to_cohort("2_symptomatic_mid", (scan_df_meta['symp'] == 1) & (scan_df_meta['timepoint'] == 2), 
                                       (scan_df_meta['symp'] == 0) & (scan_df_meta['timepoint'] == 2))
    add_to_cohort("2_symptomatic_post", (scan_df_meta['symp'] == 1) & (scan_df_meta['timepoint'] == 3), 
                                        (scan_df_meta['symp'] == 0) & (scan_df_meta['timepoint'] == 3))
    
    # --- 3. Asymmetry ---
    # Intra-subject (Gold Standard): Target (Injured L) vs Rival (Healthy R).
    # We find patients where one knee SYMP=1 and the other is SYMP=0 at the same timepoint
    intra_targets = []
    intra_rivals = []
    grouped = scan_df_meta.groupby(['team', 'player', 'timepoint'])
    for name, group in grouped:
        if len(group) == 2:
            symp_scans = group[group['symp'] == 1]
            healthy_scans = group[group['symp'] == 0]
            if len(symp_scans) == 1 and len(healthy_scans) == 1:
                intra_targets.append(symp_scans.iloc[0]['scan_id'])
                intra_rivals.append(healthy_scans.iloc[0]['scan_id'])
    cohorts["3_intra_subject_asymmetry"]["target"] = intra_targets
    cohorts["3_intra_subject_asymmetry"]["rival"] = intra_rivals
    cohorts["3_intra_subject_asymmetry"]["require_video_frames"] = True # N=2 logic
    
    # --- 4. Pathology & Prediction ---
    add_to_cohort("4_silent_pathology", (scan_df_meta['he'] == 1) & (scan_df_meta['symp'] == 1),
                                        (scan_df_meta['he'] == 1) & (scan_df_meta['symp'] == 0))
                                        
    add_to_cohort("4_predictive_injury", (scan_df_meta['timepoint'] == 1) & ((scan_df_meta['tlpt'] == 1) | (scan_df_meta['ttr'] == 1)),
                                         (scan_df_meta['timepoint'] == 1) & (scan_df_meta['tlpt'] == 0) & (scan_df_meta['ttr'] == 0))
    
    add_to_cohort("4_clinical_crash_predictor", (scan_df_meta['timepoint'] == 1) & (scan_df_meta['dv13'] == 1),
                                                (scan_df_meta['timepoint'] == 1) & (scan_df_meta['dv13'] == 0))
                                                
    add_to_cohort("4_severity_gradient", scan_df_meta['visa'] <= 70, scan_df_meta['visa'] >= 90)
    
    add_to_cohort("4_osgood_schlatter", scan_df_meta['os'] == 1, scan_df_meta['os'] == 0)

    # --- 5. Load Management ---
    add_to_cohort("5_healthy_adaptation", (scan_df_meta['full_health'] == 1) & (scan_df_meta['avg_min'] > 25),
                                          (scan_df_meta['full_health'] == 1) & (scan_df_meta['avg_min'] < 10))
                                          
    add_to_cohort("5_injury_resilience", ((scan_df_meta['symp'] == 1) | (scan_df_meta['tlpt'] == 1)) & (scan_df_meta['avg_min'] > 25),
                                         (scan_df_meta['full_health'] == 1) & (scan_df_meta['avg_min'] > 25))
                                         
    add_to_cohort("5_positional_guards_v_bigs", scan_df_meta['pos'] == 1, scan_df_meta['pos'] == 2)
    
    return cohorts

def main():
    parser = argparse.ArgumentParser(description="Build attribution cohort assignments")
    parser.add_argument("--scan_dir", type=str, default="scan")
    parser.add_argument("--ppt_csv", type=str, default="data/PPTTUS.csv")
    parser.add_argument("--cu_csv", type=str, default="data/game_stats_CU.csv")
    parser.add_argument("--fd_csv", type=str, default="data/game_stats_FD.csv")
    parser.add_argument("--out_json", type=str, default="data/attribution_cohorts.json")
    
    args = parser.parse_args()
    
    ppt_df, cu_df, fd_df, scans_df = load_data(args.ppt_csv, args.cu_csv, args.fd_csv, args.scan_dir)
    cohorts = build_cohort_assignments(ppt_df, scans_df, cu_df, fd_df)
    
    # Prune empty cohorts
    cohorts = {k: v for k, v in cohorts.items() if len(v["target"]) > 0 or len(v["rival"]) > 0}
    
    print(f"Generated {len(cohorts)} valid attribution cohorts.")
    for name, c in list(cohorts.items())[:5]:
        print(f"  {name}: Target N={len(c['target'])}, Rival N={len(c['rival'])}, VideoFallback={c.get('require_video_frames', False)}")
        
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(cohorts, f, indent=2)
        
    print(f"\nSaved cohort definitions to {args.out_json}")

if __name__ == "__main__":
    main()
