import os
import pickle
import numpy as np
import pandas as pd
import cupy as cp
from itertools import combinations

# CONFIG
OUTPUT_DIR = "data/output"
RO_CACHE_DIR = os.path.join(OUTPUT_DIR, "ro_cache")
SCORE_DIR = os.path.join(OUTPUT_DIR, "game_scores") # Path to scores_subX.p
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "combinations_up_to_8.csv")

INPUT_DIR = "data/input"
OUTCOMES_PATH = os.path.join(INPUT_DIR, "outcomes.xlsx")

TOP_PERCENTILE = 1.0
RO_EPSILON = 1e-6
USE_CONTRAST = False
MAX_K = 8 

os.makedirs(RO_CACHE_DIR, exist_ok=True)


def get_subject_ids_from_excel(path):
    """Load subject IDs from data/input/outcomes.xlsx"""
    try:
        # Read the Excel file
        df_meta = pd.read_excel(path)
        
        # Check if 'subject_id' column exists
        if 'subject_id' not in df_meta.columns:
            # Fallback: if 'subject' is the column name instead
            col_name = 'subject' if 'subject' in df_meta.columns else df_meta.columns[0]
            print(f"Warning: 'subject_id' not found. Using '{col_name}' column.")
        else:
            col_name = 'subject_id'
            
        # Get unique IDs and sort them
        ids = sorted(df_meta[col_name].dropna().unique().astype(int).tolist())
        print(f"Found {len(ids)} subjects in metadata: {ids}")
        return ids
    except Exception as e:
        print(f"Error reading metadata file: {e}")
        return []


def load_resection(path):
    with open(path, 'rb') as f:
        resection = pickle.load(f)
    print(f"[load] Resection nodes loaded.")
    return resection

# 1. Ro calculation

def compute_overlap(network, resection_nodes):
    if not network:
        return 0.0
    hits = 0
    for edge in network:
        parts = str(edge).split('-')
        node_a = parts[0]
        node_b = '-'.join(parts[1:]) if len(parts) >= 2 else ''
        if node_a in resection_nodes or node_b in resection_nodes:
            hits += 1
    return hits / len(network)


def score_networks(game_scores):
    scored = {net: sum(votes) for net, votes in game_scores.items()}
    return dict(sorted(scored.items(), key=lambda x: x[1], reverse=True))


def compute_ro(game_scores, resection_nodes, top_percentile, epsilon, use_contrast):
    if not game_scores:
        return np.nan

    scored    = score_networks(game_scores)
    networks  = list(scored.keys())
    n_total   = len(networks)
    if n_total == 0:
        return np.nan

    n_top         = max(1, int(np.ceil(n_total * top_percentile / 100.0)))
    top_networks  = networks[:n_top]
    rest_networks = networks[n_top:]

    top_overlaps  = [compute_overlap(net, resection_nodes) for net in top_networks]
    rest_overlaps = [compute_overlap(net, resection_nodes) for net in rest_networks]

    mean_top  = np.mean(top_overlaps)  if top_overlaps  else 0.0
    mean_rest = np.mean(rest_overlaps) if rest_overlaps else 0.0

    if use_contrast:
        return (mean_top - mean_rest) / (mean_top + mean_rest + epsilon)
    return mean_top / (mean_rest + epsilon)


# 2. GPU processing

def run_gpu_combinations(subject_id, cm_dict, max_k):
    cms = sorted(cm_dict.keys())
    scores_gpu = cp.array([cm_dict[c] for c in cms], dtype=cp.float32)
    
    rows = []
    for k in range(1, max_k + 1):
        print(f"  > Processing size {k}...")
        all_combs = list(combinations(range(len(cms)), k))
        
        # Batching to avoid GPU/RAM overflow for large combinations (k=8)
        batch_size = 500000 
        for i in range(0, len(all_combs), batch_size):
            batch = all_combs[i:i+batch_size]
            idx_gpu = cp.array(batch)
            
            # GPU Parallel Mean Calculation
            means = cp.mean(scores_gpu[idx_gpu], axis=1)
            means_cpu = cp.asnumpy(means)
            
            for j, m_val in enumerate(means_cpu):
                rows.append({
                    "subject_id": subject_id,
                    "CM_combination": "|".join([cms[idx] for idx in batch[j]]),
                    "n_CM": k,
                    "mean_score": float(m_val)
                })
    return rows

# 3. MAIN

def main():
    
    all_results = []
    # Load subjects IDs from outcomes.xlsx
    subject_ids = get_subject_ids_from_excel(OUTCOMES_PATH)

    for sid in subject_ids:
        cache_path = os.path.join(RO_CACHE_DIR, f"Ro_sub{sid}.csv")
        cm_ro_scores = {}

        # CHECK Ro CACHE FIRST
        if os.path.exists(cache_path):
            print(f"Subject {sid}: Loading from cache...")
            df_cache = pd.read_csv(cache_path)
            # Remove 'subject' column, convert rest to dict
            cm_ro_scores = df_cache.drop(columns=['subject']).iloc[0].to_dict()
        else:
            # COMPUTE Ro FROM SCRATCH
            raw_path = os.path.join(SCORE_DIR, f"scores_sub{sid}.p")
            if not os.path.exists(raw_path):
                print(f"Subject {sid}: Missing raw score file. Skipping.")
                continue
                
            print(f"Subject {sid}: Calculating Ro...")
            with open(raw_path, 'rb') as f:
                data = pickle.load(f)
            
            resection_nodes = load_resection(f"{INPUT_DIR}/{sid}_RESECTION.p")[sid]
            all_cms = sorted(set(cm for (cm, s) in data.keys()))
            
            for cm in all_cms:
                entry = data.get((cm, 4))
                if entry:
                    ro = compute_ro(entry.get('game_scores', {}), resection_nodes, TOP_PERCENTILE, RO_EPSILON, use_contrast=False)
                    if not np.isnan(ro):
                        cm_ro_scores[cm] = ro
            
            # Save to cache for next time
            cache_df = pd.DataFrame([cm_ro_scores])
            cache_df.insert(0, 'subject', sid)
            cache_df.to_csv(cache_path, index=False)

        # RUN GPU COMBINATIONS
        if cm_ro_scores:
            print((f"Running combinations: up to {MAX_K}..."))
            sub_rows = run_gpu_combinations(sid, cm_ro_scores, MAX_K)
            all_results.extend(sub_rows)

    # SAVE
    print(f"Saving scores table with {len(all_results)} rows...")
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    main()