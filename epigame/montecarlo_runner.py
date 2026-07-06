import os
import pickle
import warnings
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# PATHS & CONFIGURATION
BASE_DIR = "data"
INPUT_DIR = "data/input"
METADATA_PATH = os.path.join(BASE_DIR, "input/outcomes.xlsx")

CACHE_DIR = os.path.join(BASE_DIR, "output/ro_cache")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RAW_SCORE_DIR = os.path.join(BASE_DIR, "output/game_scores")

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

CFG = dict(
    n_votes_target = 3,
    subset_good = 14,
    subset_poor = 7,
    top_percentile = 1.0,
    ro_epsilon = 1e-6,
    use_contrast_ro = False,
    seed = 42,
    max_attempts = 500000
)

RNG = np.random.default_rng(CFG['seed'])


def load_resection(path):
    with open(path, 'rb') as f:
        resection = pickle.load(f)
    print(f"[load] Resection nodes loaded.")
    return resection


def compute_overlap(network, resection_nodes):
    if not network: return 0.0
    hits = 0
    for edge in network:
        parts = str(edge).split('-')
        if parts[0] in resection_nodes or (len(parts) >= 2 and parts[1] in resection_nodes):
            hits += 1
    return hits / len(network)


def compute_ro_value(game_scores, resection_nodes):
    if not game_scores: return np.nan
    # Score networks by summing votes
    scored = {net: sum(votes) for net, votes in game_scores.items()}
    sorted_nets = [k for k, v in sorted(scored.items(), key=lambda x: x[1], reverse=True)]
    
    n_total = len(sorted_nets)
    n_top = max(1, int(np.ceil(n_total * CFG['top_percentile'] / 100.0)))
    
    top_nets, rest_nets = sorted_nets[:n_top], sorted_nets[n_top:]
    top_overlaps = [compute_overlap(net, resection_nodes) for net in top_nets]
    rest_overlaps = [compute_overlap(net, resection_nodes) for net in rest_nets]
    
    m_top = np.mean(top_overlaps) if top_overlaps else 0.0
    m_rest = np.mean(rest_overlaps) if rest_overlaps else 0.0
    
    if CFG['use_contrast_ro']:
        return (m_top - m_rest) / (m_top + m_rest + CFG['ro_epsilon'])
    
    return m_top / (m_rest + CFG['ro_epsilon'])


def generate_ro_csv(sub_idx, resection_nodes):
    filename = os.path.join(RAW_SCORE_DIR, f'scores_sub{sub_idx}.p')
    if not os.path.exists(filename):
        return None
    
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    ro_per_cm = {}
    all_cms = sorted(set(cm for (cm, s) in data.keys()))
    for cm in all_cms:
        # Looking for size 4 specifically per your logic
        entry = data.get((cm, 4))
        gs = entry.get('game_scores', {}) if entry else None
        ro_per_cm[cm] = compute_ro_value(gs, resection_nodes)
    
    df = pd.DataFrame([ro_per_cm])
    df.insert(0, 'subject', sub_idx)
    save_path = os.path.join(CACHE_DIR, f'Ro_sub{sub_idx}.csv')
    df.to_csv(save_path, index=False)
    return df


def evaluate_and_vote(subset, test_patient, combo):
    cols = list(combo)
    sub_scores = subset[cols].mean(axis=1)
    labels = subset['outcome']
    g_scores, p_scores = sub_scores[labels == 1], sub_scores[labels == 0]
    
    if len(g_scores) == 0 or len(p_scores) == 0: return False, None
    max_p, min_g = p_scores.max(), g_scores.min()
    
    if not (min_g > max_p): return False, None # No gap

    # Threshold (tau) calculation
    range_p, range_g = p_scores.max() - p_scores.min(), g_scores.max() - g_scores.min()
    gap = min_g - max_p
    tau = max_p + (gap * (range_p / (range_p + range_g + 1e-9)))

    test_val = test_patient[cols].mean()
    if test_val >= tau:
        denom = g_scores.max() - tau
        vote = min(1.0, (test_val - tau) / denom if denom != 0 else 1.0)
    else:
        denom = tau - p_scores.min()
        vote = max(-1.0, (test_val - tau) / denom if denom != 0 else -1.0)
    return True, vote


def main():
    # Load external data
    metadata = pd.read_excel(METADATA_PATH)
    outcomes = dict(zip(metadata['subject_id'], metadata['outcome']))
    
    # Check Ro dir and generate files if missing
    print("Checking Ro cache and generating missing files...")

    for sub_idx in outcomes.keys():
        save_path = os.path.join(CACHE_DIR, f'Ro_sub{sub_idx}.csv')

        resection = load_resection(f"{INPUT_DIR}/{sub_idx}_RESECTION.p")[sub_idx]
        
        # Only generate if the file doesn't exist
        if not os.path.exists(save_path):
            print(f"  > Generating Ro for Subject {sub_idx}...")
            _ = generate_ro_csv(sub_idx, set(resection.get(sub_idx, [])))
        else:
            print(f"  > Subject {sub_idx} already cached. Skipping.")

    # Load everything from the cache dir
    print("\nLoading all cached Ro files into memory...")
    ro_files = [f for f in os.listdir(CACHE_DIR) if f.startswith('Ro_sub') and f.endswith('.csv')]

    if not ro_files:
        raise FileNotFoundError(f"No Ro_sub files found in {CACHE_DIR}!")

    loaded_dfs = []
    for f in ro_files:
        # Extract subject ID from filename to ensure we only load subjects present in our outcome metadata
        sub_id = int(f.replace('Ro_sub', '').replace('.csv', ''))
        if sub_id in outcomes:
            temp_df = pd.read_csv(os.path.join(CACHE_DIR, f))
            loaded_dfs.append(temp_df)

    # Concatenate all subjects into the master matrix
    full_ro_df = pd.concat(loaded_dfs, ignore_index=True).set_index('subject').sort_index()

    # 3. SCALING & OUTCOME MAPPING
    all_cms_raw = [c for c in full_ro_df.columns if c not in ['subject', 'outcome']]

    # Apply the outcome labels from our dictionary
    full_ro_df['outcome'] = full_ro_df.index.map(outcomes)

    # Handle NaNs and Scaling
    full_ro_df[all_cms_raw] = full_ro_df[all_cms_raw].fillna(full_ro_df[all_cms_raw].median())
    scaler = RobustScaler()
    full_ro_df[all_cms_raw] = scaler.fit_transform(full_ro_df[all_cms_raw])

    # Min-Max normalization to [0, 1] for the voting logic
    full_ro_df[all_cms_raw] = (full_ro_df[all_cms_raw] - full_ro_df[all_cms_raw].min()) / \
                            (full_ro_df[all_cms_raw].max() - full_ro_df[all_cms_raw].min() + 1e-9)

    print(f"Successfully assembled matrix for {len(full_ro_df)} subjects.")

    # Select top CMs
    # (Using your pre-screening logic: low MAE)
    targets = full_ro_df['outcome'].values
    errors = {cm: np.mean(np.abs(targets - full_ro_df[cm].values)) for cm in all_cms_raw}
    all_cms = sorted(errors, key=errors.get)[:7] # Keep top 7
    
    # Run ensemble
    summary_results, detailed_votes = [], []
    
    for pid in full_ro_df.index:
        print(f"Evaluate Subject {pid}...")
        cohort = full_ro_df.drop(pid)
        test_patient = full_ro_df.loc[pid]
        
        votes, used_combos = [], set()
        # Search Tier 1-4
        for size in [1, 2, 3, 4]:
            if len(votes) >= CFG['n_votes_target']: break
            # Sample combinations
            cand = list(combinations(all_cms, size))
            RNG.shuffle(cand)
            for combo in cand:
                # Subsampling for variety
                g_idx = RNG.choice(cohort[cohort['outcome']==1].index, CFG['subset_good'], replace=False)
                p_idx = RNG.choice(cohort[cohort['outcome']==0].index, CFG['subset_poor'], replace=False)
                subset = cohort.loc[np.concatenate([g_idx, p_idx])]
                
                useful, v = evaluate_and_vote(subset, test_patient, combo)
                if useful:
                    votes.append(v)
                    detailed_votes.append({'subject': pid, 'vote_val': v, 'combo': "|".join(combo)})
                    if len(votes) >= CFG['n_votes_target']: break

        avg_score = np.mean(votes) if votes else 0.0
        summary_results.append({
            'Subject': pid, 'True_Outcome': outcomes[pid],
            'Mean_Score': round(avg_score, 4), 'Pred': 1 if avg_score >= 0 else 0
        })

    # Save readable tables
    pd.DataFrame(summary_results).to_csv(os.path.join(OUTPUT_DIR, "summary_outcomes.csv"), index=False)
    pd.DataFrame(detailed_votes).to_csv(os.path.join(OUTPUT_DIR, "detailed_votes.csv"), index=False)
    print("Files saved to output folder.")

if __name__ == "__main__":
    main()