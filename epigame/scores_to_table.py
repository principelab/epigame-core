import os
import re
import pandas as pd
import numpy as np
from pickle import load
from itertools import combinations
from collections import defaultdict

output_folder = "data/output"
SCORE_DIR = os.path.join(output_folder, "game_scores")
SCORES_FILE = os.path.join(output_folder, "game_scores_combinations_up_to_4.csv")

MAX_N_CM = 4
TARGET_SIGMA = 4


def parse_subject_id(filename):
    match = re.search(r"scores_sub(\d+)\.p", filename)
    if match:
        return int(match.group(1))
    return None


def get_score(sub_result, cm, target_sigma):
    """
    Implements recursive sigma fallback logic.
    """
    sigma = target_sigma
    score = 0

    while sigma >= 1:
        entry = sub_result.get((cm, sigma))
        if entry:
            score = entry.get("overlap_ratio", 0)
            if score != 0:
                break
        sigma -= 1

    return score


def build_scores_table(score_dir, target_sigma=4, max_n_cm=4):

    rows = []

    for file in os.listdir(score_dir):

        if not file.startswith("scores_sub") or not file.endswith(".p"):
            continue

        subject_id = parse_subject_id(file)
        file_path = os.path.join(score_dir, file)

        print(f"Processing subject {subject_id}")

        try:
            sub_result = load(open(file_path, "rb"))
        except Exception as e:
            print(f"Could not load {file}: {e}")
            continue

        # Get all CMs available for this subject at any sigma
        all_cms = sorted(set(cm for (cm, s) in sub_result.keys()))

        # Compute best available score per CM using sigma fallback
        cm_scores = {}
        for cm in all_cms:
            score = get_score(sub_result, cm, target_sigma)
            if score != 0:
                cm_scores[cm] = score
        # If no valid scores, skip subject
        if not cm_scores:
            continue

        #  Generate combinations 1-4
        cms_available = sorted(cm_scores.keys())

        for k in range(1, max_n_cm + 1):
            for subset in combinations(cms_available, k):

                scores = [cm_scores[cm] for cm in subset]
                mean_score = np.mean(scores)
                rows.append({
                    "subject_id": subject_id,
                    "CM_combination": "|".join(subset),
                    "n_CM": k,
                    "mean_score": mean_score
                })
    df = pd.DataFrame(rows)
    df = df.sort_values(["subject_id", "n_CM", "CM_combination"])

    SCORES_FILE = ("data/output/game_scores_combinations_up_to_4.csv")
    df.to_csv(SCORES_FILE, index=False)
    print(f"Saved table to {SCORES_FILE}")
    return df


if __name__ == "__main__":

    df = build_scores_table(
        SCORE_DIR,
        target_sigma=TARGET_SIGMA,
        max_n_cm=MAX_N_CM
    )

    print("Total rows:", len(df))
    print(df.head())
