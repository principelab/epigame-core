import os
import re
import pandas as pd
from pickle import load

output_folder = "data/output"
SCORE_DIR = os.path.join(output_folder, "game_scores")
SCORES_FILE = os.path.join(output_folder, "game_scores_table.csv")


def parse_subject_id(filename):
    # scores_sub11.p -> 11
    match = re.search(r"scores_sub(\d+)\.p", filename)
    if match:
        return int(match.group(1))
    return None


def build_scores_table(score_dir):
    rows = []

    for file in os.listdir(score_dir):

        if not file.startswith("scores_sub") or not file.endswith(".p"):
            continue

        subject_id = parse_subject_id(file)
        file_path = os.path.join(score_dir, file)

        print(f"Parsing subject {subject_id}")

        try:
            sub_result = load(open(file_path, "rb"))
        except Exception as e:
            print(f"Could not load {file}: {e}")
            continue

        # sub_result is dict with keys (cm, sigma)
        for (cm, sigma), result in sub_result.items():

            overlap = result.get("overlap_ratio", None)
            rows.append({
                "subject_id": subject_id,
                "CM": cm,
                "sigma": sigma,
                "overlap_ratio": overlap
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["subject_id", "CM", "sigma"])
    return df


if __name__ == "__main__":

    df = build_scores_table(SCORE_DIR)

    print("Total rows:", len(df))
    print(df.head())

    df.to_csv(SCORES_FILE, index=False)

    print(f"Saved table to {SCORES_FILE}")
