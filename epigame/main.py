from epigame.utils import Record, REc
import os, re
from pickle import load
from epigame.connectivity import preprocess_from_mat, run_connectivity_matrices
from epigame.cross_validation import run_classification_pipeline
from epigame.aggregate_scores import aggregate_cv_scores
from epigame.game import run_game
from epigame.outcome_prediction import run_outcome_prediction

#Paths
main_folder = "data/output"
input_folder = "data/input"
if not os.listdir(input_folder):
    raise RuntimeError(f"Input folder is empty!!! {input_folder}")

results_dir = os.path.join(main_folder, "results")
game_scores_dir = os.path.join(main_folder, "game_scores")
connectivity_dir = os.path.join(main_folder, "connectivity")
outcome_path = os.path.join(input_folder, "outcomes.xlsx")
RESECTION = load(open(os.path.join(input_folder, "RESECTION.p"), "rb"))
NODES = load(open(os.path.join(input_folder, "NODES.p"), "rb"))

#Required files
required_files = {
    "RESECTION.p": "RESECTION: dict {subid: [resected_node_ids]}",
    "NODES.p": "NODES: dict {subid: [all_node_ids]}",
    "outcomes.xlsx": "Excel with columns: 'subject_id', 'outcome' (1 or 0)"
}
missing = []
for fname, description in required_files.items():
    path = os.path.join(input_folder, fname)
    if not os.path.exists(path):
        missing.append(f"Missing '{fname}': {description}")
if missing:
    msg = "\n".join(["Required input files not found:"] + missing)
    raise FileNotFoundError(msg)

# Detect all subject IDs based on interictal files
subject_ids = sorted({
    int(re.match(r"(\d+)_interictal\.mat", f).group(1))
    for f in os.listdir(input_folder)
    if re.match(r"\d+_interictal\.mat", f)
})
print(f"Found {len(subject_ids)} subjects: {subject_ids}")

#Step 1: Connectivity analysis
for subject_id in subject_ids:
    print(f"\nComputing connectivity for subject {subject_id}")

    interictal_path = os.path.join(input_folder, f"{subject_id}_interictal.mat")
    preictal_path = os.path.join(input_folder, f"{subject_id}_preictal.mat")

    prep = preprocess_from_mat(interictal_path, preictal_path, fs=500, band=None)
    run_connectivity_matrices(prep, subject_id, bands=None, output_dir=connectivity_dir)

    for band in [(0,4),(4,8),(8,13),(13,30),(30,70),(70,150)]:
        prep_band = preprocess_from_mat(interictal_path, preictal_path, fs=500, band=band)
        run_connectivity_matrices(prep_band, subject_id, bands=band, output_dir=connectivity_dir)

#Step 2: Cross-validation per CM
connectivity_measures = ["PAC", "SCR", "SCI", "PLV", "PLI", "CC"]

for subject_id in subject_ids:
    for bands in [None, (0,4),(4,8),(8,13),(13,30),(30,70),(70,150)]:
        cm_suffix = "" if bands is None else f"-{bands[0]}-{bands[1]}"
        ext = f"{subject_id}-{bands[0]}-{bands[1]}" if bands else f"{subject_id}"

        for measure in connectivity_measures:
            file = os.path.join(connectivity_dir, f"{subject_id}-{measure}{cm_suffix}.prep")
            if not os.path.exists(file):
                print(f"Skipping missing file: {file}")
                continue

            cm_struct = REc.load(file).data
            run_classification_pipeline(
                cm_struct=cm_struct,
                subject_id=subject_id,
                measure=measure,
                bands=bands,
                output_dir=results_dir
            )

#Step 3: Aggregate CVS scores
cvs_csv = os.path.join(main_folder, "cvs_pairs.csv")
aggregate_cv_scores(
    result_dir=results_dir,
    subject_ids=subject_ids,
    output_csv=cvs_csv
)

#Step 4: Game simulation
for subject_id in subject_ids:
    print(f"\nRunning game simulation for subject {subject_id}")
    run_game(
        subject_id=subject_id,
        main_folder=main_folder,
        output_dir=game_scores_dir,
        RESECTION=RESECTION,
        NODES=NODES,
        max_sigma=4
    )

subject_ids = []
pattern = re.compile(r"^scores_sub(\d+)\.p$")
for filename in os.listdir(game_scores_dir):
    match = pattern.match(filename)
    if match:
        subject_id = int(match.group(1))
        subject_ids.append(subject_id)

#Step 5: Outcome prediction
print(f"\nRunning outcome prediction")
run_outcome_prediction(
    score_dir=game_scores_dir,
    subject_ids=subject_ids,
    sigma=4,
    max_n_cms=1,
    outcome_path=outcome_path
)
