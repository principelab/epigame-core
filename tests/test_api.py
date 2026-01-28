import os
from pickle import load
from epigame.api import epigame_predict_from_mat

# Paths (match the repo structure)
input_folder = "data/input"
main_folder = "data/output"
connectivity_dir = os.path.join(main_folder, "connectivity")
game_scores_dir = os.path.join(main_folder, "game_scores")

# Choose a subject to test
subject_id = 1   # change to a valid ID you know exists

interictal_path = os.path.join(input_folder, f"{subject_id}_interictal.mat")
preictal_path = os.path.join(input_folder, f"{subject_id}_preictal.mat")

# Sanity checks
assert os.path.exists(interictal_path), "Missing interictal file"
assert os.path.exists(preictal_path), "Missing preictal file"

# Run Epigame API
score = epigame_predict_from_mat(
    subject_id=subject_id,
    interictal_path=interictal_path,
    preictal_path=preictal_path,
    main_dir=main_folder,
    connectivity_dir=connectivity_dir,
    game_scores_dir=game_scores_dir
)

