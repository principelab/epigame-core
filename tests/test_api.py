import os
import numpy as np
import scipy.io as sio
from epigame.api import epigame_predict_from_mat
from epigame.connectivity import load_mat_wrapper

# Paths
input_folder = "data/input/Test_data_02Feb26"
main_folder = "data/output"
connectivity_dir = os.path.join(main_folder, "connectivity")
game_scores_dir = os.path.join(main_folder, "game_scores")

subject_id = 1  # change to a valid ID

interictal_path = os.path.join(input_folder, f"{subject_id}_interictal.mat")
preictal_path = os.path.join(input_folder, f"{subject_id}_preictal.mat")

# Sanity checks
assert os.path.exists(interictal_path), "Missing interictal file"
assert os.path.exists(preictal_path), "Missing preictal file"

# Run wrapper 
data = load_mat_wrapper(interictal_path)

# Print summary 
print("\nMAT file summary ")
print(f"Signal shape: {data['signal'].shape}")
print(f"Number of channels / labels: {len(data['labels'])}")
print(f"Labels: {data['labels']}")
print(f"Sampling frequency: {data['fs']}")
print(f"SOZ channels: {data['soz']}")
print(f"Resection channels: {data['resection']}")

# Optional: basic sanity checks 
assert isinstance(data['signal'], np.ndarray) and data['signal'].ndim == 2
assert isinstance(data['labels'], list) and all(isinstance(l, str) for l in data['labels'])
assert isinstance(data['fs'], float)
assert isinstance(data['soz'], list)
assert isinstance(data['resection'], list)

# Run Epigame API
score = epigame_predict_from_mat(
    subject_id=subject_id,
    interictal_path=interictal_path,
    preictal_path=preictal_path,
    main_dir=main_folder,
    connectivity_dir=connectivity_dir,
    game_scores_dir=game_scores_dir,
    fs=500,
    max_sigma=4,
    connectivity_measures = ["PAC"],
    bands=[(1,4),(4,8),(8,13),(13,30),(30,70),(70,150)]
)

print("\nEpigame run complete.")
print("Score keys:", list(score.keys()) if isinstance(score, dict) else type(score))
