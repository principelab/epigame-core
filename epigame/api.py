import os
from pickle import load
from epigame.connectivity import preprocess_from_mat, run_connectivity_matrices
from epigame.game import run_game


main_dir = "data/output"
game_scores_dir = os.path.join(main_dir, "game_scores")
connectivity_dir = os.path.join(main_dir, "connectivity")

input_dir = "data/input"
RESECTION = load(open(os.path.join(input_dir, "RESECTION.p"), "rb"))
NODES = load(open(os.path.join(input_dir, "NODES.p"), "rb"))


def compute_connectivity_for_patient(
    subject_id,
    interictal_path,
    preictal_path,
    connectivity_dir=connectivity_dir,
    fs=500,
    bands=[None,(0,4),(4,8),(8,13),(13,30),(30,70),(70,150)]):
    """
    Generate Epigame connectivity dependencies for a single patient.

    This function takes raw interictal and preictal SEEG recordings for one
    patient and computes the connectivity matrices required by Epigame.
    Connectivity is computed for multiple frequency bands and saved to disk.

    Parameters
    ----------
    subject_id : int
        Unique identifier of the patient.
    interictal_path : str
        Path to the patient's interictal .mat file.
    preictal_path : str
        Path to the patient's preictal .mat file.
    connectivity_dir : str
        Directory where connectivity files will be saved.
    fs : int, optional
        Sampling frequency of the SEEG recordings (default: 500 Hz).
    bands : list of tuple or None, optional
        Frequency bands for which connectivity is computed.
        Use None to compute broadband connectivity.

    Returns
    -------
    None
        Connectivity files are written to disk and used as input for the
        Epigame simulation step.
    """
    for band in bands:
        prep = preprocess_from_mat(
            interictal_path,
            preictal_path,
            fs=fs,
            band=band
        )
        run_connectivity_matrices(
            prep,
            subject_id,
            bands=band,
            output_dir=connectivity_dir
        )


def run_game_for_patient(
    subject_id,
    main_dir=main_dir,
    game_scores_dir=game_scores_dir,
    RESECTION=RESECTION,
    NODES=NODES,
    max_sigma=4):
    """
    Run the Epigame simulation for a single patient.

    This function executes the Epigame model using precomputed
    connectivity data and returns the resulting Epigame score for the patient.

    Parameters
    ----------
    subject_id : int
        Unique identifier of the patient.
    main_dir : str
        Root output directory used by Epigame.
    game_scores_dir : str
        Directory where Epigame scores are stored.
    RESECTION : dict
        Dictionary mapping subject IDs to resected node indices.
    NODES : dict
        Dictionary mapping subject IDs to all node indices.
    max_sigma : int, optional
        Maximum sigma parameter used in the Epigame model (default: 4).

    Returns
    -------
    score : object
        Epigame score for the patient, loaded from disk.

    Notes
    -----
    The returned score is a continuous measure and represents 
    the model output for a single patient.
    """
    run_game(
        subject_id=subject_id,
        main_folder=main_dir,
        output_dir=game_scores_dir,
        RESECTION=RESECTION,
        NODES=NODES,
        max_sigma=max_sigma
    )
    score_file = os.path.join(game_scores_dir, f"scores_sub{subject_id}.p")
    score = load(open(score_file, "rb"))
    return score


def epigame_predict_from_mat(
    subject_id,
    interictal_path,
    preictal_path,
    main_dir=main_dir,
    connectivity_dir=connectivity_dir,
    game_scores_dir=game_scores_dir,
    RESECTION=RESECTION,
    NODES=NODES,
    fs=500,
    max_sigma=4
):
    """
    Run Epigame for a single patient and return the game score.
    
    Returns
    -------
    score : object
        Epigame score for the patient.

    Notes
    -----
    This function performs patient-level inference only. It does not use
    outcome labels, does not apply thresholds, and does not compute
    performance metrics. Outcome prediction (e.g., ROC/AUC) should be
    performed after aggregating scores across patients.
    """

    # 1. Connectivity
    compute_connectivity_for_patient(
        subject_id,
        interictal_path,
        preictal_path,
        connectivity_dir,
        fs=fs
    )

    # 2. Game
    score = run_game_for_patient(
        subject_id,
        main_dir,
        game_scores_dir,
        RESECTION,
        NODES,
        max_sigma=max_sigma
    )

    return score
