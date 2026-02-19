import os
from pickle import load
from epigame.utils import REc
from epigame.connectivity import preprocess_from_mat, run_connectivity_matrices
from epigame.cross_validation import run_classification_pipeline
from epigame.aggregate_scores import aggregate_cv_scores
from epigame.game import run_game


main_dir = "data/output"
game_scores_dir = os.path.join(main_dir, "game_scores")
connectivity_dir = os.path.join(main_dir, "connectivity")
results_dir = os.path.join(main_dir, "results")
csv_file = os.path.join(main_dir, "cvs_pairs.csv")

bands=[(1,4),(4,8),(8,13),(13,30),(30,70),(70,150)]

def compute_connectivity_for_subject(
    subject_id,
    interictal_path,
    preictal_path,
    connectivity_dir=connectivity_dir,
    fs=500,
    connectivity_measure="PAC",
    bands=bands):
    """
    Generate Epigame connectivity dependencies for a single subject.

    This function takes raw interictal and preictal SEEG recordings for one
    subject and computes the connectivity matrices required by Epigame.
    Connectivity is computed for multiple frequency bands and saved to disk.

    Parameters
    ----------
    subject_id : int
        Unique identifier of the subject.
    interictal_path : str
        Path to the subject's interictal .mat file.
    preictal_path : str
        Path to the subject's preictal .mat file.
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
    bands = [None] if connectivity_measure == "PAC" else bands
    for band in bands:
        prep = preprocess_from_mat(
            interictal_path,
            preictal_path,
            target_fs=fs,
            band=band
        )
    
        run_connectivity_matrices(
            prep,
            subject_id,
            connectivity_measure=connectivity_measure,
            band=band,
            output_dir=connectivity_dir
        )

def run_cv_for_subject(
    subject_id,
    connectivity_dir,
    results_dir,
    connectivity_measure="PAC",
    freq_bands=bands
):
    """
    Run cross-validation classification for all connectivity measures
    for a single subject.
    
    Parameters
    ----------
    subject_id : int
        subject ID to analyze.
    connectivity_dir : str
        Folder where preprocessed connectivity .prep files are stored.
    results_dir : str
        Folder where classification results will be saved.
    freq_bands : list of tuple or None
        Frequency bands to analyze. None for broadband.
    """
    if connectivity_measure == "PAC":
        cm_suffix = "" # PAC does not use frequency bands

        prep_file = os.path.join(connectivity_dir, f"{subject_id}-{connectivity_measure}{cm_suffix}.prep")

        cm_struct = REc.load(prep_file).data
        run_classification_pipeline(
            cm_struct=cm_struct,
            subject_id=subject_id,
            measure=connectivity_measure,
            bands=None,
            output_dir=results_dir
        )
    else:
        for band in freq_bands:
            cm_suffix = "" if band is None else f"-{band[0]}-{band[1]}"

            prep_file = os.path.join(connectivity_dir, f"{subject_id}-{connectivity_measure}{cm_suffix}.prep")
            if not os.path.exists(prep_file):
                print(f"Skipping missing file: {prep_file}")
                continue

            cm_struct = REc.load(prep_file).data
            run_classification_pipeline(
                cm_struct=cm_struct,
                subject_id=subject_id,
                measure=connectivity_measure,
                bands=band,
                output_dir=results_dir
            )

def run_game_for_subject(
    subject_id,
    main_dir=main_dir,
    game_scores_dir=game_scores_dir,
    max_sigma=4):
    """
    Run the Epigame simulation for a single subject.

    This function executes the Epigame model using precomputed
    connectivity data and returns the resulting Epigame score for the subject.

    Parameters
    ----------
    subject_id : int
        Unique identifier of the subject.
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
        Epigame score for the subject, loaded from disk.

    Notes
    -----
    The returned score is a continuous measure and represents 
    the model output for a single subject.
    """
    run_game(
        subject_id=subject_id,
        main_folder=main_dir,
        output_dir=game_scores_dir,
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
    fs=500,
    max_sigma=4,
    connectivity_measures = ["PAC", "SCR", "SCI", "PLV", "PLI", "CC"],
    bands=[(1,4),(4,8),(8,13),(13,30),(30,70),(70,150)]
):
    """
    Run Epigame for a single subject and return the game score.
    
    Returns
    -------
    score : object
        Epigame score for the subject.

    Notes
    -----
    This function performs subject-level inference only. It does not use
    outcome labels, does not apply thresholds, and does not compute
    performance metrics. Outcome prediction (e.g., ROC/AUC) should be
    performed after aggregating scores across subjects.
    """

    # 1. Connectivity
    for measure in connectivity_measures:

        compute_connectivity_for_subject(
            subject_id,
            interictal_path,
            preictal_path,
            connectivity_dir,
            fs=fs,
            connectivity_measure=measure,
            bands=bands
        )

        # 2. Cross-validation (Connectivity change computation)
        run_cv_for_subject(
        subject_id,
        connectivity_dir,
        results_dir,
        connectivity_measure=measure,
        freq_bands=bands
        )

    # 3. Aggregate cross-validation (CV) scores
    aggregate_cv_scores(
        result_dir=results_dir,
        subject_ids=[subject_id],
        output_csv=csv_file,
        connectivity_measures=connectivity_measures,
        freq_bands=bands
    )

    # 4. Game
    score = run_game_for_subject(
        subject_id,
        main_dir,
        game_scores_dir,
        max_sigma=max_sigma
    )

    return score
