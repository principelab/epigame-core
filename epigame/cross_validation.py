import os
import numpy as np
from itertools import combinations
from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import random
from epigame.utils import struct, REc, rec

random_state = 31  # Default random state for reproducibility
random.seed(random_state)
np.random.seed(random_state)

def classify_epochs(cm_struct, node_group, kratio=.1, random_state=31, **mopts):
    """Classify epochs based on connectivity submatrix of a node pair.
    Args:
        cm_struct (rec): Preprocessed data - connectivity matrices per epoch.
        nodes (set): Node group for which connectivity change is quantified. 
        kratio (float): Ratio of epochs considered in a fold. Defaults to 0.1, which is ~10s of data if epoch span is 1s.
        random_state (int): KFold argument; Controls randomness of each fold. Defaults to 31.

    Returns:
        ndarray of float: Array of scores for each fold.
    """
    X = np.array([
        np.array((rec.read(x).include(*node_group).T).include(*node_group)).flatten()
        for x in cm_struct.X
    ])
    Y = cm_struct.y
    model, scaler, k = SVC, StandardScaler, int(round(len(Y) * kratio))
    if random_state is None: random_state = random.randint(0xFFFFFFFF)
    clf = Pipeline([('scaler', scaler()), ('model', model(**mopts))])
    cv = KFold(k, shuffle=True, random_state=random_state)
    scores = cross_val_score(clf, X, Y, cv=cv)
    return scores

def evaluate_nodes(pair, results):
    tag = f"{pair[0]}<->{pair[1]}"
    return (pair, tag, results)

def run_classification_pipeline(cm_struct, subject_id, measure, bands=None,
                                output_dir="data/output/results/", random_state=random_state):
    """Runs SVM-based classification across all node pairs on one connectivity measure."""

    os.makedirs(output_dir, exist_ok=True)
    if bands is None:
        suffix = f"{subject_id}-{measure}"
    else:
        band_str = f"{bands[0]}-{bands[1]}".replace('.', '_')
        suffix = f"{subject_id}-{measure}-{band_str}"

    filename = f"{suffix}.res"
    save_path = os.path.join(output_dir, filename)

    # skip if already computed
    if os.path.exists(save_path):
        print(f"Skipping classification (already exists): {filename}")
        return None

    print(f"Running classification for {measure} | Band: {bands}")

    node_ids = cm_struct.nodes
    node_pairs = list(combinations(node_ids, 2))

    results = Parallel(n_jobs=-1, verbose=True)(
        delayed(evaluate_nodes)(
            pair, classify_epochs(cm_struct, pair, random_state=random_state)
        ) for pair in node_pairs
    )
    result_struct = struct(nodes=node_ids, pairs=results)
    REc_result = REc(result_struct)

    # save
    REc_result.save(save_path)
    print(f"Saved result to: {filename}")

    return result_struct
