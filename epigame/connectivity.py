from epigame.utils import struct, REc
import os
import pickle
import numpy as np
from joblib import Parallel, delayed
from scipy.io import loadmat
from scipy.signal import hilbert, csd, butter, resample
from scipy.signal import iirnotch, butter, filtfilt

def notch(data, fs, freq=50.0, Q=30.0):
    """
    Apply a notch filter to remove powerline interference.

    Args:
        data (ndarray): EEG signal of shape (channels, time)
        fs (float): Sampling frequency in Hz
        freq (float): Notch frequency (default: 50.0 Hz)
        Q (float): Quality factor (default: 30.0)

    Returns:
        ndarray: Notch-filtered signal (same shape as input)
    """
    b, a = iirnotch(w0=freq, Q=Q, fs=fs)
    return filtfilt(b, a, data, axis=1)


def bandpass(data, band, fs=500.0, order=4):
    """
    Apply a bandpass filter to EEG data.

    Args:
        data (ndarray): EEG signal of shape (channels, time)
        band (tuple): Frequency range (low, high) in Hz
        fs (float): Sampling frequency
        order (int): Filter order

    Returns:
        ndarray: Bandpass-filtered signal
    """
    low, high = band
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)

# Sanity check: sampling rate after resampling
def check_fs(original_n_samples, original_fs, new_n_samples, target_fs, name):
    orig_duration = original_n_samples / original_fs
    new_duration = new_n_samples / target_fs
    assert np.isclose(orig_duration, new_duration, rtol=1e-3), (
        f"{name}: duration mismatch after resampling "
        f"(orig={orig_duration:.3f}s, new={new_duration:.3f}s)"
    )
    print(f"{name}: resampled to {target_fs} Hz "
          f"({original_n_samples} → {new_n_samples} samples)")

def phaselock(signal1, signal2):
    """Computes the phase locking value between two notch-filtered signals.
    
    Args:
        signal1 (array): Timecourse recorded from a first node.
        signal2 (array): Timecourse recorded from a second node.

    Returns:
        float: Phase locking value.
    """
    sig1_hil = hilbert(signal1)                          
    sig2_hil = hilbert(signal2)
    phase1 = np.angle(sig1_hil)                           
    phase2 = np.angle(sig2_hil)
    phase_dif = phase1-phase2                             
    plv = abs(np.mean(np.exp(complex(0,1)*phase_dif)))    
    return plv

def phaselag(signal1, signal2):
    """Computes the phase lag index between two signals.
    
    Args:
        signal1 (array): Timecourse recorded from a first node.
        signal2 (array): Timecourse recorded from a second node.

    Returns:
        float: Phase lag index.
    """
    sig1_hil = hilbert(signal1)                 
    sig2_hil = hilbert(signal2)
    phase1 = np.angle(sig1_hil)                 
    phase2 = np.angle(sig2_hil)
    phase_dif = phase1-phase2                   
    pli = abs(np.mean(np.sign(phase_dif)))     
    return pli

def spectral_coherence(signal1, signal2, fs=500, imag=False):
    """Computes the spectral coherence between two signals.

    Args:
        signal1 (array): Timecourse recorded from a first node.
        signal2 (array): Timecourse recorded from a second node.
        fs (int): Sampling frequency.
        imag (bool): If True, computed the imaginary part of spectral coherence, if False computes the real part. Defaults to False.

    Returns:
        float: Spectral coherence.
    """
    Pxy = csd(signal1,signal2,fs=fs, scaling='spectrum')[1] 
    Pxx = csd(signal1,signal1,fs=fs, scaling='spectrum')[1]
    Pyy = csd(signal2,signal2,fs=fs, scaling='spectrum')[1]
    if imag: return np.average((Pxy.imag)**2/(Pxx*Pyy))     
    elif not imag: return np.average(abs(Pxy)**2/(Pxx*Pyy))

def cross_correlation(signal1, signal2):
    """Computes the cross correlation between two signals.
    
    Args:
        signal1 (array): Timecourse recorded from a first node.
        signal2 (array): Timecourse recorded from a second node.

    Returns:
        float: Cross correlation.
    """
    return np.correlate(signal1, signal2, mode="valid")

def PAC(signal1, signal2, fs=500):
    """Computes low frequency phase - high frequency amplitude phase coupling between two signals.
    Low frequency = 1–4 Hz; High frequency = 30–70 Hz

    Args:
        signal1 (array): Timecourse recorded from the first node.
        signal2 (array): Timecourse recorded from the second node.
        fs (int): Sampling frequency.

    Returns:
        float: Phase-amplitude coupling value (PAC).
    """
    low = bandpass(signal1[np.newaxis, :], band=(1, 4), fs=fs)[0]
    high = bandpass(signal2[np.newaxis, :], band=(30, 70), fs=fs)[0]

    low_phase = np.unwrap(np.angle(hilbert(low)))
    high_amp_envelope = np.abs(hilbert(high))
    high_amp_phase = np.unwrap(np.angle(hilbert(high_amp_envelope)))

    phase_diff = low_phase - high_amp_phase
    pac = np.abs(np.mean(np.exp(1j * phase_diff)))
    return pac

def analyze_epoch(epoch, method, dtail=True, **opts):
    mat = np.zeros((len(epoch), len(epoch)))
    nid, pairs = list(range(len(epoch))), []

    for a in range(len(nid)):
        for b in (range(len(nid)) if dtail else range(a, len(nid))):
            pairs.append((a, b))

    conn_per_pair = Parallel(n_jobs=-1)(
        delayed(method)(epoch[pair[0]], epoch[pair[1]], **opts) for pair in pairs
    )

    for pair_idx, pair in enumerate(pairs):
        mat[pair[0], pair[1]] = conn_per_pair[pair_idx]

    return mat

def connectivity_analysis(epochs, method, dtail=True, **opts):
    print('Connectivity measure:', method.__name__)

    if "bands" in opts and opts["bands"] is not None:
        print(f"Frequency band: {opts['bands']}")

    return Parallel(n_jobs=-1)(
        delayed(analyze_epoch)(e, method, dtail, **opts) for e in epochs
    )

def run_connectivity_matrices(epochs, subject_id, bands=None, output_dir="data/output/"):

    connectivity_measures = ["PAC"] if bands is None else ["SCR", "SCI", "PLV", "PLI", "CC"]

    for measure in connectivity_measures:

        print(f"Running measure: {measure}")

        cm = struct(y=epochs.y, i=epochs.i, nodes=epochs.nodes)

        if measure == "SCR":
            cm._set(X = connectivity_analysis(epochs.x_prep, spectral_coherence, fs=500, imag=False, bands=bands))
        elif measure == "SCI":
            cm._set(X = connectivity_analysis(epochs.x_prep, spectral_coherence, fs=500, imag=True, bands=bands))
        elif measure == "PLV":
            cm._set(X = connectivity_analysis(epochs.x_prep, phaselock, bands=bands))
        elif measure == "PLI":
            cm._set(X = connectivity_analysis(epochs.x_prep, phaselag, bands=bands))
        elif measure == "CC":
            cm._set(X = connectivity_analysis(epochs.x_prep, cross_correlation, bands=bands))
        elif measure == "PAC":
            cm._set(X = connectivity_analysis(epochs.x_prep, PAC, fs=500))

        os.makedirs(output_dir, exist_ok=True)
        if bands is None: suffix = f"{subject_id}-{measure}.prep"
        else:
            # replace dot with underscore in band string (e.g., 0.1-4 -> 0_1-4)
            band_str = f"{bands[0]}-{bands[1]}".replace('.', '_')
            suffix = f"{subject_id}-{measure}-{band_str}.prep"
        REc(cm).save(os.path.join(output_dir, suffix))

def sliding_window_epochs(filtered_data, fs, span_ms=1000, step_ms=125):
    """Split filtered data into overlapping epochs (channels × samples)."""
    span_samples = int((span_ms / 1000) * fs)
    step_samples = int((step_ms / 1000) * fs)
    total_samples = filtered_data.shape[1]

    n_epochs = int((total_samples / step_samples)-1)
    print(f"Creating {n_epochs} overlapping epochs")

    epochs = [
        filtered_data[:, i*step_samples : i*step_samples + span_samples]
        for i in range(n_epochs)
    ]
    return epochs

import numpy as np
from scipy.io import loadmat

def match_channels(eeg_interictal, labels_interictal,
                   eeg_preictal, labels_preictal):
    # Ensure the file is not transposed
    assert eeg_interictal.ndim == 2, "Interictal EEG must be 2D (samples × channels)"
    assert eeg_preictal.ndim == 2, "Preictal EEG must be 2D (samples × channels)"
    assert len(labels_interictal) == eeg_interictal.shape[1]
    assert len(labels_preictal) == eeg_preictal.shape[1]

    # Find common channels
    set_interictal = set(labels_interictal)
    set_preictal = set(labels_preictal)
    common_labels = sorted(set_interictal.intersection(set_preictal))

    if not common_labels:
        raise ValueError("No common channels found between interictal and preictal files.")

    # Map indices
    interictal_indices = [labels_interictal.index(lbl) for lbl in common_labels]
    preictal_indices = [labels_preictal.index(lbl) for lbl in common_labels]

    # Subset and return in common order
    eeg_interictal_matched = eeg_interictal[:, interictal_indices]
    eeg_preictal_matched = eeg_preictal[:, preictal_indices]

    return eeg_interictal_matched, eeg_preictal_matched, common_labels


def save_nodes_pickle(n_nodes, subject_id, input_dir="data/input/"):
    """
    Save node labels as a pickle file in the external dictionary format.

    Args:
        node_labels (list of str): list of channel names for this subject
        subject_id (int or str): subject identifier
        input_dir (str): directory to save the pickle file
    """
    nodes_dict = {subject_id: list(range(n_nodes))}
    filepath = os.path.join(input_dir, f"{subject_id}_NODES.p")
    with open(filepath, "wb") as f:
        pickle.dump(nodes_dict, f)
    print(f"Nodes (indices) file saved to {filepath}")


def save_resection_pickle(resection_indices, subject_id, input_dir="data/input/"):
    """
    Save resection info as a pickle file in the external dictionary format.

    Args:
        resection_indices (list of int): list of resected channel indices for this subject
        subject_id (int or str): subject identifier
        input_dir (str): directory to save the pickle file
    """
    resection_dict = {subject_id: resection_indices}
    filepath = os.path.join(input_dir, f"{subject_id}_RESECTION.p")
    with open(filepath, "wb") as f:
        pickle.dump(resection_dict, f)
    print(f"Resection (indices) file saved to {filepath}")


def load_mat_wrapper(mat_path):
    mat = loadmat(mat_path)
    assert 'sz_data' in mat, "Missing 'sz_data' key in .mat file"
    sz = mat['sz_data']

    assert sz.ndim == 2 and sz.shape[1] >= 6, f"sz_data must be (1, >=6), got {sz.shape}"

    signal = sz[0, 0]
    labels = sz[0, 1]
    fs = sz[0, 2]
    soz = sz[0, 4]
    resection = sz[0, 5]

    # Convert fs to scalar
    fs = float(np.squeeze(fs))

    # labels: MATLAB cell array → list of strings
    labels = [str(l[0]) if isinstance(l, np.ndarray) else str(l) for l in labels.squeeze()]

    def normalize_channel_list(x):
        """
        Convert MATLAB nested arrays/cells to flat list of strings.
        Example: array(['ROF5-6'], dtype='<U6') -> 'ROF5-6'
        """
        if x is None or len(np.atleast_1d(x)) == 0:
            return []

        flat_list = []
        for i in np.atleast_1d(x):
            # If nested ndarray, extract first element
            while isinstance(i, np.ndarray) and i.size == 1:
                i = i[0]
            flat_list.append(str(i))
        return flat_list

    soz = normalize_channel_list(soz)
    resection = normalize_channel_list(resection)

    assert signal.ndim == 2, "Signal must be 2D (samples × channels)"
    assert isinstance(fs, float), "fs must be scalar float"

    return {
        "signal": signal,
        "labels": labels,
        "fs": fs,
        "soz": soz,
        "resection": resection
    }


def preprocess_from_mat(interictal_path, preictal_path, target_fs=500, band=None):
    # Constants
    span, step = 1000, 500  # in ms
    min_woi_duration = 60000  # in ms

    subject_id = int(os.path.basename(preictal_path).split("_")[0])

    # Load data
    pre = load_mat_wrapper(preictal_path)
    inter = load_mat_wrapper(interictal_path)

    eeg_preictal = pre["signal"]
    eeg_interictal = inter["signal"]

    fs_preictal = pre["fs"]
    fs_interictal = inter["fs"]

    labels = pre["labels"]
    labels_interictal = inter["labels"]

    resection = pre["resection"]
    resection_interictal = inter["resection"]
    soz = pre["soz"]
    soz_interictal = inter["soz"]
    assert soz == soz_interictal, "SOZ labels does not match between interictal and preictal files."
    assert resection == resection_interictal, "Resection labels does not match between interictal and preictal files."

    # Resample to target_fs if needed
    if fs_preictal != target_fs:
        n_samples = int(eeg_preictal.shape[0] * target_fs / fs_preictal)
        eeg_preictal = resample(eeg_preictal, n_samples, axis=0)

    if fs_interictal != target_fs:
        n_samples = int(eeg_interictal.shape[0] * target_fs / fs_interictal)
        eeg_interictal = resample(eeg_interictal, n_samples, axis=0)

    # Preictal
    check_fs(
        original_n_samples=pre["signal"].shape[0],
        original_fs=fs_preictal,
        new_n_samples=eeg_preictal.shape[0],
        target_fs=target_fs,
        name="Preictal"
    )

    # Interictal
    check_fs(
        original_n_samples=inter["signal"].shape[0],
        original_fs=fs_interictal,
        new_n_samples=eeg_interictal.shape[0],
        target_fs=target_fs,
        name="Interictal"
    )

    # Align and trim channels
    interictal, preictal, common_labels = match_channels(eeg_interictal, labels_interictal, eeg_preictal, labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(common_labels)}

    missing = set(resection) - set(label_to_idx)
    if missing:
        raise ValueError(f"Resection labels not found in nodes: {missing}")

    resection_idx = [label_to_idx[lbl] for lbl in resection if lbl in label_to_idx]

    # Save nodes and resection pickles so mat files don't have to be loaded again in game step
    save_nodes_pickle(n_nodes=len(common_labels), subject_id=subject_id)
    save_resection_pickle(resection_indices=resection_idx, subject_id=subject_id)

    # Transpose to (channels, samples)
    interictal = interictal.T
    preictal = preictal.T

    # Filtering
    interictal = notch(interictal, target_fs)
    preictal = notch(preictal, target_fs)

    if band is not None:
        interictal = bandpass(interictal, band, fs=target_fs)
        preictal = bandpass(preictal, band, fs=target_fs)

    # Create overlapping epochs
    interictal_epochs = sliding_window_epochs(interictal, target_fs, span, step)
    preictal_epochs = sliding_window_epochs(preictal, target_fs, span, step)

    # Ensure minimum epoch number
    min_epochs = int(min_woi_duration / step) - 1
    interictal_epochs = interictal_epochs[:min_epochs]
    preictal_epochs = preictal_epochs[:min_epochs]

    print(f"Number of epochs: {len(interictal_epochs)}")

    # Labels and struct prep
    x = preictal_epochs + interictal_epochs
    y = [1]*len(preictal_epochs) + [0]*len(interictal_epochs)
    i = list(range(len(x)))
    node_idx = list(range(len(common_labels)))

    prep = struct(y=np.array(y), i=np.array(i), x_prep=x, nodes=node_idx, resection=resection_idx)
    return prep
