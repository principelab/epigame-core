from epigame.utils import struct, REc
import os
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
            cm._set(X = connectivity_analysis(epochs.x_prep, PAC, fs=500, bands=bands))

        os.makedirs(output_dir, exist_ok=True)
        suffix = f"{subject_id}-{measure}.prep" if bands is None else f"{subject_id}-{measure}-{bands}.prep"
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

def match_channels(interictal_raw, preictal_raw):
    # Extract EEG and labels
    eeg_interictal = interictal_raw[0]
    labels_interictal = [str(lbl[0][0]) for lbl in interictal_raw[1]]

    eeg_preictal = preictal_raw[0]
    labels_preictal = [str(lbl[0][0]) for lbl in preictal_raw[1]]

    # Find common channels
    set_interictal = set(labels_interictal)
    set_preictal = set(labels_preictal)
    common_labels = sorted(set_interictal.intersection(set_preictal))

    if not common_labels:
        raise ValueError("No common channels found between interictal and preictal data.")

    # Map indices
    interictal_indices = [labels_interictal.index(lbl) for lbl in common_labels]
    preictal_indices = [labels_preictal.index(lbl) for lbl in common_labels]

    # Subset and return in common order
    eeg_interictal_matched = eeg_interictal[:, interictal_indices]
    eeg_preictal_matched = eeg_preictal[:, preictal_indices]

    return eeg_interictal_matched, eeg_preictal_matched, common_labels


def preprocess_from_mat(interictal_path, preictal_path, target_fs=500, band=None):
    # Constants
    span, step = 1000, 500  # in ms
    min_woi_duration = 60000  # in ms

    # Load raw data
    mat_preictal = loadmat(preictal_path)
    preictal_raw = mat_preictal['sz_data'][0, 0]  # column 0: signal
    fs_preictal = float(mat_preictal['sz_data'][0, 1])  # column 1: sampling frequency

    mat_interictal = loadmat(interictal_path)
    interictal_raw = mat_interictal['sz_data'][0, 0]  # column 0: signal
    fs_interictal = float(mat_interictal['sz_data'][0, 1])  # column 1: sampling frequency

    # Resample to target_fs if needed
    if fs_preictal != target_fs:
        n_samples = int(preictal_raw.shape[1] * target_fs / fs_preictal)
        preictal_raw = resample(preictal_raw, n_samples, axis=1)

    if fs_interictal != target_fs:
        n_samples = int(interictal_raw.shape[1] * target_fs / fs_interictal)
        interictal_raw = resample(interictal_raw, n_samples, axis=1)


    # Align and trim channels
    interictal, preictal, node_labels = match_channels(interictal_raw, preictal_raw)

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
    nodes = node_labels  # channel names

    prep = struct(y=np.array(y), i=np.array(i), x_prep=x, nodes=nodes)
    return prep

