# Epigame-core

**Epigame** is a pipeline for seizure outcome prediction in epilepsy surgery, based on a competition framework of seizure generation. The pipeline localizes the networks (intracranial EEG channels as nodes) which present the highest connectivity changes between interictal and preictal state.

## Features
- Loading .mat files
- Notch and bandpass filtering
- Connectivity analysis of 1-s epochs (PAC, PLV, PLI, CC, SCR, SCI)
- Classification of connectivity via cross-validation
- Game-based networks scoring
- Surgery outcome prediction based on winner-loser resection overlap ratio

## Quick Start

```bash
git clone https://github.com/ivkarla/epigame-core
cd epigame-core
conda create -n epigame python=3.10
conda activate epigame
pip install -e .
python3 -m epigame.main