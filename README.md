# BFMRod

A modular Python toolkit for analyzing bacterial flagellar motor (BFM) rotation from TDMS signal data. Includes a PyQt6-based GUI for PCA-based phase tracking and frequency estimation, along with supporting scripts for analysis and data management.

---

## Project Structure

```
BFMRod/
├── src/                # Main GUI and backend logic
├── tools/              # Custom analysis tools outside the GUI
├── archive/            # Outdated or prototype PCA scripts
├── results/            # GUI-generated output (overwritten on each run)
├── results_backup/     # Manually saved outputs for further analysis
├── data/               # Raw TDMS input files
├── docs/               # Reference code and slides
├── results_notes.md   # Manual log of datasets and findings
├── README.md           # You’re here!
├── TODO.md             # Task tracking
├── requirements.txt    # Python dependencies
```

---

## Features

- Load and explore TDMS files with a GUI
- Apply PCA to extract dominant signal features
- Automatically detect reference cycles
- Track unwrapped phase and compute speed over time
- Visualize 3D PCA trajectories, phase, and frequency
- Save results for batch analysis or manual inspection

---

## Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the GUI**:
   ```bash
   cd src
   python -m main
   ```

---

## Alternative Analysis (Without GUI)

Standalone scripts are available in `archive/` for running PCA and speed analysis outside the GUI:

- `pca.py`: Loads TDMS, applies PCA, detects reference cycle, tracks phase
- `compute_speed.py`: Loads phase output, computes instantaneous frequency

These scripts are ideal for testing or running batch jobs.

---

## Scripts Overview

The `tools/` folder contains standalone scripts for inspecting and analyzing BFM data from the GUI:

- `compute_speed.py`: Compute and plot instantaneous frequency from saved phase.
- `harmonics_check.py`: Plot X+iY trajectory to inspect signal anisotropy and harmonic structure.
- `summarize_folder_speed.py`: Compare speed distributions across multiple datasets and plot histograms.
- `speed_per_angle.py`: Analyze speed variation as a function of angular position across revolutions.
- `speed_angle_hist.py`: Plot histograms of speed within specific angular regions across revolutions.
- `speed_angle_gmm.py`: Extract per-revolution speed from a fixed-angle region and fit a Gaussian mixture model.
