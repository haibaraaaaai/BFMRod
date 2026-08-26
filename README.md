# pca-rotation-analysis

A modular Python toolkit for analyzing bacterial flagellar motor (BFM) rotation from TDMS signal data. Includes a PyQt6-based GUI for PCA-based phase tracking and frequency estimation, along with supporting scripts for analysis and data management.

---

## ⚠️ Project Status: Paused

**This PCA-based method is no longer under active development.** A better approach for nanorod signals on the BFM was found — based directly on signal **anisotropy** — which handles changing speeds and changing trajectories much more robustly.

➡️ **See the successor project:** https://github.com/haibaraaaaai/anisotropy-rotation-analysis

That said, the PCA approach here has its own merits (e.g. extracting a dominant rotation trajectory without assuming a signal model) and may be revisited for other situations in the future. The code is kept for reference and potential reuse.

---

## Project Structure

```
pca-rotation-analysis/
├── src/                # Main GUI and backend logic
├── tools/              # Custom analysis tools outside the GUI
├── archive/            # Outdated or prototype PCA scripts
├── results/            # GUI-generated output (overwritten on each run)
├── results_backup/     # Manually saved outputs for further analysis
├── data/               # Raw TDMS input files
├── results_notes.md    # Manual log of datasets and findings
├── README.md           # You're here!
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

Standalone prototype scripts are available in `archive/` for running PCA and speed analysis outside the GUI:

- `pca.py`: Loads TDMS, applies PCA, detects reference cycle, tracks phase
- `compute_speed.py`: Loads phase output, computes instantaneous frequency

These scripts predate the GUI and are useful for testing or batch jobs.

---

## Scripts Overview

The `tools/` folder contains standalone scripts for inspecting and analyzing phase/speed data produced by the GUI:

- `compute_speed.py`: Per-revolution speed with GMM level detection and chi² smoothing.
- `compute_speed_v2.py`: Per-revolution speed with phase pre-smoothing, deterministic step fitting, and caching.
- `summarize_folder_speed.py`: Compare speed distributions across multiple datasets and plot histograms.
- `speed_per_angle.py`: Analyze speed variation as a function of angular position across revolutions.
- `speed_angle_hist.py`: Plot histograms of speed within specific angular regions across revolutions.
- `speed_angle_gmm.py`: Extract per-revolution speed from a fixed-angle region and fit a Gaussian mixture model.
- `angle_rev_speed_comparison.py`: Compare per-angle speed against average speed across revolutions.
- `polar_plot.py`: Polar/linear visualization of speed vs angle (bearing shape).
- `assign_stator_number.py`: Quantize step-fitted speed levels into discrete stator numbers.
