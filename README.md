# BFM Analysis GUI

A graphical tool for visualizing TDMS signal data, performing PCA-based phase analysis, and viewing instantaneous frequency in 2D/3D.

The last working version is the one with commit message "Bug fixes"

---

## Features
- Load and visualize TDMS files
- Apply PCA to signal data with phase tracking
- View 3D PCA trajectories with reference cycles
- Plot unwrapped phase and compute instantaneous frequency

---

## Installation
1. Fork this repository to your own GitHub account and clone your fork locally:
```bash
git clone https://github.com/your-username/your-forked-repo.git
cd your-forked-repo
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

**Dependencies include:** `numpy`, `scipy`, `scikit-learn`, `nptdms`, `pyqtgraph`, `PyQt6`, `PyOpenGL`, `numba`

---

## Usage
From the root folder, run:
```bash
cd src
python -m main
```

---

## Checking Out Last Working Version
To test or work on the last working version with a certain commit message "XXXXX":

1. **Find the commit hash** for that update:
```bash
git log --oneline
```
Look for a line like with the commit message:
```
abc1234 XXXXX
```

2. **Check out that commit into a new branch:**
```bash
git checkout -b xxxx abc1234
```

> This keeps your current branch intact and creates a new one based on the selected commit.

3. To go back to your main branch later:
```bash
git checkout main
```

This is useful for pinpointing working states while still keeping a backup version for you to work on and make changes.

---

## GUI Workflow
1. Click **"Open TDMS"** to load a `.tdms` file.
2. Use checkboxes to select channels for plotting.
3. Adjust time window using sliders or manual input.
4. Define PCA time range and segment size, then click **"Run PCA"**.
5. View:
   - 3D PCA trajectory with overlaid reference cycles.
   - Unwrapped phase over time.
   - Instantaneous frequency derived from phase.

## Quick PCA Testing (No GUI)
1. If you wnat to quickly check the core PCA algorithm, check scripts/ folder which contains minimal working code:
   - pca.py:
      - Loads a .tdms file from the data/ folder.
      - Applies PCA and detects a reference cycle.
      - Assigns phase indices and tracks updates over time.
      - Saves the computed phase and summary plots to results/.
   - compute_speed.py:
      - Loads the saved phase file from pca.py.
      - Computes instantaneous frequency from phase data.
      - Plots and saves the frequency trace.
2. Make sure you update the file paths in each script to match your own .tdms data.
3. Some algorithms are changed in src/ code, but the core idea of finding ref cycle and measuring speed from ref cycle is the same.
