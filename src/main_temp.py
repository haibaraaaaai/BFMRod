import os
from processing import run_pca_analysis
from utils import plot_3d_trajectory, plot_pca_component, plot_phase_evolution

# Get absolute path of project root (one level above "src")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Ensure data path is correctly resolved
tdms_file_path = os.path.join(PROJECT_ROOT, "data", "2025.02.23 ox", "file1.tdms")

# Run PCA analysis
results = run_pca_analysis(tdms_file_path)

# Print or process results
print("PCA Analysis Complete.")
print("Detected Cycle Indices:", results["indices"])

# --- Plot Results to Verify ---
plot_3d_trajectory(results["X_pca"], results["smooth_points"])
plot_pca_component(results["X_pca"], results["indices"], results["smooth_points"])
plot_phase_evolution(results["phase"])
