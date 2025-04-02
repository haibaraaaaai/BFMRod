import os
import numpy as np
import json
import glob

# --- Settings ---
ROOT_DIR = "results"  # Top-level folder to scan
TARGET_FILENAME = "phase_data.npz"
NEW_FILENAME = "phase_data_compressed.npz"  # Overwrite or change to something else like "phase_data_compressed.npz"
REFERENCE_NUM_POINTS = 200  # Needed if you ever reconstruct float phase from phase0

def recompress_npz_file(npz_path):
    folder = os.path.dirname(npz_path)
    print(f"\n[PROCESSING] {npz_path}")

    try:
        data = np.load(npz_path)
        if "phase_time" not in data or "phase0" not in data:
            print("  ⚠️  Missing required fields, skipping.")
            return

        # Extract and convert
        phase_time = data["phase_time"].astype(np.float32)
        phase0 = data["phase0"].astype(np.uint8)

        # Sanity check
        print(f"  - phase_time: {phase_time.dtype}, {phase_time.shape}, {phase_time.nbytes / 1e6:.2f} MB")
        print(f"  - phase0: {phase0.dtype}, {phase0.shape}, {phase0.nbytes / 1e6:.2f} MB")

        # Load existing ref_bounds if present
        ref_json_path = os.path.join(folder, "ref_bounds.json")
        if not os.path.exists(ref_json_path):
            print("  ⚠️  Missing ref_bounds.json, skipping.")
            return
        with open(ref_json_path, "r") as f:
            computed_ref_bound = json.load(f)

        # Save compressed
        np.savez_compressed(
            os.path.join(folder, NEW_FILENAME),
            phase_time=phase_time,
            phase0=phase0,
        )
        with open(ref_json_path, "w") as f:
            json.dump(computed_ref_bound, f, indent=2)

        print("  ✅ Recompression complete.")

    except Exception as e:
        print(f"  ❌ Error: {e}")


def batch_recompress(root_dir):
    npz_paths = glob.glob(os.path.join(root_dir, "**", TARGET_FILENAME), recursive=True)
    print(f"[INFO] Found {len(npz_paths)} .npz files to process.\n")

    for path in npz_paths:
        recompress_npz_file(path)


if __name__ == "__main__":
    batch_recompress(ROOT_DIR)