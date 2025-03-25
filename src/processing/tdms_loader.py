"""
tdms_loader.py – Load TDMS files and extract timestamped channel data.
"""

import numpy as np
from nptdms import TdmsFile


def load_tdms_data(tdms_file_path):
    """
    Load a TDMS file and extract timestamps and four predefined channels (C0, C90, C45, C135).

    Args:
        tdms_file_path (str): Path to the TDMS file.

    Returns:
        tuple:
            - timestamps (np.ndarray): Time vector.
            - data (np.ndarray): Matrix of channel data (columns: C0, C90, C45, C135).
            - channel_names (list): Names of the channels extracted.
    """
    try:
        # Read TDMS file
        tdms_file = TdmsFile.read(tdms_file_path)
        group = tdms_file.groups()[0]
        channels = group.channels()

        # Extract timestamps from the first channel
        timestamps = np.array(channels[0].time_track())

        # Manually define the four channels
        C90 = channels[0].data
        C45 = channels[1].data
        C135 = channels[2].data
        C0 = channels[3].data

        # Stack into data matrix
        data = np.column_stack((C0, C90, C45, C135))
        channel_names = ["C0", "C90", "C45", "C135"]

        return timestamps, data, channel_names

    except FileNotFoundError:
        print(f"[Error] TDMS file '{tdms_file_path}' not found.")
        return None, None, None

    except Exception as e:
        print(f"[Error] Failed to read TDMS file '{tdms_file_path}': {e}")
        return None, None, None
        # Optionally: raise  # Uncomment to let exception propagate
