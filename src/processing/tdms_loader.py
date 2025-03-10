from nptdms import TdmsFile
import numpy as np

def load_tdms_data(tdms_file_path):
    """
    Load the TDMS file and extract timestamps along with four predefined channels (C0, C90, C45, C135).

    Args:
        tdms_file_path (str): Path to the TDMS file.

    Returns:
        tuple: (timestamps, data matrix, channel names list)
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

        # Stack into a matrix
        data = np.column_stack((C0, C90, C45, C135))
        channel_names = ["C0", "C90", "C45", "C135"]
        
        return timestamps, data, channel_names

    except FileNotFoundError:
        print(f"Error: TDMS file '{tdms_file_path}' not found.")
        return None, None, None

    except Exception as e:
        print(f"Error reading TDMS file '{tdms_file_path}': {e}")
        return None, None, None
