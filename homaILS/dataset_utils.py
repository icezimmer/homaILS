import pandas as pd

def load_raw_dataset(file_path):
    """Load a raw AoA RSS BLE 5.1 dataset.
    Columns:
    - Epoch_Time: receiving beacon message time;
    - Tag_ID: Bluetooth tag identifier;
    - RSS_1st_Pol: RSS value of the 1st polarization;
    - AoA_Az: AoA value on the azimuth plane estimate by the anchor;
    - AoA_El: AoA value on the elevation plane estimate by the anchor;
    - RSS_2nd_Pol: RSS value of the 2nd polarization;
    - Channel: the Bluetooth channel used by the anchor to receive the beacon message;
    - Anchor_ID: the ID of the anchor node.
    Input:
    - file_path: the path to the dataset file.
    """

    # Define column names
    columns = ["Epoch_Time", "Tag_ID", "RSS_1st_Pol", "AoA_Az", "AoA_El", "RSS_2nd_Pol", "Channel", "Anchor_ID"]

    # Load the dataset
    df = pd.read_csv(file_path, header=None, names=columns)

    # Convert timestamp to datetime
    df['Epoch_Time'] = pd.to_datetime(df['Epoch_Time'], unit='ms')
    return df

def load_gt1_dataset(file_path):
    """
    Load a ground truth dataset for Calibration and Static tasks.
    Columns:
    - Start_Time: the timestamp the tag arrives in a specific position;
    - End_Time: the timestamp the tag leaves a specific
    position;
    - GT_x: the x-coordinate of the tag's location;
    - GT_y: the y-coordinate of the tag's location;
    """

    # Define column names
    columns = ["Start_Time", "End_Time", "GT_x", "GT_y"]

    # Load the dataset
    df = pd.read_csv(file_path, header=None, names=columns)

    # Convert timestamp to datetime
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], unit='ms')
    df['End_Time'] = pd.to_datetime(df['End_Time'], unit='ms')
    return df

def filter_data(df, **conditions):
    """Filter data based on column-value conditions."""
    for column, value in conditions.items():
        df = df[df[column] == value]
    return df

def create_anchors_dataset():
    """Create a dataset with the anchor nodes' information.
    Columns:
    - Anchor_ID: the ID of the anchor node;
    - Type: the type of the anchor node;
    - Pos_x: the x-coordinate of the anchor node;
    - Pos_y: the y-coordinate of the anchor node;
    """
    # Define the anchor nodes' information
    anchors = {
        "Anchor_ID": [6501, 6502, 6503, 6504],
        "Type": ["BLE", "BLE", "BLE", "BLE"],
        "Pos_x": [0, 600, 1200, 600],
        "Pos_y": [300, 0, 300, 600]
    }
    df = pd.DataFrame(anchors)
    return df
