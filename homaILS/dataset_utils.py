import pandas as pd

def load_dataset(file_path):
    # Define column names
    columns = ["timestamp", "tag_id", "rssi1", "rssi2", "rssi3", "rssi4", "angle_of_arrival", "anchor_id"]

    # Load the dataset
    df = pd.read_csv(file_path, header=None, names=columns)

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def filter_calibration_data(df, tag_id=8401):
    # Filter for calibration data
    return df[df['tag_id'] == tag_id]
