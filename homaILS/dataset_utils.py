import pandas as pd

def load_dataset(file_path):
    # Define column names
    columns = ["Epoch_Time", "Tag_ID", "RSS_1st_Pol", "AoA_Az", "AoA_El", "RSS_2nd_Pol", "Channel", "Anchor_ID"]

    # Load the dataset
    df = pd.read_csv(file_path, header=None, names=columns)

    # Convert timestamp to datetime
    df['Epoch_Time'] = pd.to_datetime(df['Epoch_Time'], unit='ms')
    return df

def filter_data(df, **conditions):
    """Filter data based on column-value conditions."""
    for column, value in conditions.items():
        df = df[df[column] == value]
    return df

