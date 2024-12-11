import pandas as pd

def load_pdr_dataset():
    """Load the PDR dataset.
    Columns:
    - Timestamp: the timestamp of the PDR measurement;
    - PDR: the PDR value.
    Input:
    - file_path: the path to the dataset file.
    """
    # Load the dataset
    df = pd.read_csv('data/CNR_Outdoor/20220407T133143-Log-PDR.csv', delimiter=';')
    return df

def load_gps_dataset():
    """Load the GPS dataset.
    Columns:
    - Timestamp: the timestamp of the GPS measurement;
    - Latitude: the latitude of the GPS measurement;
    - Longitude: the longitude of the GPS measurement.
    Input:
    - file_path: the path to the dataset file.
    """
    # Load the dataset
    df = pd.read_csv('data/CNR_Outdoor/20220407T133143-Log-GPS.csv', delimiter=';')
    return df
