from homaILS.dataset_utils import load_dataset, filter_calibration_data

def main():
    # Path to the dataset
    file_path = "data/Dataset_AoA_RSS_BLE51/calibration/beacons/beacons_calibration.txt"

    # Load the dataset
    df = load_dataset(file_path)

    # Filter calibration data
    calibration_df = filter_calibration_data(df)

    # Print a preview of the filtered dataset
    print("Filtered Calibration Data:")
    print(calibration_df.head())

if __name__ == "__main__":
    main()
