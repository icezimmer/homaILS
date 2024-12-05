from homaILS.dataset_utils import load_dataset, filter_data

def main():
    # Path to the dataset
    file_path = "data/Dataset_AoA_RSS_BLE51/calibration/beacons/beacons_calibration.txt"

    # Load the dataset
    calibration_df = load_dataset(file_path)

    # Filter calibration data
    calibration_df_filtered = filter_data(calibration_df, Tag_ID=8401)

    # Print a preview of the filtered dataset
    print("Filtered Calibration Data:")
    print(calibration_df_filtered)
    print(calibration_df_filtered.info())
    print(calibration_df_filtered[['RSS_1st_Pol', "AoA_Az", "AoA_El",  'RSS_2nd_Pol']].describe())

if __name__ == "__main__":
    main()
