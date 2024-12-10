from homaILS.dataset_utils import load_raw_dataset, load_gt1_dataset, filter_data, create_anchors_dataset
import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import argparse

# set a parse_args function to parse the arguments
def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='Path Loss Exponent Estimation')
    # Add the anchor_id argument
    parser.add_argument('--anchor_id', type=int, help='The ID of the anchor node', required=True, choices=[6501, 6502, 6503, 6504])

    return parser.parse_args()


def main():
    # Load the calibration dataset
    calibration_file_path = "data/Dataset_AoA_RSS_BLE51/calibration/beacons/beacons_calibration.txt"
    calibration_df = load_raw_dataset(calibration_file_path)

    # Take the arguments from the command line
    args = parse_args()
    anchor_id = args.anchor_id

    # Filter calibration data
    calibration_df_filtered = filter_data(calibration_df, Anchor_ID=anchor_id)
    calibration_df_filtered = calibration_df_filtered.reset_index(drop=True)

    # Print a preview of the filtered dataset
    print("Filtered Calibration Data:")
    print(calibration_df_filtered)

    # Load the ground truth dataset for calibration
    gt_calibration_file_path = "data/Dataset_AoA_RSS_BLE51/calibration/gt/gt_calibration.txt"
    gt_calibration_df = load_gt1_dataset(gt_calibration_file_path)

    # Print a preview of the ground truth dataset for calibration
    print("\nGround Truth Calibration Data:")
    print(gt_calibration_df)

    # Create an IntervalIndex from the start/end times
    intervals = pd.IntervalIndex.from_arrays(gt_calibration_df['Start_Time'], gt_calibration_df['End_Time'], closed='both')
    
    # For each RSS timestamp, find which interval it falls into
    interval_positions = intervals.get_indexer(calibration_df_filtered['Epoch_Time'])
    calibration_df_filtered['GT_ID'] = interval_positions
    calibration_df_filtered = calibration_df_filtered[calibration_df_filtered['GT_ID'] != -1]
    calibration_df_filtered = calibration_df_filtered.reset_index(drop=True)
    print("\nCalibration Data with GT_ID:")
    print(calibration_df_filtered)

    # Take only the Channel = 37 with the 2nd polarization
    rss_df = filter_data(calibration_df_filtered, Channel=37)
    rss_df = rss_df[['RSS_2nd_Pol', 'GT_ID']]
    # Drop the RSS values that have Z scores greater than 2
    rss_df = rss_df[np.abs(rss_df['RSS_2nd_Pol'] - rss_df['RSS_2nd_Pol'].mean()) <= (2 * rss_df['RSS_2nd_Pol'].std())]
    # Compute the mean RSS for each interval
    rss_df= rss_df.groupby('GT_ID').mean()
    # Reneme RSS_2nd_Pol to RSS
    rss_df = rss_df.rename(columns={'RSS_2nd_Pol': 'RSS'})
    rss_df = rss_df.reset_index()
    print("\nRSS Data:")
    print(rss_df)

    # Create a dataset with the anchor nodes' information
    anchors_df = create_anchors_dataset()
    print("\nAnchors Data:")
    print(anchors_df)
    anchors_df_filtered = filter_data(anchors_df, Anchor_ID=anchor_id)
    anchors_df_filtered = anchors_df_filtered.reset_index(drop=True)

    # Extract the points as Nx2 and Mx2 arrays
    anchors_points = anchors_df_filtered[['Pos_x', 'Pos_y']].values  # Shape (M, 2)
    gt_points = gt_calibration_df[['GT_x','GT_y']].values  # Shape (N, 2)
    # Compute the distance matrix between each pair of points in the anchors dataset and the ground truth dataset
    distance_matrix = cdist(anchors_points, gt_points, metric='euclidean')
    # Set new columns in gt_calibration_df with the distances to each anchor
    for i in range(distance_matrix.shape[0]):
        gt_calibration_df[f"Distance_to_{anchors_df_filtered['Anchor_ID'][i]}"] = distance_matrix[i, :]
    # Print a preview of the ground truth dataset for calibration with the distances to each anchor
    print("\nGround Truth Calibration Data with Distances:")
    print(gt_calibration_df)

    # Merge the RSS data with the ground truth calibration data
    cal_df = gt_calibration_df.merge(rss_df, left_index=True, right_on='GT_ID')
    cal_df = cal_df[[f'Distance_to_{anchor_id}', 'RSS']]
    # Sort the calibration data by distance to the anchor
    cal_df = cal_df.sort_values(by=f'Distance_to_{anchor_id}')
    cal_df = cal_df.reset_index(drop=True)
    
    # Print a preview of the ground truth dataset for calibration with the distances to each anchor
    print("\nGround Truth Calibration Data with Distances:")
    print(cal_df)

    # Take the logarithm (base 10) of the distances
    exp10_X = cal_df[f'Distance_to_{anchor_id}'].values.reshape(-1, 1)
    X = np.log10(exp10_X)
    y = cal_df['RSS'].values
    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    # From the model:
    # RSS = intercept + slope * log10(distance)
    # slope = -10 * alpha  => alpha = -slope / 10
    alpha = -slope / 10
    RSS_1m = intercept  # If your reference distance d_0 = 1 meter
    print("Estimated path loss exponent (alpha):", alpha)
    print("Estimated RSS at 1 meter (RSS_1m):", RSS_1m)

    # Plot the distances to the anchor vs. the RSS values
    plt.figure(num=1)
    plt.scatter(exp10_X, y, color='red')
    plt.plot(exp10_X, model.predict(X), color='black')
    plt.xlabel(f'Distance to Anchor {anchor_id}')
    plt.ylabel('RSS')
    plt.title(f'RSS vs. Distance to Anchor {anchor_id}')
    plt.grid(True)
    plt.show()

    # Plot the points of the calibration dataset in room from (0, 0) to (1200, 600)
    # Set label anchor_df['Anchor_ID'] to anchors points
    plt.figure(num=2)
    plt.scatter(gt_calibration_df['GT_x'], gt_calibration_df['GT_y'], c='blue', marker='o')
    plt.scatter(anchors_df['Pos_x'], anchors_df['Pos_y'], c='red', marker='o', s=100, label='Anchors')
    # Annotating points with labels
    for i, label in enumerate(anchors_df['Anchor_ID']):
        plt.text(anchors_df['Pos_x'][i], anchors_df['Pos_y'][i], label, fontsize=10)
    # plot the room
    plt.plot([0, 1200, 1200, 0, 0], [0, 0, 600, 600, 0], 'k-')
    plt.xlim(-200, 1400)
    plt.ylim(-200, 800)
    plt.xlabel('GT_x')
    plt.ylabel('GT_y')
    plt.title('Ground Truth Calibration Points')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
