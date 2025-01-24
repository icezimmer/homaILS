import pandas as pd
import numpy as np
from homaILS.modeling.linear import StepHeading
from homaILS.filtering.kalman import KalmanFilter
from homaILS.plotting.static import plot_2D_localization, map_2D_localization
from homaILS.plotting.dynamic import animate_2D_localization
from homaILS.printing.results import print_2D_localization
from homaILS.processing.geographic import geodetic_to_localutm

STEP_LENGTH = 0.65
STEP_STD = 0.1
MAGNETIC_DECLINATION = np.radians(3+(2/3))
HEADING_STD = np.radians(10)


def arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Positioning test 56')
    parser.add_argument('--window_heading', type=int, default=1, help='Window size for moving average of heading')
    args = parser.parse_args()

    return args


def main():

    window_heading = arg_parser().window_heading

    # Load data
    step_dataset = pd.read_csv("data/17_01_2025/20250117T141441-56-Step.csv")
    orientation_dataset = pd.read_csv("data/17_01_2025/20250117T141441-56-Orientation.csv")
    gps_dataset = pd.read_csv("data/17_01_2025/20250117T141441-56-GPS.csv")

    print(step_dataset)
    print(orientation_dataset)
    print(gps_dataset)

    # Set the step length
    step_dataset['Step'] = STEP_LENGTH

    # Add the magnetic declination to the azimuth
    orientation_dataset['Azimuth'] = orientation_dataset['Azimuth'] + MAGNETIC_DECLINATION
    # From (E,N) to (x,y)
    orientation_dataset['Azimuth'] = np.pi/2 - orientation_dataset['Azimuth']
    # Vectorial moving average of heading using cos, sin and arctan2
    orientation_dataset['cos'] = np.cos(orientation_dataset['Azimuth'])
    orientation_dataset['sin'] = np.sin(orientation_dataset['Azimuth'])
    orientation_dataset['cos_smooth'] = orientation_dataset['cos'].rolling(window=window_heading, min_periods=1, center=True).mean()
    orientation_dataset['sin_smooth'] = orientation_dataset['sin'].rolling(window=window_heading, min_periods=1, center=True).mean()
    # Compute the angle smoothed in [-pi, pi]
    orientation_dataset['Azimuth_smooth'] = np.arctan2(orientation_dataset['sin_smooth'], orientation_dataset['cos_smooth'])

    pdr_df = pd.merge(step_dataset, orientation_dataset, on='Timestamp', how='outer')
    pdr_df = pdr_df[['Timestamp', 'Step', 'Azimuth_smooth']]
    # Fill the NaN values in the azimuth column with the previous value
    pdr_df['Azimuth_smooth'] = pdr_df['Azimuth_smooth'].ffill()
    # Drop where Step is NaN
    pdr_df = pdr_df.dropna(subset=['Step'])
    # Reset the index
    pdr_df.reset_index(drop=True, inplace=True)
    # Rename the columns
    pdr_df.rename(columns={'Azimuth_smooth': 'Heading'}, inplace=True)
    print(pdr_df)

    gps_df = gps_dataset[['Timestamp', 'Longitude', 'Latitude', 'HorizontalAccuracy']]
    gps_df = gps_df.dropna()
    # Reset the index
    gps_df.reset_index(drop=True, inplace=True)
    # Convert WGS84 to UTM or ENU
    lon0_deg, lat0_deg = gps_df['Longitude'][0], gps_df['Latitude'][0]
    gps_df[['E', 'N']] = gps_df.apply(lambda row:  geodetic_to_localutm(row['Longitude'], row['Latitude'], lon0_deg, lat0_deg, 33, True), axis=1).apply(pd.Series)
    print(gps_df)
    # Drop even rows
    # gps_df = gps_df.iloc[::2]

    df = pd.merge(pdr_df, gps_df, on='Timestamp', how='outer')
    df = df[['Timestamp', 'Step', 'Heading', 'E', 'N', 'HorizontalAccuracy']]
    print(df)

    pause = input("Press Enter to continue...")

    model = StepHeading(std_L=STEP_STD, std_alpha=HEADING_STD)
    
    # Initialize the filter
    kf = KalmanFilter(model)
    
    # Initial state: Suppose we start at position = (0, 0)
    x0 = np.array([[0], [0]])
    # Initial covariance matrix
    # P0 = np.eye(x0.shape[0])
    P0 = np.zeros((2, 2))
    kf.initialize(x0, P0)    

    # model initial state equals the initial state
    model_state = x0
    model_covariance = P0

    # Store results for analysis
    model_positions = []
    observed_positions = []
    estimated_positions = []
    model_errors = []
    observed_errors = []
    estimated_errors = []
    timestamps = df['Timestamp'].values

    for k, row in df.iterrows():
        # pause = input("Press Enter to continue...")

        # STEP
        if not pd.isna(row[['Step', 'Heading']]).any():
            kf.predict(alpha=row['Heading'], L=row['Step'])
            model_state = kf.model.a_priori_state(model_state)
            model_covariance = kf.model.a_priori_covariance(model_covariance)

            # GPS (observation)
            if not pd.isna(row[['E', 'N']]).any():
                z = row[['E', 'N']].values.reshape(-1, 1)
                kf.update(z, std_r=row['HorizontalAccuracy'])
                model_positions.append(model_state[:2, :])
                estimated_positions.append(kf.x[:2, :])
                observed_positions.append(z)
                model_errors.append(model_covariance)
                observed_errors.append(kf.model.R)
                estimated_errors.append(kf.P)

            # NO GPS
            else:
                model_positions.append(model_state[:2, :])
                estimated_positions.append(kf.x[:2, :])
                observed_positions.append(None)
                model_errors.append(model_covariance)
                observed_errors.append(None)
                estimated_errors.append(kf.P)

        # NO STEP
        else:
            # GPS (observation)
            if not pd.isna(row[['E', 'N']]).any():
                z = row[['E', 'N']].values.reshape(-1, 1)
                kf.update(z, std_r=row['HorizontalAccuracy'])
                model_positions.append(None)
                estimated_positions.append(kf.x[:2, :])
                observed_positions.append(z)
                model_errors.append(None)
                observed_errors.append(kf.model.R)
                estimated_errors.append(kf.P)

            # NO GPS
            else:
                observed_positions.append(None)
                model_positions.append(None)
                estimated_positions.append(None)
                model_errors.append(None)
                observed_errors.append(None)
                estimated_errors.append(None)

    # print_2D_localization(model_positions, observed_positions, estimated_positions)
    plot_2D_localization(model_positions, observed_positions, estimated_positions, model_errors, observed_errors, estimated_errors)
    # map_2D_localization(model_positions, observed_positions, estimated_positions, lon0_deg=lon0_deg, lat0_deg=lat0_deg, utm_zone=33, northern_hemisphere=True)
    animate_2D_localization(model_positions, observed_positions, estimated_positions, model_errors, observed_errors, estimated_errors, timestamps, min_x=-300, max_x=300, min_y=-300, max_y=300)


if __name__ == "__main__":
    main()
