import pandas as pd
import numpy as np
from homaILS.modeling.linear import StepHeading
from homaILS.filtering.kalman import KalmanFilter
from homaILS.plotting.static import plot_2D_localization, map_2D_localization
from homaILS.plotting.dynamic import animate_2D_localization
from homaILS.printing.results import print_2D_localization
from data.CNR_Outdoor.data_utils import load_pdr_dataset, load_gps_dataset
from homaILS.processing.geographic import geodetic_to_enu, geodetic_to_localutm, localutm_to_geodetic
from pyproj import Proj

STEP_LENGTH = 0.7
STEP_STD = 0.1
MAGNETIC_DECLINATION = np.radians(3+(2/3))
HEADING_STD = np.radians(10)


def arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description='Positioning test outdoor')
    parser.add_argument('--window_heading', type=int, default=1, help='Window size for moving average of heading')
    args = parser.parse_args()

    return args


def main():
    window_heading = arg_parser().window_heading

    # Load PDR dataset
    pdr_dataset = load_pdr_dataset()

    # Load GPS dataset
    gps_dataset = load_gps_dataset()

    print(pdr_dataset)
    print(gps_dataset)

    pdr_df = pdr_dataset[['Timestamp', 'Step', 'Heading']]
    pdr_df = pdr_df.dropna()
    # Reset the index
    pdr_df.reset_index(drop=True, inplace=True)
    pdr_df['Step'] = STEP_LENGTH
    # From string to float
    pdr_df['Heading'] = pdr_df['Heading'].str.replace(',', '.').astype(float)
    # Add the magnetic declination to the azimuth
    pdr_df['Heading'] = pdr_df['Heading'] + MAGNETIC_DECLINATION
    # From (E,N) to (x,y)
    pdr_df['Heading'] = np.pi/2 - pdr_df['Heading']
    # Vectorial moving average of heading using cos, sin and arctan2
    pdr_df['cos'] = np.cos(pdr_df['Heading'])
    pdr_df['sin'] = np.sin(pdr_df['Heading'])
    pdr_df['cos_smooth'] = pdr_df['cos'].rolling(window=window_heading, min_periods=1, center=True).mean()
    pdr_df['sin_smooth'] = pdr_df['sin'].rolling(window=window_heading, min_periods=1, center=True).mean()
    pdr_df['Heading_smooth'] = np.arctan2(pdr_df['sin_smooth'], pdr_df['cos_smooth'])
    pdr_df = pdr_df[['Timestamp', 'Step', 'Heading_smooth']]
    pdr_df.rename(columns={'Heading_smooth': 'Heading'}, inplace=True)
    print(pdr_df)

    gps_df = gps_dataset[['Timestamp', 'Longitude', 'Latitude', 'Altitude', 'HorizontalAccuracy', 'VerticalAccuracy']]
    gps_df = gps_df.dropna()
    # Reset the index
    gps_df.reset_index(drop=True, inplace=True)
    # From string to float
    gps_df['Longitude'] = gps_df['Longitude'].str.replace(',', '.').astype(float)
    gps_df['Latitude'] = gps_df['Latitude'].str.replace(',', '.').astype(float)
    gps_df['Altitude'] = gps_df['Altitude'].str.replace(',', '.').astype(float)
    gps_df['HorizontalAccuracy'] = gps_df['HorizontalAccuracy'].str.replace(',', '.').astype(float)
    gps_df['VerticalAccuracy'] = gps_df['VerticalAccuracy'].str.replace(',', '.').astype(float)
    # Convert WGS84 to UTM or ENU
    lon0_deg, lat0_deg, h0 = gps_df['Longitude'][0], gps_df['Latitude'][0], gps_df['Altitude'][0]
    # gps_df[['E', 'N', 'U']] = gps_df.apply(lambda row: geodetic_to_enu(row['Latitude'], row['Longitude'], row['Altitude'], lat0_deg, lon0_deg, h0), axis=1).apply(pd.Series)
    gps_df[['E', 'N']] = gps_df.apply(lambda row:  geodetic_to_localutm(row['Longitude'], row['Latitude'], lon0_deg, lat0_deg, 33, True), axis=1).apply(pd.Series)
    print(gps_df)

    df = pd.merge(pdr_df, gps_df, on='Timestamp', how='outer')
    df = df[['Timestamp', 'Step', 'Heading', 'E', 'N', 'HorizontalAccuracy']]
    print(df)

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

    # Store results for analysis
    model_positions = []
    observed_positions = []
    estimated_positions = []
    timestamps = df['Timestamp'].values

    for _, row in df.iterrows():
        # pause = input("Press Enter to continue...")

        # STEP
        if not pd.isna(row[['Step', 'Heading']]).any():
            kf.predict(alpha=row['Heading'], L=row['Step'])
            model_state = kf.model.a_priori_state(model_state)

            # GPS (observation)
            if not pd.isna(row[['E', 'N']]).any():
                z = row[['E', 'N']].values.reshape(-1, 1)
                kf.update(z, std_r=row['HorizontalAccuracy'])
                model_positions.append(model_state[:2, 0])
                estimated_positions.append(kf.x[:2, 0])
                observed_positions.append(z)

            # NO GPS
            else:
                model_positions.append(model_state[:2, 0])
                estimated_positions.append(kf.x[:2, 0])
                observed_positions.append(None)

        # NO STEP
        else:
            # GPS (observation)
            if not pd.isna(row[['E', 'N']]).any():
                z = row[['E', 'N']].values.reshape(-1, 1)
                kf.update(z, std_r=row['HorizontalAccuracy'])
                model_positions.append(None)
                estimated_positions.append(kf.x[:2, 0])
                observed_positions.append(z)

            # NO GPS
            else:
                observed_positions.append(None)
                model_positions.append(None)
                estimated_positions.append(None)

    # print_2D_localization(model_positions, observed_positions, estimated_positions)
    # plot_2D_localization(model_positions, observed_positions, estimated_positions)
    map_2D_localization(model_positions, observed_positions, estimated_positions, lon0_deg=lon0_deg, lat0_deg=lat0_deg, utm_zone=33, northern_hemisphere=True)
    # animate_2D_localization(model_positions, observed_positions, estimated_positions, timestamps, min_x=-300, max_x=300, min_y=-300, max_y=300)


if __name__ == "__main__":
    main()
