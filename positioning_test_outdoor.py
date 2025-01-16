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


def main():
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
    # From string to float
    pdr_df['Step'] = pdr_df['Step'].str.replace(',', '.').astype(float)
    pdr_df['Heading'] = pdr_df['Heading'].str.replace(',', '.').astype(float)
    pdr_df['Heading'] = np.pi/2 - pdr_df['Heading']
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

    # for _, row in gps_df.iterrows():
    #     print(row[['Longitude', 'Latitude']], localutm_to_geodetic(row['E'], row['N'], lon0_deg, lat0_deg, 33, True))
    #     pause = input("Press Enter to continue...")

    df = pd.merge(pdr_df, gps_df, on='Timestamp', how='outer')
    df = df[['Timestamp', 'Step', 'Heading', 'E', 'N', 'HorizontalAccuracy']]
    print(df)

    # for _, row in df.iterrows():
    #     print(row)
    #     pause = input("Press Enter to continue...")

    # # Compute the standard deviations of the step length and heading
    # std_L = df['Step'].dropna().values.std()
    # std_alpha = df['Heading'].dropna().values.std()
    # print(f"\nStep Length std: {std_L:.2f}")
    # print(f"Alpha mean std: {std_alpha:.2f}")
    std_L = 0.1
    std_alpha = np.radians(10)

    model = StepHeading(std_L=std_L, std_alpha=std_alpha)
    
    # Initialize the filter
    kf = KalmanFilter(model)
    
    # Initial state: Suppose we start at position = (0, 0)
    x0 = np.array([[0], [0]])
    # Initial covariance matrix
    P0 = np.eye(x0.shape[0])
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
            model_state = kf.model.step(model_state)
            kf.predict(alpha=row['Heading'], L=0.7)

            # GPS (observation)
            if not pd.isna(row[['E', 'N']]).any():
                z = np.array([[row['E']], [row['N']]])
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
                z = np.array([[row['E']], [row['N']]])
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
