import pandas as pd
import numpy as np
from homaILS.modeling.linear import StepHeading
from homaILS.filtering.kalman import KalmanFilter
from homaILS.plotting.static import plot_2D_localization
from homaILS.plotting.dynamic import animate_2D_localization
from homaILS.printing.results import print_2D_localization
from data.CNR_Outdoor.data_utils import load_pdr_dataset, load_gps_dataset
from homaILS.processing.geographic import geodetic_to_enu, geodetic_to_utm
from pyproj import Proj, transform


def main():
    # Load PDR dataset
    pdr_dataset = load_pdr_dataset()

    # Load GPS dataset
    gps_dataset = load_gps_dataset()

    print(pdr_dataset)
    print(gps_dataset)

    pdr_df = pdr_dataset[['Timestamp', 'Step', 'Heading']]
    pdr_df = pdr_df.dropna()
    print(pdr_df)
    # from string to float
    pdr_df['Step'] = pdr_df['Step'].str.replace(',', '.').astype(float)
    pdr_df['Heading'] = pdr_df['Heading'].str.replace(',', '.').astype(float)
    print(pdr_df)
    pdr_df['Heading'] = np.pi/2 - pdr_df['Heading']
    print(pdr_df)

    gps_df = gps_dataset[['Timestamp', 'Latitude', 'Longitude', 'Altitude', 'Speed', 'HorizontalAccuracy', 'VerticalAccuracy', 'SpeedAccuracy']]
    gps_df = gps_df.dropna()
    print(gps_df)
    # from string to float
    gps_df['Latitude'] = gps_df['Latitude'].str.replace(',', '.').astype(float)
    gps_df['Longitude'] = gps_df['Longitude'].str.replace(',', '.').astype(float)
    gps_df['Altitude'] = gps_df['Altitude'].str.replace(',', '.').astype(float)
    gps_df['Speed'] = gps_df['Speed'].str.replace(',', '.').astype(float)
    gps_df['HorizontalAccuracy'] = gps_df['HorizontalAccuracy'].str.replace(',', '.').astype(float)
    gps_df['VerticalAccuracy'] = gps_df['VerticalAccuracy'].str.replace(',', '.').astype(float)
    gps_df['SpeedAccuracy'] = gps_df['SpeedAccuracy'].str.replace(',', '.').astype(float)
    print(gps_df)

    df = pd.merge(pdr_df, gps_df, on='Timestamp', how='outer')
    print(df)

    # lat0_deg, lon0_deg, h0 = df['Latitude'][0], df['Longitude'][0], df['Altitude'][0]
    # df[['E', 'N', 'U']] = df.apply(lambda row: geodetic_to_enu(row['Latitude'], row['Longitude'], row['Altitude'], lat0_deg, lon0_deg, h0), axis=1).apply(pd.Series)
    # df = df[['Timestamp', 'Step', 'Heading', 'E', 'N', 'U', 'Speed', 'HorizontalAccuracy', 'VerticalAccuracy', 'SpeedAccuracy']]
    # print(df)

    # Sistema di coordinate UTM per la zona
    utm_proj = Proj(proj="utm", zone=33, ellps="WGS84", south=False)

    # Convert WGS84 to UTM or ENU
    lon0_deg, lat0_deg, h0 = df['Longitude'][0], df['Latitude'][0], df['Altitude'][0]
    df[['E', 'N']] = df.apply(lambda row:  geodetic_to_utm(row['Longitude'], row['Latitude'], lon0_deg, lat0_deg, 33), axis=1).apply(pd.Series)
    # df[['E', 'N', 'U']] = df.apply(lambda row: geodetic_to_enu(row['Latitude'], row['Longitude'], row['Altitude'], lat0_deg, lon0_deg, h0), axis=1).apply(pd.Series)
    df = df[['Timestamp', 'Step', 'Heading', 'E', 'N', 'HorizontalAccuracy']]
    print(df)

    # for _, row in df.iterrows():
    #     print(row)
    #     pause = input("Press Enter to continue...")

    # Compute the mean of the step lengths and alphas
    L = df['Step'].dropna().values.mean()
    dL = df['Step'].dropna().values.std()
    alpha = df['Heading'].dropna().values.mean()
    dalpha = df['Heading'].dropna().values.std()
    print(f"\nStep Length mean (std): {L:.2f} ({dL:.2f})")
    print(f"Alpha mean (std): {alpha:.2f} ({dalpha:.2f})")

    model = StepHeading(L=L, dL=dL, dalpha=dalpha)
    
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
            kf.predict(alpha=row['Heading'])

            # GPS (observation)
            if not pd.isna(row[['E', 'N']]).any():
                z = np.array([[row['E']], [row['N']]])
                kf.update(z, r=row['HorizontalAccuracy'])
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
                kf.update(z, r=row['HorizontalAccuracy'])
                model_positions.append(None)
                estimated_positions.append(kf.x[:2, 0])
                observed_positions.append(z)

            # NO GPS
            else:
                observed_positions.append(None)
                model_positions.append(None)
                estimated_positions.append(None)

    # print_2D_localization(model_positions, observed_positions, estimated_positions)
    plot_2D_localization(model_positions, observed_positions, estimated_positions)
    animate_2D_localization(model_positions, observed_positions, estimated_positions, timestamps, min_x=-300, max_x=300, min_y=-300, max_y=300)


if __name__ == "__main__":
    main()
