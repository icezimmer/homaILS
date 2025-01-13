import pandas as pd
import numpy as np
from homaILS.modeling.linear import StepHeading
from homaILS.filtering.kalman import KalmanFilter
from homaILS.plotting.static import plot_2D_localization
from homaILS.plotting.dynamic import animate_2D_localization
from homaILS.printing.results import print_2D_localization
from data.CNR_Outdoor.data_utils import load_pdr_dataset, load_gps_dataset
from homaILS.processing.geographic import geodetic_to_enu


#TODO: repaire the code
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Test the Kalman Filter with a Uniform Linear Motion model.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    # parser.add_argument('--dt', type=float, required=True, help='Time step.')
    parser.add_argument('--measures', type=int, required=True, help='Number of steps between measurements.')
    parser.add_argument('--r', type=float, required=True, help='Measurement noise.')
    return parser.parse_args()

#TODO: Implement the main function
def main():
    # Load PDR dataset
    pdr_dataset = load_pdr_dataset()

    # Load GPS dataset
    gps_dataset = load_gps_dataset()

    print(pdr_dataset)
    print(gps_dataset)

    args = parse_args()

    pdr_df = pdr_dataset[['Timestamp', 'Step', 'Heading']]
    pdr_df = pdr_df.dropna()
    print(pdr_df)
    # from string to float
    pdr_df['Step'] = pdr_df['Step'].str.replace(',', '.').astype(float)
    pdr_df['Heading'] = pdr_df['Heading'].str.replace(',', '.').astype(float)
    print(pdr_df)
    pdr_df['Heading'] = np.pi/2 - pdr_df['Heading']
    print(pdr_df)

    gps_df = gps_dataset[['Timestamp', 'Latitude', 'Longitude', 'Altitude', 'Speed']]
    gps_df = gps_df.dropna()
    print(gps_df)
    # from string to float
    gps_df['Latitude'] = gps_df['Latitude'].str.replace(',', '.').astype(float)
    gps_df['Longitude'] = gps_df['Longitude'].str.replace(',', '.').astype(float)
    gps_df['Altitude'] = gps_df['Altitude'].str.replace(',', '.').astype(float)
    gps_df['Speed'] = gps_df['Speed'].str.replace(',', '.').astype(float)
    print(gps_df)

    df = pd.merge(pdr_df, gps_df, on='Timestamp', how='outer')
    print(df)

    # Function to fill NaN based on the nearest timestamp
    def fill_by_nearest_time(df_, col):
        df = df_.copy()
        # Take only if col is not NaN
        df2 = df.dropna(subset=[col])
        for i, row in df.iterrows():
            if pd.isna(row[col]):  # If the value is NaN
                # Compute absolute time differences
                time_diffs = (df2['Timestamp'] - row['Timestamp']).abs()
                closest_index = time_diffs.idxmin()  # Find the index of the nearest timestamp
                df.at[i, col] = df.at[closest_index, col]  # Fill the NaN with the nearest value
        return df

    df = fill_by_nearest_time(df, 'Latitude')
    df = fill_by_nearest_time(df, 'Longitude')
    df = fill_by_nearest_time(df, 'Altitude')
    df = fill_by_nearest_time(df, 'Speed')
    print(df)

    # Leave rows with no NaN in step
    df = df.dropna(subset=['Step'])
    # Reset index
    df = df.reset_index(drop=True)
    print(df)

    lat0_deg, lon0_deg, h0 = df['Latitude'][0], df['Longitude'][0], df['Altitude'][0]
    df[['E', 'N', 'U']] = df.apply(lambda row: geodetic_to_enu(row['Latitude'], row['Longitude'], row['Altitude'], lat0_deg, lon0_deg, h0), axis=1).apply(pd.Series)
    df = df[['Timestamp', 'Step', 'Heading', 'E', 'N', 'U', 'Speed']]
    print(df)

    # Compute the mean of the step lengths and alphas
    step_lengths = [104 / 164, 181 / 300, 111 / 176, 226 / 337, 62 / 100]
    # Mean and standard deviation of the step lengths
    L = sum(step_lengths) / len(step_lengths)
    dL = (sum([(l - L) ** 2 for l in step_lengths]) / len(step_lengths)) ** 0.5
    alphas = df['Heading'].values
    # Mean and standard deviation of the alphas
    alpha = alphas.mean()
    dalpha = alphas.std()
    print(f"\nStep Length mean (std): {L:.2f} ({dL:.2f})")
    print(f"Alpha mean (std): {alpha:.2f} ({dalpha:.2f})")

     # Seed for reproducibility
    np.random.seed(args.seed)

    model = StepHeading(r=args.r, L=L, dL=dL, dalpha=dalpha)
    
    # Initialize the filter
    kf = KalmanFilter(model)
    
    # Initial state: Suppose we start at position = (0, 0)
    x0 = np.array([[0], [0]])
    # Initial covariance matrix
    P0 = np.eye(x0.shape[0])*0.1
    kf.initialize(x0, P0)

    # model initial state equals the initial state
    model_state = x0

    # Store results for analysis
    model_positions = []
    measured_positions = []
    estimated_positions = []
    timestamps = df['Timestamp'].values

    steps = len(df)
    step_update = args.measures
    for t in range(steps):
        # Model state evolves
        model_state = kf.model.step(model_state)

        # Predict the next state
        kf.predict(alpha=alphas[t])

        # Update the Kalman Filter
        if t % step_update == 0:
            # Measured position with noise
            z = df[['E', 'N']].values[t].reshape(-1, 1)
            kf.update(z, alpha=alphas[t])
            measured_positions.append(z)
        else:
            measured_positions.append(None)

        # Logging for analysis
        model_positions.append(model_state[:2, 0])
        estimated_positions.append(kf.x[:2, 0])

    # print_2D_localization(model_positions, measured_positions, estimated_positions)
    # plot_2D_localization(model_positions, measured_positions, estimated_positions)
    animate_2D_localization(model_positions, measured_positions, estimated_positions, timestamps)



if __name__ == "__main__":
    main()
