import pandas as pd
import numpy as np
from homaILS.modeling.linear import StepHeading
from homaILS.filtering.kalman import KalmanFilter
from homaILS.plotting.static import plot_2D_localization
from homaILS.plotting.dynamic import animate_2D_localization
from homaILS.printing.results import print_2D_localization


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
    acceleration_path = "data/Data_17122024/20241217T133655-19-Acceleration.csv"
    GPS_path = "data/Data_17122024/20241217T133655-19-GPS.csv"
    orientation_path = "data/Data_17122024/20241217T133655-19-Orientation.csv"
    step_path = "data/Data_17122024/20241217T133655-19-Step.csv"

    acceleration_df = pd.read_csv(acceleration_path)
    GPS_df = pd.read_csv(GPS_path)
    orientation_df = pd.read_csv(orientation_path)
    step_df = pd.read_csv(step_path)

    print("\nAcceleration Data:")
    print(acceleration_df)
    print("\nGPS Data:")
    print(GPS_df)
    print("\nOrientation Data:")
    print(orientation_df)
    print("\nStep Data:")
    print(step_df)

    args = parse_args()

    # Add column count equal to 1 in step data
    step_df['Count'] = 1
    # Take only the azimuth column from the orientation data
    azimuth_df = orientation_df[['Timestamp', 'Azimuth']]
    # Change the orientation to north
    azimuth_df['Azimuth'] = np.pi/2 - azimuth_df['Azimuth']
    # Merge the step and orientation data keeping all the information from both dataframes
    df = pd.merge(step_df, azimuth_df, on='Timestamp', how='outer')
    # Fill the NaN values in the azimuth column with the previous value
    df['Azimuth'] = df['Azimuth'].ffill()
    # Fill the NaN values in the Count column with 0
    # df['Count'] = df['Count'].fillna(0)
    # Drop where Count is NaN
    df = df.dropna(subset=['Count'])
    print("\nStep Data and Orientation:")
    print(df)

    # Compute the mean of the step lengths and alphas
    step_lengths = [104 / 164, 181 / 300, 111 / 176, 226 / 337, 62 / 100]
    # Mean and standard deviation of the step lengths
    L = sum(step_lengths) / len(step_lengths)
    dL = (sum([(l - L) ** 2 for l in step_lengths]) / len(step_lengths)) ** 0.5
    alphas = df['Azimuth'].values
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
            z = model_state[:2, 0] + np.random.normal(0, args.r, 2)
            kf.update(z, alpha=alphas[t])
            measured_positions.append(z)
        else:
            measured_positions.append(None)

        # Logging for analysis
        model_positions.append(model_state[:2, 0])
        estimated_positions.append(kf.x[:2, 0])

    # Example timestamps in milliseconds (simulating 12 measurements over time)
    timestamps = [
        1000000,
        1001000,
        1002000,
        1003000,
        1004000,
        1005000,
        1006000,
        1007000,
        1008000,
        1022000,
        1023000,
        1024000,
    ]

    # Example model positions (simulated trajectory)
    model_positions = [
        (0, 0),
        (1, 2),
        (2, 4),
        (3, 6),
        (4, 8),
        (5, 10),
        (6, 12),
        (7, 14),
        (8, 16),
        (9, 18),
        (10, 20),
        (11, 22),
    ]

    # Example measured positions (with some missing measurements represented as None)
    measured_positions = [
        (0.1, -0.1),
        (1.2, 2.1),
        None,
        (3.0, 6.2),
        (4.1, 8.1),
        None,
        (6.2, 12.1),
        (7.3, 14.2),
        None,
        (9.4, 18.3),
        (10.5, 20.4),
        (11.6, 22.5),
    ]

    # Example estimated positions (Kalman filter output)
    estimated_positions = [
        (0, 0),
        (1, 2),
        (2, 4.1),
        (3, 6),
        (4, 8.2),
        (5, 10.1),
        (6, 12.1),
        (7, 14),
        (8, 16.2),
        (9, 18.1),
        (10, 20),
        (11, 22.2),
    ]

    # print_2D_localization(model_positions, measured_positions, estimated_positions)
    # plot_2D_localization(model_positions, measured_positions, estimated_positions)
    animate_2D_localization(model_positions, measured_positions, estimated_positions, timestamps)



if __name__ == "__main__":
    main()
