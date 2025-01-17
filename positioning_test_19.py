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
    parser.add_argument('--obs', type=int, required=True, help='Number of steps between observations.')
    parser.add_argument('--std_r', type=float, required=True, help='Measurement noise.')
    return parser.parse_args()

#TODO: Implement the main function
def main():
    acceleration_path = "data/17_12_2024/20241217T133655-19-Acceleration.csv"
    GPS_path = "data/17_12_2024/20241217T133655-19-GPS.csv"
    orientation_path = "data/17_12_2024/20241217T133655-19-Orientation.csv"
    step_path = "data/17_12_2024/20241217T133655-19-Step.csv"

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
    std_L = (sum([(l - L) ** 2 for l in step_lengths]) / len(step_lengths)) ** 0.5
    alphas = df['Azimuth'].values
    # Mean and standard deviation of the alphas
    alpha = alphas.mean()
    std_alpha = alphas.std()
    print(f"\nStep Length mean (std): {L:.2f} ({std_L:.2f})")
    print(f"Alpha mean (std): {alpha:.2f} ({std_alpha:.2f})")

     # Seed for reproducibility
    np.random.seed(args.seed)

    model = StepHeading(std_L=std_L, std_alpha=std_alpha)
    
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
    observed_positions = []
    estimated_positions = []
    timestamps = df['Timestamp'].values

    steps = len(df)
    step_update = args.obs
    for t in range(steps):
        # Model state evolves
        model_state = kf.model.step(model_state)

        # Predict the next state
        kf.predict(alpha=alphas[t], L=L)

        # Update the Kalman Filter
        if t % step_update == 0:
            # Measured position with noise
            z = model_state[:2, 0] + np.random.normal(0, args.std_r, 2)
            kf.update(z, std_r=args.std_r)
            observed_positions.append(z)
        else:
            observed_positions.append(None)

        # Logging for analysis
        model_positions.append(model_state[:2, 0])
        estimated_positions.append(kf.x[:2, 0])

    # print_2D_localization(model_positions, observed_positions, estimated_positions)
    # plot_2D_localization(model_positions, observed_positions, estimated_positions)
    animate_2D_localization(model_positions, observed_positions, estimated_positions, timestamps)



if __name__ == "__main__":
    main()
