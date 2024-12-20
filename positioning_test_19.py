import pandas as pd
import numpy as np
from homaILS.modeling.linear import StepHeading
from homaILS.filtering.kalman import KalmanFilter


#TODO: repaire the code
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Test the Kalman Filter with a Uniform Linear Motion model.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--dt', type=float, required=True, help='Time step.')
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

    # Merge the orientation and step data on the nearest timestamp, keeping the timestamp from the step data
    df = pd.merge_asof(step_df, orientation_df, on='Timestamp', direction='nearest')
    print("\nStep Data and Orientation:")
    print(df)

    args = parse_args()
    dt = args.dt

    # Compute the time differences
    df['Time_diff'] = df['Timestamp'].diff()

    # Create a group id that increments whenever a gap larger than dt is encountered
    df['group_id'] = (df['Time_diff'] > dt * 1000).cumsum()

    # Now group by this group_id and compute the mean azimuth and count
    df = df.groupby('group_id').agg(
        alpha=('Azimuth', 'mean'),
        dk=('Azimuth', 'count')
    ).reset_index()
    print(df)

    # Compute the number of rows in the step data
    steps = len(df)
    step_update = 10000

    # Gropup the data by the delta time and compute the mean of the step lengths and alphas
    step_lengths = [104 / 164, 181 / 300, 111 / 176, 226 / 337, 62 / 100]
    # Mean and standard deviation of the step lengths
    L = sum(step_lengths) / len(step_lengths)
    dL = (sum([(l - L) ** 2 for l in step_lengths]) / len(step_lengths)) ** 0.5

    alphas = orientation_df['Azimuth'].values
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
    
    # Initial state: Suppose we start at position = 0, velocity = 5 m/s
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

    for t in range(steps):
        # Model state evolves
        model_state = kf.model.step(model_state)

        # Predict the next state
        a = alphas[t]
        kf.predict(a)

        # Update the Kalman Filter
        if t % step_update == 0:
            # Measured position with noise
            z = model_state[:2, 0] + np.random.normal(0, args.r, 2)
            kf.update(z)
            measured_positions.append(z)

        # Logging for analysis
        model_positions.append(model_state[:2, 0])
        estimated_positions.append(kf.x[:2, 0])

    # Print results
    j = 0
    for i in range(steps):
        print(f"Step {i+1}:")
        print(f"  Model Position: x={model_positions[i][0]:.2f}, y={model_positions[i][1]:.2f}")
        if i % step_update == 0:
            print(f"  Measured Pos:   x={measured_positions[j][0]:.2f}, y={measured_positions[j][1]:.2f}")
            j += 1
        print(f"  Estimated Pos:  x={estimated_positions[i][0]:.2f}, y={estimated_positions[i][1]:.2f}")
        print("------------------------------------------------")

    # If desired, you can also plot the results using matplotlib:
    import matplotlib.pyplot as plt
    model_xs, model_ys = zip(*model_positions)
    meas_xs, meas_ys = zip(*measured_positions)
    est_xs, est_ys = zip(*estimated_positions)

    plt.figure(figsize=(10,5))
    plt.plot(model_xs, model_ys, 'g-', label='Model Trajectory')
    plt.plot(meas_xs, meas_ys, 'r.', label='Measurements')
    plt.plot(est_xs, est_ys, 'b-', label='Kalman Estimates')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Uniform Linear Motion - Kalman Filter')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
