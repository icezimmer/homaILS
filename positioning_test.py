import numpy as np
from homaILS.modeling.linear import UniformLinearMotion
from homaILS.filtering.kalman import KalmanFilter

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Test the Kalman Filter with a Uniform Linear Motion model.')
    parser.add_argument('--dt', type=float, required=True, help='Time step.')
    parser.add_argument('--q', type=float, required=True, help='Process noise.')
    parser.add_argument('--r', type=float, required=True, help='Measurement noise.')
    parser.add_argument('--steps', type=int, required=True, help='Number of steps to run the simulation.')
    parser.add_argument('--measures', type=int, required=True, help='Number of steps between measurements.')
    return parser.parse_args()

def main():
    # Parameters
    args = parse_args()
    dt = args.dt
    q = args.q
    r = args.r
    steps = args.steps
    step_update = args.measures

    model = UniformLinearMotion(dt=dt, q=q, r=r)
    
    # Initialize the filter
    kf = KalmanFilter(model)
    
    # Initial state: Suppose we start at position = 0, velocity = 5 m/s
    x0 = np.array([[0], [0], [2], [3]])
    # Initial covariance matrix
    P0 = np.eye(4)*0.1
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
        kf.predict()

        # Update the Kalman Filter
        if t % step_update == 0:
            # Measured position with noise
            z = model_state[:2, 0] + np.random.normal(0, r, 2)
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
