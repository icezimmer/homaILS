import numpy as np
from homaILS.modeling.linear import UniformLinearMotion
from homaILS.filtering.kalman import KalmanFilter
from homaILS.plotting.static import plot_2D_localization
from homaILS.printing.results import print_2D_localization

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Test the Kalman Filter with a Uniform Linear Motion model.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--dt', type=float, required=True, help='Time step.')
    parser.add_argument('--q', type=float, required=True, help='Process noise.')
    parser.add_argument('--r', type=float, required=True, help='Measurement noise.')
    parser.add_argument('--steps', type=int, required=True, help='Number of steps to run the simulation.')
    parser.add_argument('--obs', type=int, required=True, help='Number of steps between observations.')
    return parser.parse_args()

def main():
    # Parameters
    args = parse_args()
    dt = args.dt
    q = args.q
    r = args.r
    steps = args.steps
    step_update = args.obs

    # Seed for reproducibility
    np.random.seed(args.seed)

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
    observed_positions = []
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
            observed_positions.append(z)
        else:
            observed_positions.append(None)

        # Logging for analysis
        model_positions.append(model_state[:2, 0])
        estimated_positions.append(kf.x[:2, 0])

    print_2D_localization(model_positions, observed_positions, estimated_positions)
    plot_2D_localization(model_positions, observed_positions, estimated_positions)

if __name__ == "__main__":
    main()
