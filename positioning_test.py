import numpy as np
from homaILS.modeling.linear import UniformLinearMotion
from homaILS.filtering.kalman import KalmanFilter

def main():
    # Parameters
    dt = 0.1    # time step
    q = 0.001   # process noise
    r = 0.5     # measurement noise
    steps = 100
    step_update = 5

    F, H, Q, R, P, B = UniformLinearMotion(dt=dt, q=q, r=r).get_params()
    
    # Initialize the filter
    kf = KalmanFilter(F=F, H=H, Q=Q, R=R, P=P, B=B)
    
    # Initial state: Suppose we start at position = 0, velocity = 1 m/s
    x0 = np.array([[0], [1], [0], [1]])
    kf.initialize_state(x0)

    # model initial state (x=0, vx=1 m/s, y=0, vy=0.5 m/s)
    model_state = x0

    # Store results for analysis
    model_positions = []
    measured_positions = []
    estimated_positions = []

    for t in range(steps):
        # model state evolves
        model_state = kf.model_step(model_state)

        # Measured position with noise
        measured_x = model_state[0, 0] + np.random.normal(0, np.sqrt(r))
        measured_y = model_state[2, 0] + np.random.normal(0, np.sqrt(r))
        z = np.array([[measured_x],
                      [measured_y]])

        # Kalman prediction and update
        kf.predict()

        if t % step_update == 0:
            kf.update(z)

        # Logging for analysis
        model_positions.append((model_state[0, 0], model_state[2, 0]))
        measured_positions.append((measured_x, measured_y))
        estimated_positions.append((kf.x[0,0], kf.x[2,0]))

    # Print results
    for i in range(steps):
        print(f"Step {i+1}:")
        print(f"  Model Position: x={model_positions[i][0]:.2f}, y={model_positions[i][1]:.2f}")
        print(f"  Measured Pos:   x={measured_positions[i][0]:.2f}, y={measured_positions[i][1]:.2f}")
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
