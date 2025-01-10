# If desired, you can also plot the results using matplotlib:
import matplotlib.pyplot as plt


def plot_2D_localization(model_positions, measured_positions, estimated_positions):
    """
    Plot the 2D localization results using matplotlib.
    """
    # Check if the lengths of the lists are the same
    if not (len(model_positions) == len(measured_positions) == len(estimated_positions)):
        raise ValueError("Lengths of the input lists are not the same")

    # Remove None values
    model_positions = [m for m in model_positions if m is not None]
    estimated_positions = [m for m in estimated_positions if m is not None]
    measured_positions = [m for m in measured_positions if m is not None]

    model_xs, model_ys = zip(*model_positions)
    meas_xs, meas_ys = zip(*measured_positions)
    est_xs, est_ys = zip(*estimated_positions)

    plt.figure(figsize=(10,5))
    plt.plot(model_xs, model_ys, 'g-', label='Model Trajectory')
    plt.plot(meas_xs, meas_ys, 'r.', label='Measurements')
    plt.plot(est_xs, est_ys, 'b-', label='Kalman Estimates')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('2D Localization - Kalman Filter')
    plt.legend()
    plt.grid(True)
    plt.show()
