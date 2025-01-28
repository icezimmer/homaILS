import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
from matplotlib.patches import Circle


def animate_2D_localization(model_positions, observed_positions, estimated_positions, timestamps,
                            interval=10, min_x=-100, max_x=100, min_y=-100, max_y=100): 
    """
    Animate using a small, fixed interval in FuncAnimation, while tracking an 
    internal "cumulative clock" to decide which frame index to display at a given time.
    This respects your real timestamps without needing 'start_event_loop' or draw events.

    :param model_positions: List of (x, y) or None, length = N
    :param observed_positions: List of (x, y) or None, length = N
    :param estimated_positions: List of (x, y) or None, length = N
    :param timestamps: List of strictly increasing times in ms, length = N
    :param interval: Interval in ms between frames
    """
    # Basic validation
    N = len(timestamps)
    if not (len(model_positions) == len(observed_positions) == len(estimated_positions) == N):
        raise ValueError("All input lists must have the same length.")
    if any(timestamps[i] >= timestamps[i + 1] for i in range(N - 1)):
        raise ValueError("Timestamps must be strictly increasing.")

    # --- Set up the figure and axes ---
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_title("2D Localization - Kalman Filter")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True)

    ax.set_xlim(min_x - 1, max_x + 1)
    ax.set_ylim(min_y - 1, max_y + 1)

    model_points, = ax.plot([], [], 'g.', label="Model")
    meas_points, = ax.plot([], [], 'r.', label="Observations")
    est_points, = ax.plot([], [], 'b.', label="Estimates")
    ax.legend()

    # We'll track how far we've progressed in the data
    current_index = 0

    # We'll also capture a "real" start time for the animation
    real_start_time = time.time()
    total_frames = N

    def update(_frame):
        """
        This function is called ~every 50 ms (or whatever interval you set), but
        we decide which 'current_index' to show based on how much real time has passed.
        """
        nonlocal current_index

        # How many real seconds have elapsed since we started?
        elapsed_sec = time.time() - real_start_time
        elapsed_ms = elapsed_sec * 1000.0

        # Convert real elapsed time to the "absolute" clock 
        # relative to the first timestamp.
        # If timestamps[0] = 1,000,000 ms, then 'current_abs_time'
        # is that plus how many ms have elapsed in real time.
        current_abs_time = timestamps[0] + elapsed_ms

        # We advance 'current_index' as long as the next timestamp is <= current_abs_time
        while (current_index < total_frames - 1 
               and timestamps[current_index + 1] <= current_abs_time):
            current_index += 1

        # Now 'current_index' is where we should be in the data at this moment.
        # Slice data up to current_index
        m_slice = [mp for mp in model_positions[:current_index + 1] if mp is not None]
        z_slice = [mp for mp in observed_positions[:current_index + 1] if mp is not None]
        e_slice = [mp for mp in estimated_positions[:current_index + 1] if mp is not None]

        # Update model trajectory
        if m_slice:
            xs, ys = zip(*m_slice)
            model_points.set_data(xs, ys)

        # Update observetions
        if z_slice:
            xm, ym = zip(*z_slice)
            meas_points.set_data(xm, ym)

        # Update estimates
        if e_slice:
            xe, ye = zip(*e_slice)
            est_points.set_data(xe, ye)

        # If we're at the last frame, we can choose to keep going or do something special
        if current_index >= total_frames - 1:
            # E.g., you could stop the animation or just let it run
            pass

        return model_points, meas_points, est_points

    # You can adjust 'interval' if you want more or less "smoothness" in the UI.
    ani = animation.FuncAnimation(
        fig, update, interval=interval, blit=False, repeat=False, save_count=total_frames
    )

    # Show the plot
    plt.show()


def animate_2D_localization_errors(model_positions, observed_positions, estimated_positions, 
                            model_errors, observed_errors, estimated_errors, timestamps,
                            interval=10, min_x=-100, max_x=100, min_y=-100, max_y=100): 
    """
    Animate a 2D localization process, adding uncertainty circles based on sqrt(trace(Q)).

    :param model_positions: List of (x, y) or None, length = N
    :param observed_positions: List of (x, y) or None, length = N
    :param estimated_positions: List of (x, y) or None, length = N
    :param model_errors: List of covariance matrices (2x2) or None, length = N
    :param observed_errors: List of covariance matrices (2x2) or None, length = N
    :param estimated_errors: List of covariance matrices (2x2) or None, length = N
    :param timestamps: List of strictly increasing times in ms, length = N
    :param interval: Interval in ms between frames
    """

    # Basic validation
    N = len(timestamps)
    if not (len(model_positions) == len(observed_positions) == len(estimated_positions) ==
            len(model_errors) == len(observed_errors) == len(estimated_errors) == N):
        raise ValueError("All input lists must have the same length.")
    if any(timestamps[i] >= timestamps[i + 1] for i in range(N - 1)):
        raise ValueError("Timestamps must be strictly increasing.")

    # --- Set up the figure and axes ---
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_title("2D Localization - Kalman Filter (Animated)")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True)

    ax.set_xlim(min_x - 1, max_x + 1)
    ax.set_ylim(min_y - 1, max_y + 1)

    model_points, = ax.plot([], [], 'g.', label="Model")
    meas_points, = ax.plot([], [], 'r.', label="Observations")
    est_points, = ax.plot([], [], 'b.', label="Estimates")
    ax.legend()

    # Containers for uncertainty circles
    model_circle = Circle((0, 0), 0, color='green', alpha=0.2)
    observed_circle = Circle((0, 0), 0, color='red', alpha=0.2)
    estimated_circle = Circle((0, 0), 0, color='blue', alpha=0.2)

    ax.add_patch(model_circle)
    ax.add_patch(observed_circle)
    ax.add_patch(estimated_circle)

    # Tracking variables
    current_index = 0
    real_start_time = time.time()
    total_frames = N

    def update(_frame):
        """
        Updates the animation frame-by-frame based on elapsed real-time.
        """
        nonlocal current_index

        # Compute elapsed real-world time
        elapsed_sec = time.time() - real_start_time
        elapsed_ms = elapsed_sec * 1000.0
        current_abs_time = timestamps[0] + elapsed_ms

        # Advance to the correct frame based on real time
        while (current_index < total_frames - 1 
               and timestamps[current_index + 1] <= current_abs_time):
            current_index += 1

        # Slice data up to the current frame
        m_slice = [mp for mp in model_positions[:current_index + 1] if mp is not None]
        z_slice = [mp for mp in observed_positions[:current_index + 1] if mp is not None]
        e_slice = [mp for mp in estimated_positions[:current_index + 1] if mp is not None]

        # Update model trajectory
        if m_slice:
            xs, ys = zip(*m_slice)
            model_points.set_data(xs, ys)

        # Update observations
        if z_slice:
            xm, ym = zip(*z_slice)
            meas_points.set_data(xm, ym)

        # Update estimates
        if e_slice:
            xe, ye = zip(*e_slice)
            est_points.set_data(xe, ye)

        # Update uncertainty circles
        if model_positions[current_index] is not None and model_errors[current_index] is not None:
            x, y = model_positions[current_index]
            radius = np.sqrt(np.trace(model_errors[current_index]))  # sqrt(trace(Q))
            model_circle.set_center((x, y))
            model_circle.set_radius(radius)

        if observed_positions[current_index] is not None and observed_errors[current_index] is not None:
            x, y = observed_positions[current_index]
            radius = np.sqrt(np.trace(observed_errors[current_index]))  # sqrt(trace(Q))
            observed_circle.set_center((x, y))
            observed_circle.set_radius(radius)

        if estimated_positions[current_index] is not None and estimated_errors[current_index] is not None:
            x, y = estimated_positions[current_index]
            radius = np.sqrt(np.trace(estimated_errors[current_index]))  # sqrt(trace(Q))
            estimated_circle.set_center((x, y))
            estimated_circle.set_radius(radius)

        return model_points, meas_points, est_points, model_circle, observed_circle, estimated_circle

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, interval=interval, blit=False, repeat=False, save_count=total_frames
    )

    plt.show()
