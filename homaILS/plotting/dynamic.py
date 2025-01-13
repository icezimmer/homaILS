#TODO: Repair the code
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

def animate_2D_localization(model_positions, measured_positions, estimated_positions, timestamps):
    """
    Animate using a small, fixed interval in FuncAnimation, while tracking an 
    internal "cumulative clock" to decide which frame index to display at a given time.
    This respects your real timestamps without needing 'start_event_loop' or draw events.

    :param model_positions: List of (x, y) or None, length = N
    :param measured_positions: List of (x, y) or None, length = N
    :param estimated_positions: List of (x, y) or None, length = N
    :param timestamps: List of strictly increasing times in ms, length = N
    """
    # Basic validation
    N = len(timestamps)
    if not (len(model_positions) == len(measured_positions) == len(estimated_positions) == N):
        raise ValueError("All input lists must have the same length.")
    if any(timestamps[i] >= timestamps[i + 1] for i in range(N - 1)):
        raise ValueError("Timestamps must be strictly increasing.")

    # --- Set up the figure and axes ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_title("2D Localization - Kalman Filter")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True)

    # Determine axis limits (simple example)
    # Filter out None from model_positions to compute limits
    valid_models = [mp for mp in model_positions if mp is not None]
    min_x = min(x for x, y in valid_models)
    max_x = max(x for x, y in valid_models)
    min_y = min(y for x, y in valid_models)
    max_y = max(y for x, y in valid_models)

    ax.set_xlim(min_x - 1, max_x + 1)
    ax.set_ylim(min_y - 1, max_y + 1)

    model_line, = ax.plot([], [], 'g.', label="Model")
    meas_points, = ax.plot([], [], 'r.', label="Measurements")
    est_line, = ax.plot([], [], 'b.', label="Estimates")
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
        z_slice = [mp for mp in measured_positions[:current_index + 1] if mp is not None]
        e_slice = [mp for mp in estimated_positions[:current_index + 1] if mp is not None]

        # Update model trajectory
        if m_slice:
            xs, ys = zip(*m_slice)
            model_line.set_data(xs, ys)

        # Update measurements
        if z_slice:
            xm, ym = zip(*z_slice)
            meas_points.set_data(xm, ym)

        # Update estimates
        if e_slice:
            xe, ye = zip(*e_slice)
            est_line.set_data(xe, ye)

        # If we're at the last frame, we can choose to keep going or do something special
        if current_index >= total_frames - 1:
            # E.g., you could stop the animation or just let it run
            pass

        return model_line, meas_points, est_line

    # We'll call 'update' every 50 ms (20 times per second).
    # You can adjust 'interval' if you want more or less "smoothness" in the UI.
    ani = animation.FuncAnimation(
        fig, update, interval=50, blit=False, repeat=False, save_count=total_frames
    )

    # Show the plot
    plt.show()
