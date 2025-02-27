# If desired, you can also plot the results using matplotlib:
import matplotlib.pyplot as plt
import folium
from pyproj import Proj, Transformer
import numpy as np
import webbrowser
from homaILS.processing.geographic import localutm_to_geodetic
import numpy as np
from matplotlib.patches import Circle


def plot_2D_localization(model_positions, observed_positions, estimated_positions):
    """
    Plot the 2D localization results using matplotlib.
    """
    # Check if the lengths of the lists are the same
    if not (len(model_positions) == len(observed_positions) == len(estimated_positions)):
        raise ValueError("Lengths of the input lists are not the same")

    # Remove None values
    model_positions = [m for m in model_positions if m is not None]
    estimated_positions = [m for m in estimated_positions if m is not None]
    observed_positions = [m for m in observed_positions if m is not None]

    model_xs, model_ys = zip(*model_positions)
    meas_xs, meas_ys = zip(*observed_positions)
    est_xs, est_ys = zip(*estimated_positions)

    plt.figure(figsize=(9,9))
    # Plot if they are not empty
    if model_positions:
        plt.plot(model_xs, model_ys, 'g.', label='Model Trajectory')
    if observed_positions:
        plt.plot(meas_xs, meas_ys, 'r.', label='Observations')
    if estimated_positions:
        plt.plot(est_xs, est_ys, 'b.', label='Kalman Estimates')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('2D Localization - Kalman Filter')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_2D_localization_errors(model_positions, observed_positions, estimated_positions, model_errors, observed_errors, estimated_errors):
    """
    Plot the 2D localization results using simplified circular uncertainty visualization.
    """
    # Check if the lengths of the lists are the same
    if not (len(model_positions) == len(observed_positions) == len(estimated_positions) == len(model_errors) == len(observed_errors) == len(estimated_errors)):
        raise ValueError("Lengths of the input lists are not the same")

    # Remove None values
    model_positions = [m for m in model_positions if m is not None]
    estimated_positions = [m for m in estimated_positions if m is not None]
    observed_positions = [m for m in observed_positions if m is not None]
    model_errors = [m for m in model_errors if m is not None]
    observed_errors = [m for m in observed_errors if m is not None]
    estimated_errors = [m for m in estimated_errors if m is not None]

    fig, ax = plt.subplots(figsize=(9, 9))

    # Plot points
    if model_positions:
        model_xs, model_ys = zip(*model_positions)
        ax.plot(model_xs, model_ys, 'g.', label='Model Trajectory')
    if observed_positions:
        obs_xs, obs_ys = zip(*observed_positions)
        ax.plot(obs_xs, obs_ys, 'r.', label='Observations')
    if estimated_positions:
        est_xs, est_ys = zip(*estimated_positions)
        ax.plot(est_xs, est_ys, 'b.', label='Kalman Estimates')

    def plot_uncertainty_circle(position, cov, color, ax):
        radius = np.sqrt(np.trace(cov) / 2)
        circle = Circle(position, radius, color=color, alpha=0.2, fill=True)
        ax.add_patch(circle)

    if model_errors:
        for pos, cov in zip(model_positions, model_errors):
            plot_uncertainty_circle(pos, cov, 'green', ax)

    if observed_errors:
        for pos, cov in zip(observed_positions, observed_errors):
            plot_uncertainty_circle(pos, cov, 'red', ax)

    if estimated_errors:
        for pos, cov in zip(estimated_positions, estimated_errors):
            plot_uncertainty_circle(pos, cov, 'blue', ax)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('2D Localization - Uncertainty')
    ax.legend()
    ax.grid(True)
    plt.show()


def map_2D_localization(model_positions, observed_positions, estimated_positions, lon0_deg, lat0_deg, utm_zone, northern_hemisphere):
    """
    Plots the 2D localization results on an interactive map using Folium.
    """

    # Convert positions from UTM to lat/lon
    def convert_positions(positions):
        """
        Convert a list of UTM coordinates to latitude/longitude, handling None values and NumPy arrays safely.
        """
        converted = []
        for pos in positions:
            if pos is None:  # Skip None values
                continue
            if isinstance(pos, np.ndarray):  # Convert NumPy array to tuple
                pos = tuple(pos)
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                try:
                    converted.append(localutm_to_geodetic(pos[0], pos[1], lon0_deg, lat0_deg, utm_zone, northern_hemisphere))
                except Exception as e:
                    print(f"Warning: Skipping invalid position {pos} - {e}")
            else:
                print(f"Warning: Invalid position format {pos}")
        return converted

    model_latlon = convert_positions(model_positions)
    observed_latlon = convert_positions(observed_positions)
    estimated_latlon = convert_positions(estimated_positions)

    # Create a folium map
    map_ = folium.Map(location=[lat0_deg, lon0_deg], zoom_start=19)

    # Plot model trajectory
    for lon, lat in model_latlon:
        folium.CircleMarker([lat, lon], color='green', radius=1, fill=True, fill_color='green', fill_opacity=0.7).add_to(map_)

    # Plot observations
    for lon, lat in observed_latlon:
        folium.CircleMarker([lat, lon], color='red', radius=1, fill=True, fill_color='red', fill_opacity=0.7).add_to(map_)

    # Plot estimated positions
    for lon, lat in estimated_latlon:
        folium.CircleMarker([lat, lon], color='blue', radius=1, fill=True, fill_color='blue', fill_opacity=0.7).add_to(map_)

    # Add a legend
    legend_html = """
    <div style="position: fixed; top: 50px; right: 50px; z-index:9999; font-size:14px; background-color:white; padding: 10px; border: 2px solid black;">
    <p><span style="color:green">Model Trajectory</span></p>
    <p><span style="color:red">Observations</span></p>
    <p><span style="color:blue">Kalman Estimates</span></p>
    </div>
    """
    map_.get_root().html.add_child(folium.Element(legend_html))

    map_.save('2D_Localization_Map.html')
    webbrowser.open(url='2D_Localization_Map.html', new=1)


# TODO: Check radii on the map, they seems wrong
def map_2D_localization_errors(model_positions, observed_positions, estimated_positions,
                               model_errors, observed_errors, estimated_errors,
                               lon0_deg, lat0_deg, utm_zone, northern_hemisphere):
    """
    Plots the 2D localization results with uncertainty on an interactive map using Folium.
    """
    # Convert positions from UTM to lat/lon
    def convert_positions(positions):
        converted = []
        for pos in positions:
            if pos is None:
                continue
            if isinstance(pos, np.ndarray):
                pos = tuple(pos)
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                try:
                    converted.append(localutm_to_geodetic(pos[0], pos[1], lon0_deg, lat0_deg, utm_zone, northern_hemisphere))
                except Exception as e:
                    print(f"Warning: Skipping invalid position {pos} - {e}")
            else:
                print(f"Warning: Invalid position format {pos}")
        return converted
    
    model_latlon = convert_positions(model_positions)
    observed_latlon = convert_positions(observed_positions)
    estimated_latlon = convert_positions(estimated_positions)

    # Convert errors from meters (UTM) to degrees for accurate map plotting
    def compute_radii(errors, lat):
        meters_per_degree = 111320 * np.cos(np.radians(lat))  # Adjust for latitude
        return [np.sqrt(np.trace(cov) / 2) / meters_per_degree if cov is not None else 0 for cov in errors]
    
    model_radii = compute_radii(model_errors, lat0_deg)
    observed_radii = compute_radii(observed_errors, lat0_deg)
    estimated_radii = compute_radii(estimated_errors, lat0_deg)

    # Create a folium map
    map_ = folium.Map(location=[lat0_deg, lon0_deg], zoom_start=19)

    # Plot points with uncertainty
    for (lon, lat), radius in zip(model_latlon, model_radii):
        folium.Circle([lat, lon], radius=radius, color='green', fill=True, fill_color='green', fill_opacity=0.2).add_to(map_)
    
    for (lon, lat), radius in zip(observed_latlon, observed_radii):
        folium.Circle([lat, lon], radius=radius, color='red', fill=True, fill_color='red', fill_opacity=0.2).add_to(map_)
    
    for (lon, lat), radius in zip(estimated_latlon, estimated_radii):
        folium.Circle([lat, lon], radius=radius, color='blue', fill=True, fill_color='blue', fill_opacity=0.2).add_to(map_)

    # Add a legend
    legend_html = """
    <div style="position: fixed; top: 50px; right: 50px; z-index:9999; font-size:14px; background-color:white; padding: 10px; border: 2px solid black;">
    <p><span style="color:green">Model Trajectory (with uncertainty)</span></p>
    <p><span style="color:red">Observations (with uncertainty)</span></p>
    <p><span style="color:blue">Kalman Estimates (with uncertainty)</span></p>
    </div>
    """
    map_.get_root().html.add_child(folium.Element(legend_html))

    map_.save('2D_Localization_Map_with_Errors.html')
    webbrowser.open(url='2D_Localization_Map_with_Errors.html', new=1)
