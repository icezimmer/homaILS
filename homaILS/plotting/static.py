# If desired, you can also plot the results using matplotlib:
import matplotlib.pyplot as plt
import folium
from pyproj import Proj, Transformer
import numpy as np
import webbrowser
from homaILS.processing.geographic import localutm_to_geodetic


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

    # Determine the map center
    if model_latlon:
        center_lon, center_lat = model_latlon[0]
    else:
        raise ValueError("Model positions list is empty; cannot determine map center.")

    # Create a folium map
    map_ = folium.Map(location=[center_lat, center_lon], zoom_start=19)

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
    webbrowser.open('2D_Localization_Map.html')
