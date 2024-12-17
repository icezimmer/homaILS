import pandas as pd


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

    dt = 0.6

    # Compute the time differences
    df['Time_diff'] = df['Timestamp'].diff()

    # Create a group id that increments whenever a gap larger than dt is encountered
    df['group_id'] = (df['Time_diff'] > dt * 1000).cumsum()

    # Now group by this group_id and compute the mean azimuth and count
    df = df.groupby('group_id').agg(
        alpha=('Azimuth', 'mean'),
        dk=('Azimuth', 'count')
    ).reset_index()
    print(dt)

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

if __name__ == "__main__":
    main()
