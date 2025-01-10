def print_2D_localization(model_positions, measured_positions, estimated_positions):
    """
    Print the 2D localization results.
    """
    # Check if the lengths of the lists are the same
    if not (len(model_positions) == len(measured_positions) == len(estimated_positions)):
        raise ValueError("Lengths of the input lists are not the same")
    
    # Print results
    for i in range(len(model_positions)):
        print(f"Step {i+1}:")
        if model_positions[i] is not None:
            print(f"  Model Position: x={model_positions[i][0]:.2f}, y={model_positions[i][1]:.2f}")
        if measured_positions[i] is not None:
            print(f"  Measured Pos:   x={measured_positions[i][0]:.2f}, y={measured_positions[i][1]:.2f}")
        if estimated_positions[i] is not None:
            print(f"  Estimated Pos:  x={estimated_positions[i][0]:.2f}, y={estimated_positions[i][1]:.2f}")
        print("------------------------------------------------")
