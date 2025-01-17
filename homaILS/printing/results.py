def print_2D_localization(model_positions, observed_positions, estimated_positions):
    """
    Print the 2D localization results.
    """
    # Check if the lengths of the lists are the same
    if not (len(model_positions) == len(observed_positions) == len(estimated_positions)):
        raise ValueError("Lengths of the input lists are not the same")
    
    # Print results
    for i in range(len(model_positions)):
        print(f"Step {i+1}:")

        if model_positions[i] is not None:
            p = model_positions[i].reshape(-1, 1)
            print(f"  Model Position: x={p[0, 0]:.2f}, y={p[1, 0]:.2f}")
        if observed_positions[i] is not None:
            p = observed_positions[i].reshape(-1, 1)
            print(f"  Observed Pos:  x={p[0, 0]:.2f}, y={p[1, 0]:.2f}")
        if estimated_positions[i] is not None:
            p = estimated_positions[i].reshape(-1, 1)
            print(f"  Estimated Pos:  x={p[0, 0]:.2f}, y={p[1, 0]:.2f}")
        print("------------------------------------------------")
