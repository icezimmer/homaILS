import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='2D Gaussian Distribution with Declination')
    parser.add_argument('--length', type=float, required=True, help='Length in meters of the step size')
    parser.add_argument('--dlength', type=float, required=True, help='Change in meters of the step size')
    parser.add_argument('--angle', type=float, required=True, help='Angle in radians')
    parser.add_argument('--dangle', type=float, required=True, help='Angle change in radians')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    L = args.length
    dL = args.dlength
    alpha = args.angle
    dalpha = args.dangle

    # Set the new direction
    alpha = np.pi/2 - alpha

    # Compute dx and dy
    dx = (L + dL) - (L - dL) * np.cos(dalpha)
    dy = 2 * (L + dL) * np.sin(dalpha)

    # Variance
    sigma_x2 = dx**2
    sigma_y2 = dy**2

    # Means
    mu_x = L * np.cos(alpha)
    mu_y = L * np.sin(alpha)

    # Create a grid
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)

    # Compute the covariance matrix
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)
    cov_matrix = np.array([
        [sigma_x2 * cos_alpha**2 + sigma_y2 * sin_alpha**2,
        (sigma_x2 - sigma_y2) * cos_alpha * sin_alpha],
        [(sigma_x2 - sigma_y2) * cos_alpha * sin_alpha,
        sigma_x2 * sin_alpha**2 + sigma_y2 * cos_alpha**2]
    ])

    # Compute the Gaussian
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    det_cov_matrix = np.linalg.det(cov_matrix)

    # Evaluate the Gaussian at each grid point
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            v = np.array([X[i, j] - mu_x, Y[i, j] - mu_y])
            Z[i, j] = (1 / (2 * np.pi * np.sqrt(det_cov_matrix))) * \
                    np.exp(-0.5 * v.T @ inv_cov_matrix @ v)

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Probability Density')
    plt.title('2D Gaussian Distribution with Declination (Angle α)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.scatter(mu_x, mu_y, color='black', label='Mean (x, y)')
    plt.scatter(0, 0, color='red', label='Origin (0, 0)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()