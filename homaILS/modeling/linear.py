import numpy as np

class UniformLinearMotion:
    def __init__(self, dt, q, r):
        """
        Initialize the Uniform Linear Motion model, with a costant velocity and zero acceleration in 2D-space.
        The state is [x, vx, y, vy].
        The observations are only the positions [x, y].
        Input:
        - dt: the time step;
        - q: the process noise;
        - r: the measurement noise.
        """
        # Set parameters for kalman filter
        self.F = np.array([[1, dt, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, dt],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])
        self.Q = np.eye(4)*q
        self.R = np.eye(2)*r
        self.P = np.eye(4)
        self.B = None

    def get_params(self):
        """Return the parameters of the Uniform Linear Motion model for Kalman Filter."""
        return self.F, self.H, self.Q, self.R, self.P, self.B


class UniformLinearMotion2:
    def __init__(self, dt, q, r):
        """
        Initialize the Uniform Linear Motion model, with a costant velocity and zero acceleration in 2D-space.
        The state is [x, vx, y, vy].
        The observations are the positions and velocities [x, vx, y, vy].
        Input:
        - dt: the time step;
        - q: the process noise;
        - r: the measurement noise.
        """
        # Set parameters for kalman filter
        self.F = np.array([[1, dt, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, dt],
                           [0, 0, 0, 1]])
        self.H = np.eye(4)
        self.Q = np.eye(4)*q
        self.R = np.eye(4)*r
        self.P = np.eye(4)
        self.B = None

    def get_params(self):
        """Return the parameters of the Uniform Linear Motion model for Kalman Filter."""
        return self.F, self.H, self.Q, self.R, self.P, self.B
    

class StepHeading:
    def __init__(self, dt, L, dL, alpha, dalpha, r):
        """
        Initialize the Step Heading model. The input has number of new steps and direction.
        The step size is L.
        The state is [x, vx, y, vy].
        The observations are [x, vx, y, vy].
        """
        # Set parameters for kalman filter
        self.F = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.eye(4)
        self.Q = self.compute_covariance_process_noise(L, dL, alpha, dalpha)
        self.R = np.eye(2)*r
        self.P = np.eye(4)
        self.B = np.array([[L, 0],
                           [0, L],
                           [L/dt, 0],
                           [0, L/dt]])
        
    @staticmethod
    def compute_covariance_process_noise(L, dL, alpha, dalpha):
        # Set the new direction
        alpha = np.pi/2 - alpha

        # Compute dx and dy
        dx = (L + dL) - (L - dL) * np.cos(dalpha)
        dy = 2 * (L + dL) * np.sin(dalpha)

        # Variance
        sigma_x2 = dx**2
        sigma_y2 = dy**2

        # Compute the covariance matrix
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        cov_matrix = np.array([
            [sigma_x2 * cos_alpha**2 + sigma_y2 * sin_alpha**2,
            (sigma_x2 - sigma_y2) * cos_alpha * sin_alpha],
            [(sigma_x2 - sigma_y2) * cos_alpha * sin_alpha,
            sigma_x2 * sin_alpha**2 + sigma_y2 * cos_alpha**2]
        ])

        return cov_matrix
    
    def get_params(self):
        """Return the parameters of the Step Heading model for Kalman Filter."""
        return self.F, self.H, self.Q, self.R, self.P, self.B
    