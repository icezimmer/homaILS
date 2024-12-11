import numpy as np

class UniformLinearMotion:
    def __init__(self, dt, q, r):
        """
        Initialize the Uniform Linear Motion model, with a costant velocity and zero acceleration in 2D-space.
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
