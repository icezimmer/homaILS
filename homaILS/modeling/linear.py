import numpy as np

# Define a superclass for linear models
class LinearModel:
    def __init__(self, **static_params):
        self.F = None
        self.H = None
        self.Q = None
        self.R = None
        self.B = None

    def step(self, x, u=None):
        """
        Compute the next state given the current state and control input,
        without updating the state estimate.

        Parameters:
        x : np.array
            State vector.
        u : np.array (optional)
            Control input vector.
        """
        if u is None:
            return self.F @ x
        else:
            return self.F @ x + self.B @ u

    def get_params(self, **dynamic_params):
        """Return the parameters of the model for Kalman Filter."""
        return self.F, self.H, self.Q, self.R, self.B

class UniformLinearMotion(LinearModel):
    def __init__(self, **static_params):
        """
        Initialize the Uniform Linear Motion model, with a costant velocity and zero acceleration in 2D-space.
        The state is [x, y, vx, vy].
        The observations are only the positions [x, y].
        Input:
        - dt: the time step;
        - q: the process noise;
        - r: the measurement noise.
        """
        dt = static_params.get('dt', 0.1)
        q = static_params.get('q', 0.001)
        r = static_params.get('r', 0.5)

        # Set parameters for kalman filter
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = np.eye(4)*q
        self.R = np.eye(2)*r
        self.B = np.zeros((4, 2))


class UniformLinearMotionSpeedObs(LinearModel):
    def __init__(self, **static_params):
        """
        Initialize the Uniform Linear Motion model, with a costant velocity and zero acceleration in 2D-space.
        The state is [x, y, vx, vy].
        The observations are the positions and velocities [x, y, vx, vy].
        Input:
        - dt: the time step;
        - q: the process noise;
        - r: the measurement noise.
        """
        dt = static_params.get('dt', 0.1)
        q = static_params.get('q', 0.001)
        r = static_params.get('r', 0.5)

        # Set parameters for kalman filter
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.eye(4)
        self.Q = np.eye(4)*q
        self.R = np.eye(4)*r
        self.B = np.zeros((4, 2))
    

class StepHeading(LinearModel):
    def __init__(self, **static_params):
        """
        Initialize the Step Heading model. The input has number of new steps and direction.
        The step size is L.
        The state is [x, y].
        The observations are [x, y].
        """
        r = static_params.get('r', 0.5)
        self.L = static_params.get('L', 0.6)
        self.dL = static_params.get('dL', 0.1)
        self.dalpha = static_params.get('dalpha', np.pi/36)

        # Set parameters for kalman filter
        self.F = np.array([[1, 0],
                           [0, 1]])
        self.H = np.eye(2)
        self.Q = None
        self.R = np.eye(2)*r
        self.B = np.eye(2)*self.L
        
    @staticmethod
    def compute_Q(L, dL, alpha, dalpha):
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
    
    def get_params(self, **dynamic_params):
        """Return the parameters of the Step Heading model for Kalman Filter."""
        alpha = dynamic_params.get('alpha', 0)

        self.Q = self.compute_Q(self.L, self.dL, alpha, self.dalpha)
        return self.F, self.H, self.Q, self.R, self.B
    