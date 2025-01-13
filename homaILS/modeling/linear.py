import numpy as np

# Define a superclass for linear models
class LinearModel:
    def __init__(self, **static_params):
        """
        Initialize the Linear Model.
        x_{k+1} = F*x_k + B*u_k + w_k
        z_{k+1} = H*x_{k+1} + v_k.
        """
        # Force the user to provide the static parameters
        self.F = None
        self.H = None
        self.Q = None
        self.R = None
        self.B = None
        self.u = None

    def get_params(self, **dynamic_params):
        """
        Return the parameters of the model for Kalman Filter.
        """
        # Force the user to provide the dynamic parameters
        return self.F, self.H, self.Q, self.R, self.B, self.u

    def step(self, x):
        """
        Compute the next state given the current state and control input,
        without updating the state estimate.

        - Input:
            - x : np.array (State vector).
        """
        if self.u is None:
            return self.F @ x
        else:
            return self.F @ x + self.B @ self.u
        

class UniformLinearMotion(LinearModel):
    def __init__(self, **static_params):
        """
        Initialize the Uniform Linear Motion model, with a costant velocity and zero acceleration in 2D-space.
        - The static parameters are:
            - dt: the time step;
            - q: the process noise;
            - r: the measurement noise.
        - The state is [x, y, vx, vy].
        - The observations are only the positions [x, y].
        """
        # Force the user to provide the static parameters
        if 'dt' not in static_params:
            raise ValueError("The time step 'dt' must be provided.")
        if 'q' not in static_params:
            raise ValueError("The process noise 'q' must be provided.")
        if 'r' not in static_params:
            raise ValueError("The measurement noise 'r' must be provided.")
        # Take the static parameters
        dt = static_params['dt']
        q = static_params['q']
        r = static_params['r']

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
        self.u = None


class UniformLinearMotionSpeedObs(LinearModel):
    def __init__(self, **static_params):
        """
        Initialize the Uniform Linear Motion model, with a costant velocity and zero acceleration in 2D-space.
        - The static parameters are:
            - dt: the time step;
            - q: the process noise;
            - r: the measurement noise.
        - The state is [x, y, vx, vy].
        - The observations are the positions and velocities [x, y, vx, vy].
        """
        # Force the user to provide the static parameters
        if 'dt' not in static_params:
            raise ValueError("The time step 'dt' must be provided.")
        if 'q' not in static_params:
            raise ValueError("The process noise 'q' must be provided.")
        if 'r' not in static_params:
            raise ValueError("The measurement noise 'r' must be provided.")
        # Take the static parameters
        dt = static_params['dt']
        q = static_params['q']
        r = static_params['r']

        # Set parameters for kalman filter
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.eye(4)
        self.Q = np.eye(4)*q
        self.R = np.eye(4)*r
        self.B = np.zeros((4, 2))
        self.u = None
    

class StepCountHeading(LinearModel):
    def __init__(self, **static_params):
        """
        Initialize the Step Heading model.
        - The static parameters are:
            - r: the measurement noise;
            - L: the mean step length;
            - dL: the standard deviation of the step length;
            - dalpha: the standard deviation of the direction.
        - The dynamic parameters are:
            - alpha: the direction of the step.
            - num_steps: the number of steps.
        - The state is [x, y].
        - The observations are [x, y].
        """
        # Force the user to provide the static parameters
        if 'r' not in static_params:
            raise ValueError("The measurement noise 'r' must be provided.")
        if 'L' not in static_params:
            raise ValueError("The mean step length 'L' must be provided.")
        if 'dL' not in static_params:
            raise ValueError("The standard deviation of the step length 'dL' must be provided.")
        if 'dalpha' not in static_params:
            raise ValueError("The standard deviation of the direction 'dalpha' must be provided.")
        # Take the static parameters
        r = static_params['r']
        self.L = static_params['L']
        self.dL = static_params['dL']
        self.dalpha = static_params['dalpha']

        # Set parameters for kalman filter
        self.F = np.array([[1, 0],
                           [0, 1]])
        self.H = np.eye(2)
        self.Q = None
        self.R = np.eye(2)*r
        self.B = np.eye(2)*self.L
        self.u = None

    @staticmethod
    def compute_Q(L, dL, alpha, dalpha):

        # Compute dx and dy
        dx = (L + dL) - (L - dL) * np.cos(dalpha)
        dy = 2 * (L + dL) * np.sin(dalpha)

        # Variance
        sigma_x2 = dx**2
        sigma_y2 = dy**2

        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        # Compute the rotation matrix
        rotation_matrix = np.array([[cos_alpha, -sin_alpha],
                                    [sin_alpha, cos_alpha]])
        # Canonical covariance matrix
        cov_matrix = np.array([
            [sigma_x2, 0],
            [0, sigma_y2]
        ])
        # Rotate the canonical covariance matrix
        Q = rotation_matrix @ cov_matrix @ rotation_matrix.T

        return Q
    
    @staticmethod
    def compute_u(num_steps, alpha):
        # Compute the control input vector
        u = np.array([num_steps * np.cos(alpha), num_steps * np.sin(alpha)]).reshape(-1, 1)
        return u
    
    def get_params(self, **dynamic_params):
        """
        Return the parameters of the Step Heading model for Kalman Filter.
        """
        # Force the user to provide the dynamic parameters
        if 'num_steps' not in dynamic_params:
            raise ValueError("The number of steps 'num_steps' must be provided.")
        if 'alpha' not in dynamic_params:
            raise ValueError("The direction 'alpha' must be provided.")
        # Take the dynamic parameters
        num_steps = dynamic_params['num_steps']
        alpha = dynamic_params['alpha']

        self.Q = self.compute_Q(self.L, self.dL, alpha, self.dalpha)
        self.u = self.compute_u(num_steps, alpha)
        return self.F, self.H, self.Q, self.R, self.B, self.u


class StepHeading(LinearModel):
    def __init__(self, **static_params):
        """
        Initialize the Step Heading model.
        - The static parameters are:
            - r: the measurement noise;
            - L: the mean step length;
            - dL: the standard deviation of the step length;
            - dalpha: the standard deviation of the direction.
        - The dynamic parameters are:
            - alpha: the direction of the step.
        - The state is [x, y].
        - The observations are [x, y].
        """
        # Force the user to provide the static parameters
        if 'r' not in static_params:
            raise ValueError("The measurement noise 'r' must be provided.")
        if 'L' not in static_params:
            raise ValueError("The mean step length 'L' must be provided.")
        if 'dL' not in static_params:
            raise ValueError("The standard deviation of the step length 'dL' must be provided.")
        if 'dalpha' not in static_params:
            raise ValueError("The standard deviation of the direction 'dalpha' must be provided.")
        # Take the static parameters
        r = static_params['r']
        self.L = static_params['L']
        self.dL = static_params['dL']
        self.dalpha = static_params['dalpha']

        # Set parameters for kalman filter
        self.F = np.array([[1, 0],
                           [0, 1]])
        self.H = np.eye(2)
        self.Q = None
        self.R = np.eye(2)*r
        self.B = np.eye(2)*self.L
        self.u = None

    @staticmethod
    def compute_Q(L, dL, alpha, dalpha):

        # Compute dx and dy
        dx = (L + dL) - (L - dL) * np.cos(dalpha)
        dy = 2 * (L + dL) * np.sin(dalpha)

        # Variance
        sigma_x2 = dx**2
        sigma_y2 = dy**2

        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        # Compute the rotation matrix
        rotation_matrix = np.array([[cos_alpha, -sin_alpha],
                                    [sin_alpha, cos_alpha]])
        # Canonical covariance matrix
        cov_matrix = np.array([
            [sigma_x2, 0],
            [0, sigma_y2]
        ])
        # Rotate the canonical covariance matrix
        Q = rotation_matrix @ cov_matrix @ rotation_matrix.T

        return Q
    
    @staticmethod
    def compute_u(alpha):
        # Compute the control input vector
        u = np.array([np.cos(alpha), np.sin(alpha)]).reshape(-1, 1)
        return u
    
    def get_params(self, **dynamic_params):
        """
        Return the parameters of the Step Heading model for Kalman Filter.
        """
        # Force the user to provide the dynamic parameters
        if 'alpha' not in dynamic_params:
            raise ValueError("The direction 'alpha' must be provided.")
        alpha = dynamic_params['alpha']
        self.Q = self.compute_Q(self.L, self.dL, alpha, self.dalpha)
        self.u = self.compute_u(alpha)
        return self.F, self.H, self.Q, self.R, self.B, self.u
    

class StepHeadingSpeed(LinearModel):
    def __init__(self, **static_params):
        """
        Initialize the Step Heading model.
        - The static parameters are:
            - r: the measurement noise;
            - L: the mean step length;
            - dL: the standard deviation of the step length;
            - dalpha: the standard deviation of the direction.
        - The dynamic parameters are:
            - alpha: the direction of the step.
        - The state is [x, y, vx, vy].
        - The observations are [x, y].
        """
        # Force the user to provide the static parameters
        if 'r' not in static_params:
            raise ValueError("The measurement noise 'r' must be provided.")
        if 'L' not in static_params:
            raise ValueError("The mean step length 'L' must be provided.")
        if 'dL' not in static_params:
            raise ValueError("The standard deviation of the step length 'dL' must be provided.")
        if 'dalpha' not in static_params:
            raise ValueError("The standard deviation of the direction 'dalpha' must be provided.")
        # Take the static parameters
        r = static_params['r']
        self.L = static_params['L']
        self.dL = static_params['dL']
        self.dalpha = static_params['dalpha']

        # Set static parameters for kalman filter
        self.F = np.eye(4)
        self.H = np.eye(4)
        self.Q = None
        self.R = np.eye(4)*r
        self.B = None
        self.u = None

    @staticmethod
    def compute_B(L, dt):
        return np.array([[L, 0],
                         [0, L],
                         [L/dt, 0],
                         [0, L/dt]])

    @staticmethod
    def compute_Q(L, dL, alpha, dalpha, dt):
        """
        Compute the process noise covariance matrix Q.
        - State increment:
            - f(L, alpha) = [
            L*cos(alpha),
            L*sin(alpha),
            (L/dt)*cos(alpha),
            (L/dt)*sin(alpha)
        ]
        - Cov(L, alpha) = diag(dL^2, dAlpha^2).
        - Q = J_f * Cov(L,alpha) * J_f^T.
        """

        # Jacobian J_f wrt [L, alpha].
        J = np.array([
            [ np.cos(alpha),      -L*np.sin(alpha)     ],
            [ np.sin(alpha),       L*np.cos(alpha)     ],
            [ np.cos(alpha)/dt,  -(L/dt)*np.sin(alpha) ],
            [ np.sin(alpha)/dt,   (L/dt)*np.cos(alpha) ]
        ])

        # Covariance of [L, alpha].
        cov_matrix = np.diag([dL**2, dalpha**2])

        # Q = J_f * Cov(L,alpha) * J_f^T
        Q = J @ cov_matrix @ J.T
        return Q
    
    @staticmethod
    def compute_u(alpha):
        # Compute the control input vector
        u = np.array([np.cos(alpha), np.sin(alpha)]).reshape(-1, 1)
        return u
    
    def get_params(self, **dynamic_params):
        """
        Return the parameters of the Step Heading model for Kalman Filter.
        """
        # Force the user to provide the dynamic parameters
        if 'alpha' not in dynamic_params:
            raise ValueError("The direction 'alpha' must be provided.")
        if 'dt' not in dynamic_params:
            raise ValueError("The time step 'dt' must be provided.")
        alpha = dynamic_params['alpha']
        dt = dynamic_params['dt']

        self.B = self.compute_B(self.L, dt)
        self.Q = self.compute_Q(self.L, self.dL, alpha, self.dalpha, dt)
        self.u = self.compute_u(alpha)
        return self.F, self.H, self.Q, self.R, self.B, self.u
    