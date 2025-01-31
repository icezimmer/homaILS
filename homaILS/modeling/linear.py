import numpy as np

# Define a superclass for linear models
class LinearModel:
    def __init__(self, **static_params):
        """
        Initialize the Linear Model.
        x_k = F*x_{k-1} + B*u_k + w_k
        z_k = H*x_k + v_k.
        """
        # Force the user to provide the static parameters

        # Transtion model parameters
        self.F = None
        self.B = None
        self.u = None
        self.Q = None

        # Observation model parameters
        self.H = None
        self.R = None

    def get_transition_model(self, **dynamic_params):
        """
        Return the transition model for Kalman Filter.
        Input:
            - dynamic_params : dict, Dynamic parameters for the model.
        Return:
            - F, B, u, Q : np.array, np.array, np.array, np.array.
        """
        # Force the user to provide the dynamic parameters
        return self.F, self.B, self.u, self.Q

    def get_observation_model(self, **dynamic_params):
        """
        Return the observation model for Kalman Filter.
        - Input:
            - dynamic_params : dict, Dynamic parameters for the model.
        - Return:
            - H, R : np.array, np.array.
        """
        # Force the user to provide the dynamic parameters
        return self.H, self.R
        

class UniformLinearMotion(LinearModel):
    def __init__(self, **static_params):
        """
        Initialize the Uniform Linear Motion model, with a costant velocity and zero acceleration in 2D-space.
        - The static parameters are:
            - dt: the time step;
            - std_q: the process noise.
            - std_r: the observation noise.
        - The state is [x, y, vx, vy].
        - The observations are only the positions [x, y].
        - The state model is:
            - x_k = x_{k-1} + vx_{k-1}*dt + w_k[0]
            - y_k = y_{k-1} + vy_{k-1}*dt + w_k[1]
            - vx_k = vx_{k-1} + w_k[2]
            - vy_k = vy_{k-1} + w_k[3]
        - The observation model is:
            - x_k = x_{k-1} + v_k[0]
            - y_k = y_{k-1} + v_k[1]
        """
        # Force the user to provide the static parameters
        if 'dt' not in static_params:
            raise ValueError("The time step 'dt' must be provided.")
        if 'std_q' not in static_params:
            raise ValueError("The process noise 'std_q' must be provided.")
        if 'std_r' not in static_params:
            raise ValueError("The observation noise 'std_r' must be provided.")
        # Take the static parameters
        dt = static_params['dt']
        std_q = static_params['std_q']
        std_r = static_params['std_r']

        # Transition model parameters
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.B = np.zeros((4, 2))
        self.u = None
        self.Q = np.eye(4)*(std_q**2)

        # Observation model parameters
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.R = np.eye(2)*(std_r**2)


class UniformLinearMotionSpeedObs(LinearModel):
    def __init__(self, **static_params):
        """
        Initialize the Uniform Linear Motion model, with a costant velocity and zero acceleration in 2D-space.
        - The static parameters are:
            - dt: the time step;
            - std_q: the process noise;
            - std_r: the observation noise.
        - The state is [x, y, vx, vy].
        - The observations are the positions and velocities [x, y, vx, vy].
        - The state model is:
            - x_k = x_{k-1} + vx_{k-1}*dt + w_k[0]
            - y_k = y_{k-1} + vy_{k-1}*dt + w_k[1]
            - vx_k = vx_{k-1} + w_k[2]
            - vy_k = vy_{k-1} + w_k[3]
        - The observation model is:
            - x_k = x_{k-1} + vx_{k-1}*dt + v_k[0]
            - y_k = y_{k-1} + vy_{k-1}*dt + v_k[1]
            - vx_k = vx_{k-1} + v_k[2]
            - vy_k = vy_{k-1} + v_k[3]
        """
        # Force the user to provide the static parameters
        if 'dt' not in static_params:
            raise ValueError("The time step 'dt' must be provided.")
        if 'std_q' not in static_params:
            raise ValueError("The process noise 'std_q' must be provided.")
        if 'std_r' not in static_params:
            raise ValueError("The observation noise 'std_r' must be provided.")
        # Take the static parameters
        dt = static_params['dt']
        std_q = static_params['std_q']
        std_r = static_params['std_r']

        # Transition model parameters
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.B = np.zeros((4, 2))
        self.u = None
        self.Q = np.eye(4)*(std_q**2)

        # Observation model parameters
        self.H = np.eye(4)
        self.R = np.eye(4)*(std_r**2)


class StepHeading(LinearModel):
    def __init__(self, **static_params):
        """
        Initialize the Step Heading model.
        - The static parameters are:
            - std_L: the standard deviation of the step length;
            - std_theta: the standard deviation of the direction.
        - The dynamic parameters are:
            - theta: the direction of the step.
        - The state is [x, y].
        - The observations are [x, y].
        - The state model is:
            - x_k = x_{k-1} + L*cos(theta) + w_k[0]
            - y_k = y_{k-1} + L*sin(theta) + w_k[1]
        - The observation model is:
            - x_k = x_{k-1} + v_k[0]
            - y_k = y_{k-1} + v_k[1]
        """
        # Force the user to provide the static parameters
        if 'std_L' not in static_params:
            raise ValueError("The standard deviation of the step length 'std_L' must be provided.")
        if 'std_theta' not in static_params:
            raise ValueError("The standard deviation of the direction 'std_theta' must be provided.")
        self.std_L = static_params['std_L']
        self.std_theta = static_params['std_theta']

        # Transition model parameters
        self.F = np.array([[1, 0],
                           [0, 1]])
        self.B = None
        self.u = None
        self.Q = None

        # Observation model parameters
        self.H = np.eye(2)
        self.R = None

    @staticmethod
    def compute_B(L):
        B = np.eye(2)*L
        return B
    
    @staticmethod
    def compute_u(theta):
        # Compute the control input vector
        u = np.array([np.cos(theta), np.sin(theta)]).reshape(-1, 1)
        return u
    
    @staticmethod
    def compute_Q(L, std_L, theta, std_theta):
        """
        Computes the covariance matrix Q using a canonical covariance matrix
        and rotating it with a rotation matrix.

        Parameters:
            L (float): Step size (length).
            std_L (float): Standard deviation of the step size.
            theta (float): Heading angle in radians.
            std_theta (float): Standard deviation of the heading angle.

        Returns:
            numpy.ndarray: 2x2 covariance matrix Q.
        """
        # Canonical covariance matrix
        sigma_x2 = std_L**2
        sigma_y2 = (L * np.sin(std_theta))**2
        C = np.array([
            [sigma_x2, 0],
            [0, sigma_y2]
        ])
        
        # Rotation matrix
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        R = np.array([
            [cos_theta, -sin_theta],
            [sin_theta,  cos_theta]
        ])
        
        # Rotate the canonical covariance matrix
        Q = R @ C @ R.T
        
        return Q
    
    def get_transition_model(self, **dynamic_params):
        """
        Return the parameters of the Step Heading state model for Kalman Filter.
        - Inputs:
            - dynamic_params : dict, Dynamic parameters for the state model.
        - Return:
            - F, B, u, Q : np.array, np.array, np.array, np.array.
        """
        # Force the user to provide the dynamic parameters
        if 'L' not in dynamic_params:
            raise ValueError("The mean step length 'L' must be provided.")
        if 'theta' not in dynamic_params:
            raise ValueError("The direction 'theta' must be provided.")
        L = dynamic_params['L']
        theta = dynamic_params['theta']

        self.B = self.compute_B(L)
        self.u = self.compute_u(theta)
        self.Q = self.compute_Q(L, self.std_L, theta, self.std_theta)
        return self.F, self.B, self.u, self.Q
    
    @staticmethod
    def compute_R(std_r):
        R = np.eye(2)*(std_r**2)
        return R

    def get_observation_model(self, **dynamic_params):
        """
        Return the parameters of the Step Heading observation model for Kalman Filter.
        - Inputs:
            - dynamic_params : dict, Dynamic parameters for the observation model.
        - Return:
            - H, R : np.array, np.array.
        """
        # Force the user to provide the dynamic parameters
        if 'std_r' not in dynamic_params:
            raise ValueError("The deviation of the observation 'std_r' must be provided.")
        std_r = dynamic_params['std_r']

        self.R = self.compute_R(std_r)
        return self.H, self.R
    