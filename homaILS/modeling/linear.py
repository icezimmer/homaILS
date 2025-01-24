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

    def a_priori_state(self, x):
        """
        Compute the a priori state estimate given the current state estimate and control input.
        - Input:
            - x : np.array (State estimate at time n given observations up to and including at time m ≤ n).
        - Return:
            - x : np.array (New a priori state estimate at time n+1 given the old observations).
        """
        x = x.reshape(-1, 1)
        if self.u is None:
            x = self.F @ x
        else:
            u = self.u.reshape(-1, 1)
            x = self.F @ x + self.B @ u
        return x
    
    def a_priori_covariance(self, P):
        """
        Compute the a priori estimate covariance matrix.
        - Input:
            - P : np.array (Estimate covariance matrix at time n given observations up to and including at time m ≤ n).
        - Return:
            - P : np.array (New a priori estimate covariance matrix at time n+1 given the old observations).
        """
        P = self.F @ P @ self.F.T + self.Q
        return P
        

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
            - std_alpha: the standard deviation of the direction.
        - The dynamic parameters are:
            - alpha: the direction of the step.
        - The state is [x, y].
        - The observations are [x, y].
        - The state model is:
            - x_k = x_{k-1} + L*cos(alpha) + w_k[0]
            - y_k = y_{k-1} + L*sin(alpha) + w_k[1]
        - The observation model is:
            - x_k = x_{k-1} + v_k[0]
            - y_k = y_{k-1} + v_k[1]
        """
        # Force the user to provide the static parameters
        if 'std_L' not in static_params:
            raise ValueError("The standard deviation of the step length 'std_L' must be provided.")
        if 'std_alpha' not in static_params:
            raise ValueError("The standard deviation of the direction 'std_alpha' must be provided.")
        self.std_L = static_params['std_L']
        self.std_alpha = static_params['std_alpha']

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
    def compute_u(alpha):
        # Compute the control input vector
        u = np.array([np.cos(alpha), np.sin(alpha)]).reshape(-1, 1)
        return u
    
    @staticmethod
    def compute_Q(L, std_L, alpha, std_alpha):
        """
        Computes the covariance matrix Q using a canonical covariance matrix
        and rotating it with a rotation matrix.

        Parameters:
            L (float): Step size (length).
            std_L (float): Standard deviation of the step size.
            alpha (float): Heading angle in radians.
            std_alpha (float): Standard deviation of the heading angle.

        Returns:
            numpy.ndarray: 2x2 covariance matrix Q.
        """
        # Canonical covariance matrix
        sigma_x2 = std_L**2
        sigma_y2 = (L * np.sin(std_alpha))**2
        C = np.array([
            [sigma_x2, 0],
            [0, sigma_y2]
        ])
        
        # Rotation matrix
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)
        R = np.array([
            [cos_alpha, -sin_alpha],
            [sin_alpha,  cos_alpha]
        ])
        
        # Rotate the canonical covariance matrix
        Q = R @ C @ R.T
        
        return Q
    # def compute_Q(L, std_L, alpha, std_alpha):
    #     """
    #     Computes the covariance matrix Q for a step-heading model. The transition function is
    #     x_k = x_{k-1} + L*cos(alpha) + w_k[0]
    #     y_k = y_{k-1} + L*sin(alpha) + w_k[1]
    #     f = [x + L*cos(alpha), y + L*sin(alpha)]

    #     Parameters:
    #         L (float): Step size (length).
    #         std_L (float): Standard deviation of the step size.
    #         alpha (float): Heading angle in radians.
    #         std_alpha (float): Standard deviation of the heading angle.

    #     Returns:
    #         numpy.ndarray: 2x2 covariance matrix Q.
    #     """
    #     # Jacobian matrix of f respect to L and alpha
    #     G = np.array([
    #         [np.cos(alpha), -L * np.sin(alpha)],
    #         [np.sin(alpha),  L * np.cos(alpha)]
    #     ])

    #     # Covariance matrix of L and alpha
    #     Sigma_u = np.array([
    #         [std_L**2, 0],
    #         [0, std_alpha**2]
    #     ])

    #     # Compute Q = G * Sigma_u * G.T
    #     Q = G @ Sigma_u @ G.T

    #     return Q
    # def compute_Q(L, std_L, alpha, std_alpha):
    #     # Compute dx and dy
    #     std_x = (L + std_L) - (L - std_L) * np.cos(std_alpha)
    #     std_y = 2 * (L + std_L) * np.sin(std_alpha)

    #     # Variance
    #     sigma_x2 = std_x**2
    #     sigma_y2 = std_y**2

    #     cos_alpha = np.cos(alpha)
    #     sin_alpha = np.sin(alpha)
    #     # Compute the rotation matrix
    #     rotation_matrix = np.array([[cos_alpha, -sin_alpha],
    #                                 [sin_alpha, cos_alpha]])
    #     # Canonical covariance matrix
    #     cov_matrix = np.array([
    #         [sigma_x2, 0],
    #         [0, sigma_y2]
    #     ])
    #     # Rotate the canonical covariance matrix
    #     Q = rotation_matrix @ cov_matrix @ rotation_matrix.T

    #     return Q
    
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
        if 'alpha' not in dynamic_params:
            raise ValueError("The direction 'alpha' must be provided.")
        L = dynamic_params['L']
        alpha = dynamic_params['alpha']

        self.B = self.compute_B(L)
        self.u = self.compute_u(alpha)
        self.Q = self.compute_Q(L, self.std_L, alpha, self.std_alpha)
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
    

# class StepCountHeading(LinearModel):
#     def __init__(self, **static_params):
#         """
#         Initialize the Step Heading model.
#         - The static parameters are:
#             - r: the observation noise;
#             - L: the mean step length;
#             - std_L: the standard deviation of the step length;
#             - std_alpha: the standard deviation of the direction.
#         - The dynamic parameters are:
#             - alpha: the direction of the step.
#             - num_steps: the number of steps.
#         - The state is [x, y].
#         - The observations are [x, y].
#         """
#         # Force the user to provide the static parameters
#         if 'r' not in static_params:
#             raise ValueError("The observation noise 'r' must be provided.")
#         if 'L' not in static_params:
#             raise ValueError("The mean step length 'L' must be provided.")
#         if 'std_L' not in static_params:
#             raise ValueError("The standard deviation of the step length 'std_L' must be provided.")
#         if 'std_alpha' not in static_params:
#             raise ValueError("The standard deviation of the direction 'std_alpha' must be provided.")
#         # Take the static parameters
#         r = static_params['r']
#         self.L = static_params['L']
#         self.std_L = static_params['std_L']
#         self.std_alpha = static_params['std_alpha']

#         # Transition model parameters
#         self.F = np.array([[1, 0],
#                            [0, 1]])
#         self.B = np.eye(2)*self.L
#         self.u = None
#         self.Q = None

#         # Observation model parameters
#         self.H = np.eye(2)
#         self.R = np.eye(2)*r
    
#     @staticmethod
#     def compute_u(num_steps, alpha):
#         # Compute the control input vector
#         u = np.array([num_steps * np.cos(alpha), num_steps * np.sin(alpha)]).reshape(-1, 1)
#         return u
    
#     @staticmethod
#     def compute_Q(L, std_L, alpha, std_alpha):

#         # Compute dx and dy
#         dx = (L + std_L) - (L - std_L) * np.cos(std_alpha)
#         dy = 2 * (L + std_L) * np.sin(std_alpha)

#         # Variance
#         sigma_x2 = dx**2
#         sigma_y2 = dy**2

#         cos_alpha = np.cos(alpha)
#         sin_alpha = np.sin(alpha)
#         # Compute the rotation matrix
#         rotation_matrix = np.array([[cos_alpha, -sin_alpha],
#                                     [sin_alpha, cos_alpha]])
#         # Canonical covariance matrix
#         cov_matrix = np.array([
#             [sigma_x2, 0],
#             [0, sigma_y2]
#         ])
#         # Rotate the canonical covariance matrix
#         Q = rotation_matrix @ cov_matrix @ rotation_matrix.T

#         return Q
    
#     def get_transition_model(self, **dynamic_params):
#         """
#         Return the parameters of the Step Heading state model for Kalman Filter.
#         - Inputs:
#             - dynamic_params : dict, Dynamic parameters for the state model.
#         - Return:
#             - F, B, u, Q : np.array, np.array, np.array, np.array.
#         """
#         # Force the user to provide the dynamic parameters
#         if 'num_steps' not in dynamic_params:
#             raise ValueError("The number of steps 'num_steps' must be provided.")
#         if 'alpha' not in dynamic_params:
#             raise ValueError("The direction 'alpha' must be provided.")
#         # Take the dynamic parameters
#         num_steps = dynamic_params['num_steps']
#         alpha = dynamic_params['alpha']

#         self.u = self.compute_u(num_steps, alpha)
#         self.Q = self.compute_Q(self.L, self.std_L, alpha, self.std_alpha)
#         return self.F, self.B, self.u, self.Q


# class StepHeadingPlus(LinearModel):
#     def __init__(self, **static_params):
#         """
#         Initialize the Step Heading model.
#         - The static parameters are:
#             - std_L: the standard deviation of the step length;
#             - std_alpha: the standard deviation of the direction.
#         - The dynamic parameters are:
#             - alpha: the direction of the step.
#         - The state is [x, y, alpha].
#         - The observations are [x, y, alpha].
#         - The state model is:

#         """
#         # Force the user to provide the static parameters
#         if 'std_L' not in static_params:
#             raise ValueError("The standard deviation of the step length 'std_L' must be provided.")
#         if 'std_alpha' not in static_params:
#             raise ValueError("The standard deviation of the direction 'std_alpha' must be provided.")
#         self.std_L = static_params['std_L']
#         self.std_alpha = static_params['std_alpha']

#         # Transition model parameters
#         self.F = np.array([[1, 0, 0],
#                            [0, 1, 0],
#                            [0, 0, 0]])
#         self.B = None
#         self.u = None
#         self.Q = None

#         # Observation model parameters
#         self.H = np.eye(3)
#         self.R = None

#     @staticmethod
#     def compute_B(L):
#         B = np.array([[L, 0, 0],
#                       [0, L, 0],
#                       [0, 0, 1]])
#         return B
    
#     @staticmethod
#     def compute_u(alpha):
#         # Compute the control input vector
#         u = np.array([np.cos(alpha), np.sin(alpha), alpha]).reshape(-1, 1)
#         return u
    
#     @staticmethod
#     def compute_Q(L, std_L, alpha, std_alpha):
#         # Compute dx and dy
#         std_x = (L + std_L) - (L - std_L) * np.cos(std_alpha)
#         std_y = 2 * (L + std_L) * np.sin(std_alpha)

#         # Variance
#         sigma_x2 = std_x**2
#         sigma_y2 = std_y**2
#         sigma_alpha2 = std_alpha**2

#         cos_alpha = np.cos(alpha)
#         sin_alpha = np.sin(alpha)
#         # Compute the rotation matrix
#         rotation_matrix = np.array([[cos_alpha, -sin_alpha, 0],
#                                     [sin_alpha, cos_alpha, 0],
#                                     [0, 0, 1]])
        
#         # Canonical covariance matrix
#         cov_matrix = np.array([
#             [sigma_x2, 0, 0],
#             [0, sigma_y2, 0],
#             [0, 0, sigma_alpha2]
#         ])

#         # Rotate the canonical covariance matrix
#         Q = rotation_matrix @ cov_matrix @ rotation_matrix.T

#         return Q
    
#     def get_transition_model(self, **dynamic_params):
#         """
#         Return the parameters of the Step Heading state model for Kalman Filter.
#         - Inputs:
#             - dynamic_params : dict, Dynamic parameters for the state model.
#         - Return:
#             - F, B, u, Q : np.array, np.array, np.array, np.array.
#         """
#         # Force the user to provide the dynamic parameters
#         if 'L' not in dynamic_params:
#             raise ValueError("The mean step length 'L' must be provided.")
#         if 'alpha' not in dynamic_params:
#             raise ValueError("The direction 'alpha' must be provided.")
#         L = dynamic_params['L']
#         alpha = dynamic_params['alpha']

#         self.B = self.compute_B(L)
#         self.u = self.compute_u(alpha)
#         self.Q = self.compute_Q(L, self.std_L, alpha, self.std_alpha)
#         return self.F, self.B, self.u, self.Q
    
#     @staticmethod
#     def compute_R(std_horizontal, std_bearing):
#         R = np.array([
#             [std_horizontal**2, 0, 0],
#             [0, std_horizontal**2, 0],
#             [0, 0, std_bearing**2]
#         ])
#         return R

#     def get_observation_model(self, **dynamic_params):
#         """
#         Return the parameters of the Step Heading observation model for Kalman Filter.
#         - Inputs:
#             - dynamic_params : dict, Dynamic parameters for the observation model.
#         - Return:
#             - H, R : np.array, np.array.
#         """
#         # Force the user to provide the dynamic parameters
#         if 'std_horizontal' not in dynamic_params:
#             raise ValueError("The deviation of the horizontal observation 'std_horizontal' must be provided.")
#         if 'std_bearing' not in dynamic_params:
#             raise ValueError("The deviation of the bearing observation 'std_bearing' must be provided.")
#         std_horizontal = dynamic_params['std_horizontal']
#         std_bearing = dynamic_params['std_bearing']

#         self.R = self.compute_R(std_horizontal, std_bearing)
#         return self.H, self.R


# class StepHeadingSpeed(LinearModel):
#     def __init__(self, **static_params):
#         """
#         Initialize the Step Heading model.
#         - The static parameters are:
#             - std_L: the standard deviation of the step length;
#             - std_alpha: the standard deviation of the direction.
#         - The dynamic parameters are:
#             - alpha: the direction of the step.
#         - The state is [x, y, vx, vy].
#         - The observations are [x, y].
#         """
#         # Force the user to provide the static parameters
#         if 'std_L' not in static_params:
#             raise ValueError("The standard deviation of the step length 'std_L' must be provided.")
#         if 'std_alpha' not in static_params:
#             raise ValueError("The standard deviation of the direction 'std_alpha' must be provided.")
#         # Take the static parameters
#         r = static_params['r']
#         self.L = static_params['L']
#         self.std_L = static_params['std_L']
#         self.std_alpha = static_params['std_alpha']

#         # Set the state model parameters
#         self.F = np.eye(4)
#         self.B = None
#         self.u = None
#         self.Q = None

#         # Set the observation model parameters
#         self.H = np.eye(4)
#         self.R = None

#     @staticmethod
#     def compute_B(L, dt):
#         return np.array([[L, 0],
#                          [0, L],
#                          [L/dt, 0],
#                          [0, L/dt]])

#     @staticmethod
#     def compute_Q(L, std_L, alpha, std_alpha, dt):
#         """
#         Compute the process noise covariance matrix Q.
#         - State increment:
#             - f(L, alpha) = [
#             L*cos(alpha),
#             L*sin(alpha),
#             (L/dt)*cos(alpha),
#             (L/dt)*sin(alpha)
#         ]
#         - Cov(L, alpha) = diag(std_L^2, std_alpha^2).
#         - Q = J_f * Cov(L,alpha) * J_f^T.
#         """

#         # Jacobian J_f wrt [L, alpha].
#         J = np.array([
#             [ np.cos(alpha),      -L*np.sin(alpha)     ],
#             [ np.sin(alpha),       L*np.cos(alpha)     ],
#             [ np.cos(alpha)/dt,  -(L/dt)*np.sin(alpha) ],
#             [ np.sin(alpha)/dt,   (L/dt)*np.cos(alpha) ]
#         ])

#         # Covariance of [L, alpha].
#         cov_matrix = np.diag([std_L**2, std_alpha**2])

#         # Q = J_f * Cov(L,alpha) * J_f^T
#         Q = J @ cov_matrix @ J.T
#         return Q
    
#     @staticmethod
#     def compute_u(alpha):
#         # Compute the control input vector
#         u = np.array([np.cos(alpha), np.sin(alpha)]).reshape(-1, 1)
#         return u
    
#     def get_transition_model(self, **dynamic_params):
#         """
#         Return the parameters of the Step Heading state model for Kalman Filter.
#         - Inputs:
#             - dynamic_params : dict, Dynamic parameters for the state model.
#         - Return:
#             - F, B, u, Q : np.array, np.array, np.array, np.array.
#         """
#         # Force the user to provide the dynamic parameters
#         if 'L' not in dynamic_params:
#             raise ValueError("The mean step length 'L' must be provided.")
#         if 'alpha' not in dynamic_params:
#             raise ValueError("The direction 'alpha' must be provided.")
#         if 'dt' not in dynamic_params:
#             raise ValueError("The time step 'dt' must be provided.")
#         L = dynamic_params['L']
#         alpha = dynamic_params['alpha']
#         dt = dynamic_params['dt']

#         self.B = self.compute_B(L, dt)
#         self.u = self.compute_u(alpha)
#         self.Q = self.compute_Q(L, self.std_L, alpha, self.std_alpha, dt)
#         return self.F, self.B, self.u, self.Q
    
#     @staticmethod
#     def compute_R(std_r):
#         R = np.eye(2)*(std_r**2)
#         return R
    
#     def get_observation_model(self, **dynamic_params):
#         """
#         Return the parameters of the Step Heading observation model for Kalman Filter.
#         - Inputs:
#             - dynamic_params : dict, Dynamic parameters for the observation model.
#         - Return:
#             - H, R : np.array, np.array.
#         """
#         # Force the user to provide the dynamic parameters
#         if 'std_r' not in dynamic_params:
#             raise ValueError("The deviation of the observation 'std_r' must be provided.")
#         std_r = dynamic_params['std_r']

#         self.R = self.compute_R(std_r)
#         return self.H, self.R
    