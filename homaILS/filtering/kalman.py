import numpy as np

class KalmanFilter:
    def __init__(self, model):
        """
        Initialize the Kalman Filter.
        - Inputs:
            - model : object, The model object for the Kalman Filter.
        """
        self.model = model

        # State covariance and state estimate will be set later
        self.P = None
        self.x = None
    
    def initialize(self, x0, P0):
        """
        Set the initial state estimate and estimate error covariance.
        - Inputs:
            - x0 : np.array, Initial state vector.
            - P0 : np.array, Initial covariance matrix. Must match the state vector size.
        """
        self.x = x0.reshape(-1, 1)

        if P0.shape[0] != P0.shape[1]:
            raise ValueError("Covariance matrix P0 must be square.")
        if P0.shape[0] != self.x.shape[0]:
            raise ValueError("Covariance matrix P0 size must match state vector size.")
        self.P = P0

    @staticmethod
    def a_priori_state(x, F, B, u):
        """
        Compute the a priori state estimate given the current state estimate and control input.
        - Input:
            - x : np.array (State estimate at time n given observations up to and including at time m ≤ n).
            - F : np.array, Transition matrix.
            - B : np.array, Control matrix.
            - u : np.array, Control input vector.
        - Return:
            - x : np.array, New a priori state estimate at time n+1 given the old observations.
        """
        x = x.reshape(-1, 1)
        if u is None:
            x = F @ x
        else:
            u = u.reshape(-1, 1)
            x = F @ x + B @ u
        return x
    
    @staticmethod
    def a_priori_covariance(P, F, Q):
        """
        Compute the a priori estimate covariance matrix.
        - Input:
            - P : np.array, Estimate covariance matrix at time n given observations up to and including at time m ≤ n.
            - F : np.array, Transition matrix.
            - Q : np.array, Process noise covariance matrix.
        - Return:
            - P : np.array, New a priori estimate covariance matrix at time n+1 given the old observations.
        """
        P = F @ P @ F.T + Q
        return P

    def predict(self, **dynamic_params):
        """
        Predict the next state and covariance.
        - Inputs:
            - dynamic_params : dict, Dynamic parameters for the transition model. Must match the model's get_transition_model() method.
        """
        if self.x is None:
            raise ValueError("State is not initialized. Use initialize() method.")
        
        # Get the state model parameters updated with dynamic parameters
        F, B, u, Q = self.model.get_transition_model(**dynamic_params)
               
        # Predict the state
        self.x = self.a_priori_state(self.x, F, B, u)
        
        # Predict the error covariance
        self.P = self.a_priori_covariance(self.P, F, Q)

    @staticmethod
    def residual(z, H, x):
        """
        Compute the residual (innovation) between the measurement and the predicted measurement.
        - Inputs:
            - z : np.array, Measurement vector.
            - H : np.array, Observation matrix.
            - x : np.array, State estimate.
        - Return:
            - y : np.array, Residual vector.
        """
        y = z - H @ x
        return y
    
    @staticmethod
    def innovation_covariance(H, P, R):
        """
        Compute the innovation covariance.
        - Inputs:
            - H : np.array, Observation matrix.
            - P : np.array, Estimate covariance matrix.
            - R : np.array, Measurement noise covariance matrix.
        - Return:
            - S : np.array, Innovation covariance matrix.
        """
        S = H @ P @ H.T + R
        return S
    
    @staticmethod
    def kalman_gain(P, H, S):
        """
        Compute the Kalman gain.
        - Inputs:
            - P : np.array, Estimate covariance matrix.
            - H : np.array, Observation matrix.
            - S : np.array, Innovation covariance matrix.
        - Return:
            - K : np.array, Kalman gain matrix.
        """
        K = P @ H.T @ np.linalg.inv(S)
        return K
    
    @staticmethod
    def a_posteriori_state(x, K, y):
        """
        Compute the a posteriori state estimate.
        - Inputs:
            - x : np.array, State estimate.
            - K : np.array, Kalman gain matrix.
            - y : np.array, Residual vector.
        - Return:
            - x : np.array, A posteriori state estimate.
        """
        x = x + K @ y
        return x
    
    @staticmethod
    def a_posteriori_covariance(P, K, H):
        """
        Compute the a posteriori estimate covariance matrix.
        - Inputs:
            - P : np.array, Estimate covariance matrix.
            - K : np.array, Kalman gain matrix.
            - H : np.array, Observation matrix.
        - Return:
            - P : np.array, A posteriori estimate covariance matrix.
        """
        I = np.eye(P.shape[0])
        P = (I - K @ H) @ P
        return P

    def update(self, z, **dynamic_params):
        """
        Update the Kalman Filter with a new observation z.
        - Inputs:
            - z : np.array, Measurement vector.
            - dynamic_params : dict, Dynamic parameters for the observation model. Must match the model's get_observation_model() method.
        """
        if self.x is None or self.P is None:
            raise ValueError("State is not initialized. Use initialize_state() method.")
        
        z = z.reshape(-1, 1)
        
        # Get the observation model parameters
        H, R = self.model.get_observation_model(**dynamic_params)
        
        # Compute the innovation (residual) y
        y = self.residual(z, H, self.x)
        
        # Compute the innovation covariance S
        S = self.innovation_covariance(H, self.P, R)
        
        # Compute the Kalman gain K
        K = self.kalman_gain(self.P, H, S)
        
        # Update the state estimate
        self.x = self.a_posteriori_state(self.x, K, y)
        
        # Update the estimate covariance
        self.P = self.a_posteriori_covariance(self.P, K, H)
