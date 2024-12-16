import numpy as np

class KalmanFilter:
    def __init__(self, model):
        """
        Initialize the Kalman Filter.

        Parameters:
        model : object
            The model object that contains the Kalman Filter matrices:
            F, H, Q, R, and optionally B (control input matrix).
        """
        self.model = model

        # State covariance and state estimate will be set later
        self.P = None
        self.x = None
    
    def initialize(self, x0, P0):
        """
        Set the initial state estimate and estimate error covariance.

        Parameters:
        x0 : np.array
            Initial state vector.
        P0 : np.array
            Initial covariance matrix. Must match the state vector size.
        """
        self.x = x0.reshape(-1, 1)

        if P0.shape[0] != P0.shape[1]:
            raise ValueError("Covariance matrix P0 must be square.")
        if P0.shape[0] != self.x.shape[0]:
            raise ValueError("Covariance matrix P0 size must match state vector size.")
        self.P = P0

    def predict(self, u=None):
        """
        Predict the next state and covariance.

        Parameters:
        u : np.array (optional)
            Control input vector.
        """
        if self.x is None:
            raise ValueError("State is not initialized. Use initialize() method.")

        # Set the control input ndarray column vector
        if u is not None:
            u = u.reshape(-1, 1)
        
        # Get the model parameters
        F, _, Q, _, _ = self.model.get_params()
               
        # Predict the state
        self.x = self.model.step(self.x, u)
        
        # Predict the error covariance
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        """
        Update the Kalman Filter with a new measurement z.

        Parameters:
        z : np.array
            Measurement vector.
        """
        if self.x is None or self.P is None:
            raise ValueError("State is not initialized. Use initialize_state() method.")
        
        z = z.reshape(-1, 1)
        
        # Get the model parameters
        _, H, _, R, _ = self.model.get_params()
        
        # Compute the innovation (residual) y
        y = z - H @ self.x
        
        # Compute the innovation covariance S
        S = H @ self.P @ H.T + R
        
        # Compute the Kalman gain K
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update the state estimate
        self.x = self.x + K @ y
        
        # Update the estimate covariance
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P
