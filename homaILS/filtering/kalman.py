import numpy as np

class KalmanFilter:
    def __init__(self, F, H, Q, R, P, B=None):
        """
        Initialize the Kalman Filter.

        Parameters:
        F : np.array
            State transition matrix.
        H : np.array
            Observation matrix.
        Q : np.array
            Process noise covariance.
        R : np.array
            Measurement noise covariance.
        P : np.array
            Initial estimate error covariance.
        B : np.array (optional)
            Control-input model matrix.
        """
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.P = P
        self.B = B if B is not None else np.zeros((F.shape[0], 1))
        
        # State estimate will be set later
        self.x = None

    def initialize_state(self, x0):
        """Set the initial state estimate."""
        self.x = x0

    def model_step(self, x, u=None):
        """
        Model step function of the Kalman Filter.
        Compute the next state given the current state and control input,
        without updating the state estimate.
        
        Parameters:
        x : np.array
            State vector.
        u : np.array (optional)
            Control input vector.
        """
        if u is None:
            u = np.zeros((self.B.shape[1], 1))
        
        # Predict the state
        x = self.F @ x + self.B @ u

        return x

    def predict(self, u=None):
        """
        Predict the next state and covariance.
        
        Parameters:
        u : np.array (optional)
            Control input vector.
        """
        if self.x is None:
            raise ValueError("State is not initialized. Use initialize_state() method.")
               
        # Predict the state
        self.x = self.model_step(self.x, u)
        
        # Predict the error covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """
        Update the Kalman Filter with a new measurement z.
        
        Parameters:
        z : np.array
            Measurement vector.
        """
        if self.x is None:
            raise ValueError("State is not initialized. Use initialize_state() method.")
        
        # Compute the innovation (residual) y
        y = z - self.H @ self.x
        
        # Compute the innovation covariance S
        S = self.H @ self.P @ self.H.T + self.R
        
        # Compute the Kalman gain K
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update the state estimate
        self.x = self.x + K @ y
        
        # Update the estimate covariance
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
