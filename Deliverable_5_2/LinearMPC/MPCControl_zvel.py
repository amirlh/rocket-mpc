import numpy as np
from scipy.signal import place_poles

from .MPCControl_base import MPCControl_base


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])
    A_hat: np.ndarray  # Augmented A matrix
    B_hat: np.ndarray  # Augmented B matrix
    C_hat: np.ndarray  # Augmented C matrix
    # Storage for data
    stored_x_data: np.ndarray = np.array([10.0])
    stored_u_data: np.ndarray = np.array([0.0])
    L: np.ndarray   # Observer gain
    x_hat: np.ndarray = np.array([10.0])  # State estimate
    d_hat: float = 0.0  # Disturbance estimate
    Bd : np.ndarray = np.array([[-0.1]])
    Cd : np.ndarray = np.array([[0]])

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        # tuning weights
        # state: [vz]
        Q = np.diag([100.0]) 
        R = np.diag([0.1])
        
        # constraints
        # state: [vz] - unconstrained
        x_min = None
        x_max = None
        
        # input: P_avg
        # 40 <= P_avg <= 80
        u_min = np.array([40.05])
        u_max = np.array([79.95])
        
        self._setup_mpc_problem(Q, R, x_min, x_max, u_min, u_max, dist=True, Bd=self.Bd)
        # Setup augmented observer
        self._setup_observer()

        # YOUR CODE HERE
        #################################################
    def _setup_observer(self) -> None:
        """Setup the augmented observer for disturbance estimation"""
        # Initialize observer parameters with default values
        # Bd must be non-zero for the disturbance to be observable

        # Get the discrete-time A and B matrices
        A = self.A  # 1x1 for z-velocity
        B = self.B  # 1x1

        # State dimension (nx = 1 for mpc_z)
        nx = A.shape[0]

        # C matrix (identity - we measure all states)
        C = np.eye(nx)

        # Augmented system matrices: A_hat = [A  Bd; 0  I]
        self.A_hat = np.block([
            [A, self.Bd],
            [np.zeros((1, nx)), np.eye(1)]
        ])

        # B_hat = [B; 0]
        self.B_hat = np.vstack([B, np.zeros((1, B.shape[1]))])

        # C_hat = [C  Cd]
        self.C_hat = np.hstack([C, self.Cd])

        # Place poles for observer design
        poles = np.array([0.4, 0.5])
        res = place_poles(self.A_hat.T, self.C_hat.T, poles)
        self.L = res.gain_matrix.T  # FIXED: removed incorrect negative sign
        
        # Initialize history
        self.x_hat_history = []
        self.d_hat_history = []

    def estimate_parameters(self) -> None:
        """
        Update state and disturbance estimates using augmented observer

        Uses stored_x_data and stored_u_data from the instance
        """
        # Use stored data
        x_data = np.atleast_1d(self.stored_x_data).flatten()
        u_data = np.atleast_1d(self.stored_u_data).flatten()

        # Combine current estimates into augmented state
        x_hat_aug = np.concatenate((self.x_hat.reshape(-1), np.array([self.d_hat])))

        # Observer update: x_hat_next = A_hat @ x_hat + B_hat @ u + L @ (y - C_hat @ x_hat)
        term1 = self.A_hat @ x_hat_aug
        term2 = (self.B_hat @ (u_data - self.us).reshape(-1, 1)).flatten()
        innovation = x_data - self.C_hat @ x_hat_aug  # FIXED: y - C_hat @ x_hat (correct sign!)
        term3 = (self.L @ innovation).flatten()

        x_hat_next = term1 + term2 + term3

        # Extract state and disturbance estimates
        nx = self.x_hat.shape[0]
        self.x_hat = x_hat_next[:nx]
        self.d_hat = x_hat_next[nx]

        # Store history
        self.x_hat_history.append(self.x_hat)
        self.d_hat_history.append(self.d_hat)

        return 

    def store_data(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        """Store the x_data and u_data for later use"""
        self.stored_x_data = x_data
        self.stored_u_data = u_data

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE
        self.estimate_parameters()
        return super().get_u(self.x_hat, x_target, u_target)

        # YOUR CODE HERE
        #################################################
        