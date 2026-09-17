import numpy as np

from .MPCControl_base import MPCControl_base


class MPCControl_roll(MPCControl_base):
    x_ids: np.ndarray = np.array([2, 5])
    u_ids: np.ndarray = np.array([3])

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        # tuning weights
        # state: [wz, gamma]
        Q = np.diag([1.0, 100.0]) 
        R = np.diag([1.0])
        
        # constraints
        # state: [wz, gamma] - unconstrained
        x_min = None
        x_max = None
        
        # input: P_diff
        # -20 <= P_diff <= 20
        u_min = np.array([-20.0])
        u_max = np.array([20.0])
        
        self._setup_mpc_problem(Q, R, x_min, x_max, u_min, u_max)

        # YOUR CODE HERE
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE
        
        return super().get_u(x0, x_target, u_target)
    
        # YOUR CODE HERE
        #################################################
