import numpy as np

from .MPCControl_base import MPCControl_base


class MPCControl_yvel(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7])
    u_ids: np.ndarray = np.array([0])

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE
        # tuning weights
        # state: [wx, alpha, vy]
        Q = np.diag([1.0, 10.0, 100.0]) 
        R = np.diag([1.0])
        
        # constraints
        # state: [wx, alpha, vy]
        # alpha constraint: +/- 10 deg
        alpha_contraint = 10 * np.pi / 180
        x_min = np.array([-np.inf, -alpha_contraint, -np.inf])
        x_max = np.array([np.inf, alpha_contraint, np.inf])
        
        # input: delta1
        # +/- 15 deg
        delta_constraint = 15 * np.pi / 180
        u_min = np.array([-delta_constraint])
        u_max = np.array([delta_constraint])
        
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
