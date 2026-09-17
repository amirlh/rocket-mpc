import numpy as np

from .MPCControl_base import MPCControl_base


class MPCControl_xvel(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6])
    u_ids: np.ndarray = np.array([1])

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE
        # tuning weights
        # state: [wy, beta, vx]
        # we want to track vx, so penalize vx error heavily
        # we also want to keep beta small (constraint)
        Q = np.diag([1.0, 10.0, 100.0]) 
        R = np.diag([1.0])
        
        # constraints
        # state: [wy, beta, vx]
        # beta constraint: +/- 10 deg
        beta_constraint = 10 * np.pi / 180
        x_min = np.array([-np.inf, -beta_constraint, -np.inf])
        x_max = np.array([np.inf, beta_constraint, np.inf])
        
        # input: delta2
        # +/- 15 deg
        delta_constraint = 15 * np.pi / 180
        u_min = np.array([-delta_constraint])
        u_max = np.array([delta_constraint])
        
        self._setup_mpc_problem(Q, R, x_min, x_max, u_min, u_max)

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE
        
        return super().get_u(x0, x_target, u_target)
    
        # YOUR CODE HERE
        #################################################