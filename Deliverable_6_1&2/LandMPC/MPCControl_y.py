import numpy as np
from control import dlqr
from mpt4py import Polyhedron

from .MPCControl_base import MPCControl_base


class MPCControl_y(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7, 10])
    u_ids: np.ndarray = np.array([0])

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE
        
        # Enable soft constraints
        self.soft_constraints = True
        self.slack_weight = 1000.0

        # Step 1: Define Constraints
        # Alpha constraint: |alpha| <= 0.1745
        # alpha is state index 1
        alpha_max = 0.1745
        A_x = np.array([
            [0., 1., 0., 0.],
            [0., -1., 0., 0.]
        ])
        b_x = np.array([
            alpha_max - self.xs[1],
            alpha_max + self.xs[1]
        ])
        X = Polyhedron.from_Hrep(A_x, b_x)
        
        # Input constraint: |delta_1| <= 0.26
        u_max_val = 0.26
        A_u = np.array([[1.], [-1.]])
        b_u = np.array([
            u_max_val - self.us[0],
            u_max_val + self.us[0]
        ])
        U = Polyhedron.from_Hrep(A_u, b_u)
        
        # Step 2: Nominal MPC Controller
        Q_mpc = np.diag([10.0, 10.0, 10.0, 50.0])
        R_mpc = 1.0 * np.eye(self.nu)
        K_mpc, Qf_mpc, _ = dlqr(self.A, self.B, Q_mpc, R_mpc)
        K_mpc = -K_mpc
        
        self.Qf = Qf_mpc
        self.Q = Q_mpc
        self.R = R_mpc
        
        A_cl_mpc = self.A + self.B @ K_mpc
        
        # Step 3: Terminal Set
        X_and_KU = X.intersect(
            Polyhedron.from_Hrep(U.A @ K_mpc, U.b)
        )
        
        try:
            Xf = self._compute_max_invariant_set(A_cl_mpc, X_and_KU)
        except Exception as e:
            print(f"Warning: Could not compute Xf for Y: {e}")
            Xf = X
            
        # Step 4: Setup Parameters for Base Class
        self.K = np.zeros((self.nu, self.nx))
        self.E = None
        
        self.X_tilde = X
        self.U_tilde = U
        self.Xf_tilde = Xf

        self.ocp = None
        MPCControl_base._setup_controller(self)

        # YOUR CODE HERE
        #################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE

        u0, x_traj, u_traj = super().get_u(x0, x_target, u_target)
        
        # Convert to absolute coordinates
        u0 = u0 + self.us
        x_traj = x_traj + self.xs.reshape(-1, 1)
        u_traj = u_traj + self.us.reshape(-1, 1)

        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj
