import numpy as np
import casadi as ca
from typing import Tuple
from scipy.signal import cont2discrete
import scipy.linalg

class NmpcCtrl:
    """
    Nonlinear MPC controller.
    get_u should provide this functionality: u0, x_ol, u_ol, t_ol = mpc_z_rob.get_u(t0, x0).
    - x_ol shape: (12, N+1); u_ol shape: (4, N); t_ol shape: (N+1,)
    You are free to modify other parts    
    """

    def __init__(self, rocket, H=2.0, xs=None, us=None):
        """
        Hint: As in our NMPC exercise, you can evaluate the dynamics of the rocket using 
            CASADI variables x and u via the call rocket.f_symbolic(x,u).
            We create a self.f for you: x_dot = self.f(x,u)
        """
        self.xs = xs
        self.us = us
        # symbolic dynamics f(x,u) from rocket
        self.f = lambda x,u: rocket.f_symbolic(x,u)[0]
        self.Ts = rocket.Ts
        self.N = int(H / self.Ts) # horizon length

        # linearization for terminal cost
        A, B = rocket.linearize(xs, us)
        
        # discretization for lqr
        C = np.eye(12)
        D = np.zeros((12, 4))
        Ad, Bd, _, _, _ = cont2discrete((A, B, C, D), self.Ts)
        
        # tuning weights
        # state: [wx, wy, wz, alpha, beta, gamma, vx, vy, vz, x, y, z]
        Q_diag = np.array([
            1.0, 1.0, 1.0,      # omega
            50.0, 50.0, 50.0,   # angles
            10.0, 10.0, 10.0,   # velocity
            50.0, 50.0, 50.0    # position
        ])
        self.Q = np.diag(Q_diag)
        
        R_diag = np.array([1.0, 1.0, 0.1, 1.0]) # inputs [d1, d2, Pavg, Pdiff]
        self.R = np.diag(R_diag)
        
        # DARE for terminal cost P
        self.P_term = scipy.linalg.solve_discrete_are(Ad, Bd, self.Q, self.R)

        self._setup_controller()

    def _setup_controller(self) -> None:
        
        opti = ca.Opti()
        
        # variables
        X = opti.variable(12, self.N + 1)
        U = opti.variable(4, self.N)
        
        # parameters
        x0_param = opti.parameter(12)
        
        # objective
        obj = 0
        for k in range(self.N):
            # state error
            e_x = X[:, k] - self.xs
            # input error
            e_u = U[:, k] - self.us
            
            # cost
            obj += ca.mtimes([e_x.T, self.Q, e_x]) + ca.mtimes([e_u.T, self.R, e_u])
            
        # terminal cost
        e_xN = X[:, self.N] - self.xs
        obj += ca.mtimes([e_xN.T, self.P_term, e_xN])
        
        opti.minimize(obj)
        
        # constraints
        
        # initial state
        opti.subject_to(X[:, 0] == x0_param)
        
        # dynamics (RK4)
        dt = self.Ts
        for k in range(self.N):
            k1 = self.f(X[:, k], U[:, k])
            k2 = self.f(X[:, k] + dt/2 * k1, U[:, k])
            k3 = self.f(X[:, k] + dt/2 * k2, U[:, k])
            k4 = self.f(X[:, k] + dt * k3, U[:, k])
            x_next = X[:, k] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            opti.subject_to(X[:, k+1] == x_next)
            
        # state constraints
        # z >= 0 (index 11)
        opti.subject_to(X[11, :] >= 0)
        
        # abs(beta) <= 80 deg (index 4)
        beta_lim = np.deg2rad(80)
        opti.subject_to(opti.bounded(-beta_lim, X[4, :], beta_lim))
        
        # input constraints
        # delta1, delta2 (0, 1) in [-0.26, 0.26]
        opti.subject_to(opti.bounded(-0.26, U[0, :], 0.26))
        opti.subject_to(opti.bounded(-0.26, U[1, :], 0.26))
        
        # Pavg (2) in [10, 90]
        opti.subject_to(opti.bounded(10, U[2, :], 90))
        
        # Pdiff (3) in [-20, 20]
        opti.subject_to(opti.bounded(-20, U[3, :], 20))
        
        # solver options
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'expand': True,
            'ipopt.max_iter': 500
        }
        opti.solver('ipopt', opts)
        
        self.opti = opti
        self.X = X
        self.U = U
        self.x0_param = x0_param

    def get_u(self, t0: float, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        self.opti.set_value(self.x0_param, x0)
        
        # warm start
        # initialize with steady state
        self.opti.set_initial(self.X, ca.repmat(self.xs, 1, self.N+1))
        self.opti.set_initial(self.U, ca.repmat(self.us, 1, self.N))
        
        try:
            sol = self.opti.solve()
            u0 = sol.value(self.U[:, 0])
            x_ol = sol.value(self.X)
            u_ol = sol.value(self.U)
        except RuntimeError:
            # if solver fails, return steady state input
            print(f"Warning: NMPC solver failed at t={t0:.2f}")
            u0 = self.us
            x_ol = np.zeros((12, self.N+1))
            u_ol = np.zeros((4, self.N))
            
            try:
                u0 = self.opti.debug.value(self.U[:, 0])
                x_ol = self.opti.debug.value(self.X)
                u_ol = self.opti.debug.value(self.U)
            except:
                pass

        t_ol = np.arange(self.N + 1) * self.Ts + t0

        return u0, x_ol, u_ol, t_ol