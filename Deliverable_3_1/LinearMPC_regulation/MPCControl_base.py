import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import cont2discrete


class MPCControl_base:
    """Complete states indices"""

    x_ids: np.ndarray
    u_ids: np.ndarray

    """Optimization system"""
    A: np.ndarray
    B: np.ndarray
    xs: np.ndarray
    us: np.ndarray
    nx: int
    nu: int
    Ts: float
    H: float
    N: int

    """Optimization problem"""
    ocp: cp.Problem

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        xs: np.ndarray,
        us: np.ndarray,
        Ts: float,
        H: float,
    ) -> None:
        self.Ts = Ts
        self.H = H
        self.N = int(H / Ts)
        self.nx = self.x_ids.shape[0]
        self.nu = self.u_ids.shape[0]

        # System definition
        xids_xi, xids_xj = np.meshgrid(self.x_ids, self.x_ids)
        A_red = A[xids_xi, xids_xj].T
        uids_xi, uids_xj = np.meshgrid(self.x_ids, self.u_ids)
        B_red = B[uids_xi, uids_xj].T

        self.A, self.B = self._discretize(A_red, B_red, Ts)
        self.xs = xs[self.x_ids]
        self.us = us[self.u_ids]

        self._setup_controller()

    def _setup_controller(self) -> None:
        # this method should be overridden by subclasses
        pass

    def _compute_terminal_set(self, K, x_min, x_max, u_min, u_max):
        # closed loop dynamics
        A_cl = self.A - self.B @ K
        
        # constraints in form F*x <= g
        F_list = []
        g_list = []
        
        # state constraints: x_min <= x + xs <= x_max  =>  x_min - xs <= x <= x_max - xs
        if x_min is not None:
            # x <= x_max - xs
            F_list.append(np.eye(self.nx))
            g_list.append(x_max - self.xs)
            # -x <= -(x_min - xs) = xs - x_min
            F_list.append(-np.eye(self.nx))
            g_list.append(self.xs - x_min)
            
        # input constraints: u_min <= u + us <= u_max  =>  u_min - us <= -K*x <= u_max - us
        # -K*x <= u_max - us
        if u_max is not None:
            F_list.append(-K)
            g_list.append(u_max - self.us)
        
        # -K*x >= u_min - us  =>  Kx <= -(u_min - us) = us - u_min
        if u_min is not None:
            F_list.append(K)
            g_list.append(self.us - u_min)
            
        if not F_list:
            return None, None
            
        F = np.vstack(F_list)
        g = np.hstack(g_list)
        
        try:
            P_poly = Polyhedron.from_Hrep(F, g)
            
            # invariant set
            Omega = P_poly
            max_iter = 100
            for i in range(max_iter):
                F_omega = Omega.A
                g_omega = Omega.b
                
                # pre-set: F_omega * (A_cl * x) <= g_omega
                F_pre = F_omega @ A_cl
                g_pre = g_omega
                
                Pre_Omega = Polyhedron.from_Hrep(F_pre, g_pre)
                
                Omega_next = Omega.intersect(Pre_Omega)
                
                if Pre_Omega.contains(Omega):
                    break
                
                Omega = Omega_next
            return Omega.A, Omega.b
        except Exception as e:
            print(f"Warning: Could not compute terminal set: {e}")
            return None, None

    def _setup_mpc_problem(
        self,
        Q: np.ndarray,
        R: np.ndarray,
        x_min: np.ndarray = None,
        x_max: np.ndarray = None,
        u_min: np.ndarray = None,
        u_max: np.ndarray = None,
    ) -> None:
        # compute lqr controller K and terminal cost P
        K, P, _ = dlqr(self.A, self.B, Q, R)
        
        # compute terminal set
        F_f, g_f = self._compute_terminal_set(K, x_min, x_max, u_min, u_max)
        
        self.x_var = cp.Variable((self.N + 1, self.nx))
        self.u_var = cp.Variable((self.N, self.nu))
        
        self.x_init_param = cp.Parameter(self.nx)
        self.x_ref_param = cp.Parameter(self.nx)
        self.u_ref_param = cp.Parameter(self.nu)

        # affine term for linearized dynamics: x_s - A*x_s - B*u_s
        d_affine = self.xs - self.A @ self.xs - self.B @ self.us
        
        # constraints and cost
        constraints = [self.x_var[0] == self.x_init_param]
        cost = 0
        
        for k in range(self.N):
            # cost: (x - x_ref).T * Q * (x - x_ref) + (u - u_ref).T * R * (u - u_ref)
            cost += cp.quad_form(self.x_var[k] - self.x_ref_param, Q) + cp.quad_form(self.u_var[k] - self.u_ref_param, R)
            
            # dynamics: x_{k+1} = A x_k + B u_k + d_affine
            constraints += [self.x_var[k+1] == self.A @ self.x_var[k] + self.B @ self.u_var[k] + d_affine]
            
            # input constraints
            if u_min is not None:
                constraints += [self.u_var[k] >= u_min]
            if u_max is not None:
                constraints += [self.u_var[k] <= u_max]
                
            # state constraints
            if x_min is not None:
                constraints += [self.x_var[k] >= x_min]
            if x_max is not None:
                constraints += [self.x_var[k] <= x_max]

        # terminal cost
        cost += cp.quad_form(self.x_var[self.N] - self.x_ref_param, P)
        
        # terminal set constraint
        if F_f is not None:
            constraints += [F_f @ (self.x_var[self.N] - self.x_ref_param) <= g_f]
        
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    @staticmethod
    def _discretize(A: np.ndarray, B: np.ndarray, Ts: float):
        nx, nu = B.shape
        C = np.zeros((1, nx))
        D = np.zeros((1, nu))
        A_discrete, B_discrete, _, _, _ = cont2discrete(system=(A, B, C, D), dt=Ts)
        return A_discrete, B_discrete

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE
        self.x_init_param.value = x0
        self.x_ref_param.value = x_target if x_target is not None else self.xs
        self.u_ref_param.value = u_target if u_target is not None else self.us
        
        # self.ocp.solve(solver=cp.OSQP, warm_start=True)
        self.ocp.solve()
        
        if self.ocp.status not in ["optimal", "optimal_inaccurate"]:
            # fallback: return steady state input
            u0 = self.us
            x_traj = np.zeros((self.N+1, self.nx)) + self.xs
            u_traj = np.zeros((self.N, self.nu)) + self.us
            return u0, x_traj.T, u_traj.T

        u0 = self.u_var[0].value
        x_traj = self.x_var.value
        u_traj = self.u_var.value
        # YOUR CODE HERE
        #################################################
        
        return u0, x_traj.T, u_traj.T
        
