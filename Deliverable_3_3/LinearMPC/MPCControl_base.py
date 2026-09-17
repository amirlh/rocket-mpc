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
        x_target: np.ndarray = None,
        use_terminal_set: bool = False,
    ) -> None:
        self.Ts = Ts
        self.H = H
        self.N = int(H / Ts)
        self.nx = self.x_ids.shape[0]
        self.nu = self.u_ids.shape[0]
        
        self.use_terminal_set = use_terminal_set
        if x_target is not None:
            self.x_target_val = x_target[self.x_ids]
        else:
            self.x_target_val = xs[self.x_ids]

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
        
        # optimization variables (delta coordinates: x - xs, u - us)
        self.x_var = cp.Variable((self.N + 1, self.nx))
        self.u_var = cp.Variable((self.N, self.nu))
        
        self.x_init_param = cp.Parameter(self.nx)
        self.x_ref_param = cp.Parameter(self.nx)
        self.u_ref_param = cp.Parameter(self.nu)

        # constraints and cost
        constraints = [self.x_var[0] == self.x_init_param]
        cost = 0
        
        for k in range(self.N):
            # cost: (dx - dx_ref).T * Q * (dx - dx_ref) + (du - du_ref).T * R * (du - du_ref)
            cost += cp.quad_form(self.x_var[k] - self.x_ref_param, Q) + cp.quad_form(self.u_var[k] - self.u_ref_param, R)
            
            # dynamics: dx_{k+1} = A dx_k + B du_k
            constraints += [self.x_var[k+1] == self.A @ self.x_var[k] + self.B @ self.u_var[k]]
            
            # input constraints (u = du + us)
            if u_min is not None:
                constraints += [self.u_var[k] + self.us >= u_min]
            if u_max is not None:
                constraints += [self.u_var[k] + self.us <= u_max]
                
            # state constraints (x = dx + xs)
            if x_min is not None:
                constraints += [self.x_var[k] + self.xs >= x_min]
            if x_max is not None:
                constraints += [self.x_var[k] + self.xs <= x_max]

        # terminal cost
        cost += cp.quad_form(self.x_var[self.N] - self.x_ref_param, P)
        
        # terminal set constraint
        if self.use_terminal_set:
            self._add_terminal_set_constraint(constraints, K, x_min, x_max, u_min, u_max)
        
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    def _add_terminal_set_constraint(self, constraints, K, x_min, x_max, u_min, u_max):
        # invariant set for error dynamics e = x - x_ref
        # constraints on e:
        # x_min - x_ref <= e <= x_max - x_ref
        # u_min - u_ref <= -K*e <= u_max - u_ref
        
        # we assume u_ref = us (delta u_ref = 0) for velocity tracking
        # so u_min - us <= -K*e <= u_max - us
        
        A_cl = self.A - self.B @ K
        
        F_list = []
        g_list = []
        
        # state constraints
        if x_min is not None:
            # e <= x_max - x_ref
            F_list.append(np.eye(self.nx))
            g_list.append(x_max - self.x_target_val)
            # -e <= -(x_min - x_ref) = x_ref - x_min
            F_list.append(-np.eye(self.nx))
            g_list.append(self.x_target_val - x_min)
            
        # input constraints
        if u_max is not None:
            # -K*e <= u_max - us
            F_list.append(-K)
            g_list.append(u_max - self.us)
        if u_min is not None:
            # K*e <= -(u_min - us) = us - u_min
            F_list.append(K)
            g_list.append(self.us - u_min)
            
        if not F_list:
            return # no constraints, no set
            
        F = np.vstack(F_list)
        g = np.hstack(g_list)
        
        # filter out infinite constraints
        finite_mask = np.isfinite(g)
        F = F[finite_mask]
        g = g[finite_mask]
        
        try:
            P_poly = Polyhedron.from_Hrep(F, g)
            
            Omega = P_poly
            max_iter = 20
            for i in range(max_iter):
                F_omega = Omega.A
                g_omega = Omega.b
                F_pre = F_omega @ A_cl
                g_pre = g_omega
                Pre_Omega = Polyhedron.from_Hrep(F_pre, g_pre)
                Omega_next = Omega.intersect(Pre_Omega)
                if Pre_Omega.contains(Omega):
                    break
                Omega = Omega_next
                
            F_f = Omega.A
            g_f = Omega.b
            
            constraints += [F_f @ (self.x_var[self.N] - self.x_ref_param) <= g_f]
            print(f"Terminal set added with {F_f.shape[0]} inequalities.")
            
        except Exception as e:
            print(f"Failed to compute terminal set: {e}")

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
        
        # convert to delta coordinates
        dx0 = x0 - self.xs
        self.x_init_param.value = dx0
        
        if x_target is not None:
            self.x_ref_param.value = x_target - self.xs
        else:
            self.x_ref_param.value = np.zeros(self.nx)
            
        if u_target is not None:
            self.u_ref_param.value = u_target - self.us
        else:
            self.u_ref_param.value = np.zeros(self.nu)
        
        # self.ocp.solve(solver=cp.OSQP, warm_start=True)
        self.ocp.solve()
        
        if self.ocp.status not in ["optimal", "optimal_inaccurate"]:
            # fallback: return steady state input
            u0 = self.us
            x_traj = np.zeros((self.N+1, self.nx)) + self.xs
            u_traj = np.zeros((self.N, self.nu)) + self.us
            return u0, x_traj.T, u_traj.T

        # convert back to absolute coordinates
        du0 = self.u_var[0].value
        u0 = du0 + self.us
        
        x_traj = self.x_var.value + self.xs
        u_traj = self.u_var.value + self.us
        
        return u0, x_traj.T, u_traj.T
        
