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
        dist: bool = False,
    ) -> None:
        self.Ts = Ts
        self.H = H
        self.N = int(H / Ts)
        self.nx = self.x_ids.shape[0]
        self.nu = self.u_ids.shape[0]

        self.use_terminal_set = use_terminal_set
        self.dist = dist
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
        soft_constraints: bool = True,
        slack_weight: float = 100000.0,
        dist: bool = False,
        Bd: np.ndarray = None,
    ) -> None:
        # Store disturbance parameters
        self.dist = dist
        if Bd is not None:
            self.Bd = Bd

        # Store input constraints
        self.u_min = u_min
        self.u_max = u_max

        # compute lqr controller K and terminal cost P
        K, P, _ = dlqr(self.A, self.B, Q, R)
        
        self.x_var = cp.Variable((self.N + 1, self.nx))
        self.u_var = cp.Variable((self.N, self.nu))
        
        # slack variables for soft constraints
        self.slack_var = cp.Variable((self.N, self.nx), nonneg=True) if soft_constraints else None
        
        self.x_init_param = cp.Parameter(self.nx)
        self.x_ref_param = cp.Parameter(self.nx) # ref state (delta from xs)
        self.u_ref_param = cp.Parameter(self.nu) # ref input (delta from us)

        # Disturbance parameter
        if dist and Bd is not None:
            self.d_param = cp.Parameter()  # scalar disturbance

        # constraints and cost
        constraints = [self.x_var[0] == self.x_init_param]
        cost = 0
        
        for k in range(self.N):
            # cost: (x - x_ref).T * Q * (x - x_ref) + u.T * R * u
            # assume u_ref is 0 (delta u = 0 => u = us) for velocity tracking
            cost += cp.quad_form(self.x_var[k] - self.x_ref_param, Q) + cp.quad_form(self.u_var[k] - self.u_ref_param, R)
            
            if soft_constraints:
                cost += slack_weight * cp.sum(self.slack_var[k])

            # Dynamics constraint with optional disturbance
            if dist and Bd is not None:
                constraints += [self.x_var[k+1] == self.A @ self.x_var[k] + self.B @ self.u_var[k] + self.Bd.flatten() * self.d_param]
            else:
                constraints += [self.x_var[k+1] == self.A @ self.x_var[k] + self.B @ self.u_var[k]]
            
            # input constraints
            if u_min is not None:
                constraints += [self.u_var[k] + self.us >= u_min]
            if u_max is not None:
                constraints += [self.u_var[k] + self.us <= u_max]
                
            # state constraints
            if x_min is not None:
                if soft_constraints:
                    constraints += [self.x_var[k] + self.xs >= x_min - self.slack_var[k]]
                else:
                    constraints += [self.x_var[k] + self.xs >= x_min]
            if x_max is not None:
                if soft_constraints:
                    constraints += [self.x_var[k] + self.xs <= x_max + self.slack_var[k]]
                else:
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

    def compute_equilibrium_input(self, x_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute u_target that maintains x_target at equilibrium with input constraints.
        Solves: min ||du_target||^2
                s.t. B*du_target = (I - A)*dx_target - Bd*d_hat
                     u_min <= du_target + us <= u_max

        Args:
            x_target: Target state (absolute coordinates)

        Returns:
            u_target: Equilibrium input (absolute coordinates)
            x_target_new: Modified target state if original was infeasible, otherwise None
        """
        dx_target = x_target - self.xs
        # At equilibrium: x_target = A*x_target + B*u_target + Bd*d_hat (if disturbance)
        # In deviation form: dx_target = A*dx_target + B*du_target + Bd*d_hat
        # Rearranging: (I - A)*dx_target - Bd*d_hat = B*du_target
        rhs = (np.eye(self.nx) - self.A) @ dx_target

        # Subtract disturbance contribution if using disturbance compensation
        if self.dist and hasattr(self, 'd_hat') and hasattr(self, 'Bd'):
            rhs = rhs - self.Bd.flatten() * self.d_hat

        # Solve with input constraints using cvxpy
        du_var = cp.Variable(self.nu)
        cost = cp.sum_squares(du_var)  # Minimize norm of du
        constraints = []

        # Equilibrium constraint (strict equality)
        constraints.append(self.B @ du_var == rhs)

        # Input constraints
        if hasattr(self, 'u_min') and self.u_min is not None:
            constraints.append(du_var + self.us >= self.u_min)
        if hasattr(self, 'u_max') and self.u_max is not None:
            constraints.append(du_var + self.us <= self.u_max)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()

        if du_var.value is None:
            # Infeasible: find closest feasible x_target
            # Solve: min ||dx_target_new - dx_target_original||^2
            #        s.t. (I - A) @ dx_target_new = B @ du_target + Bd @ d_hat
            #             u_min <= du_target + us <= u_max
            dx_target_new_var = cp.Variable(self.nx)
            du_target_var = cp.Variable(self.nu)

            cost_fallback = cp.sum_squares(dx_target_new_var - dx_target)
            constraints_fallback = []

            # Equilibrium constraint: (I - A) @ dx_target_new = B @ du_target + Bd @ d_hat
            lhs = (np.eye(self.nx) - self.A) @ dx_target_new_var
            rhs_fallback = self.B @ du_target_var
            if self.dist and hasattr(self, 'd_hat') and hasattr(self, 'Bd'):
                rhs_fallback = rhs_fallback + self.Bd.flatten() * self.d_hat
            constraints_fallback.append(lhs == rhs_fallback)

            # Input constraints
            if hasattr(self, 'u_min') and self.u_min is not None:
                constraints_fallback.append(du_target_var + self.us >= self.u_min)
            if hasattr(self, 'u_max') and self.u_max is not None:
                constraints_fallback.append(du_target_var + self.us <= self.u_max)

            # State constraints (if they exist)
            if hasattr(self, 'x_min') and hasattr(self, 'x_max'):
                if self.x_min is not None:
                    constraints_fallback.append(dx_target_new_var + self.xs >= self.x_min)
                if self.x_max is not None:
                    constraints_fallback.append(dx_target_new_var + self.xs <= self.x_max)

            prob_fallback = cp.Problem(cp.Minimize(cost_fallback), constraints_fallback)
            prob_fallback.solve()

            if du_target_var.value is not None:
                du_target = du_target_var.value
                # Extract the new x_target
                x_target_new = dx_target_new_var.value + self.xs
            else:
                # Ultimate fallback: use u_min as default
                du_target = self.u_min - self.us if (hasattr(self, 'u_min') and self.u_min is not None) else np.zeros(self.nu)
                x_target_new = None
        else:
            du_target = du_var.value
            x_target_new = None  # Original x_target was feasible

        u_target = du_target + self.us
        return u_target, x_target_new

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE
        # x0 is already reduced
        dx0 = x0 - self.xs
        self.x_init_param.value = dx0
        # ref
        if x_target is not None:
            # Compute u_target if not provided

            u_target, x_target_new = self.compute_equilibrium_input(x_target)
            # Update x_target if it was modified for feasibility
            if x_target_new is not None:
                x_target = x_target_new
            # x_target is already reduced
            dx_ref = x_target - self.xs
            self.x_ref_param.value = dx_ref

        else:
            self.x_ref_param.value = np.zeros(self.nx)

        # Set u_ref (deviation from us)
        if u_target is not None:
            self.u_ref_param.value = u_target - self.us
        else:
            self.u_ref_param.value = np.zeros(self.nu)

        # Set disturbance parameter if using disturbance compensation
        if self.dist and hasattr(self, 'd_param'):
            self.d_param.value = self.d_hat if hasattr(self, 'd_hat') else 0.0

        # self.ocp.solve(solver=cp.OSQP, warm_start=True)
        self.ocp.solve()

        du0 = self.u_var[0].value
        u0 = du0 + self.us

        # Debug: check constraint violations
        
        x_traj = self.x_var.value + self.xs
        u_traj = self.u_var.value + self.us

        # YOUR CODE HERE
        #################################################
        #################################################
        
        return u0, x_traj.T, u_traj.T
        
