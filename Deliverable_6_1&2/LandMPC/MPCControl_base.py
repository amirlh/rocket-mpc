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
        soft_constraints: bool = False,
        slack_weight: float = 100000.0,
    ) -> None:
        self.Ts = Ts
        self.H = H
        self.N = int(H / Ts)
        self.nx = self.x_ids.shape[0]
        self.nu = self.u_ids.shape[0]
        
        self.soft_constraints = soft_constraints
        self.slack_weight = slack_weight

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
        #################################################
        # YOUR CODE HERE
        
        # Unified MPC formulation (Tube-based structure)
        # For Nominal MPC, E={0}, K=0, X_tilde=X, U_tilde=U
        
        # Decision variables: z (nominal state trajectory), v (nominal input trajectory)
        z_var = cp.Variable((self.N+1, self.nx), name='z')
        v_var = cp.Variable((self.N, self.nu), name='v')
        x0_var = cp.Parameter((self.nx,), name='x0')
        
        # Slack variables for soft constraints
        if self.soft_constraints:
            # Number of constraints in X_tilde
            n_constr = self.X_tilde.A.shape[0]
            slack_var = cp.Variable((self.N, n_constr), nonneg=True)
        else:
            slack_var = None
        
        # Cost function: sum_{k=0}^{N-1} (z_k^T Q z_k + v_k^T R v_k) + z_N^T Qf z_N
        cost = 0
        for i in range(self.N):
            cost += cp.quad_form(z_var[i], self.Q)
            cost += cp.quad_form(v_var[i], self.R)
            if self.soft_constraints:
                cost += self.slack_weight * cp.sum(slack_var[i])
        cost += cp.quad_form(z_var[-1], self.Qf)
        
        # Constraints
        constraints = []
        
        # Initial condition
        if hasattr(self, 'E') and self.E is not None:
            # Tube MPC: x0 in z0 + E  <=>  E.A @ (x0 - z0) <= E.b
            constraints.append(self.E.A @ (x0_var - z_var[0]) <= self.E.b)
        else:
            # Nominal MPC: x0 = z0
            constraints.append(z_var[0] == x0_var)
        
        # Dynamics: z[k+1] = A*z[k] + B*v[k]
        constraints.append(z_var[1:].T == self.A @ z_var[:-1].T + self.B @ v_var.T)
        
        # State constraints: z[k] in X_tilde
        if self.soft_constraints:
            constraints.append(self.X_tilde.A @ z_var[:-1].T <= self.X_tilde.b.reshape(-1, 1) + slack_var.T)
        else:
            constraints.append(self.X_tilde.A @ z_var[:-1].T <= self.X_tilde.b.reshape(-1, 1))
        
        # Input constraints: v[k] in U_tilde
        constraints.append(self.U_tilde.A @ v_var.T <= self.U_tilde.b.reshape(-1, 1))
        
        # Terminal constraint: z[N] in Xf_tilde
        constraints.append(self.Xf_tilde.A @ z_var[-1] <= self.Xf_tilde.b)
        
        # Create optimization problem
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        self.z_var = z_var
        self.v_var = v_var
        self.x0_var = x0_var
            

        # YOUR CODE HERE
        #################################################

    def _compute_mRPI(self, A_cl: np.ndarray, W: Polyhedron, max_iter: int = 100) -> Polyhedron:
        """Compute minimal robust invariant set E for system x+ = A_cl*x + w, w in W"""
        nx = A_cl.shape[0]
        Omega = W
        itr = 1
        A_cl_ith_power = np.eye(nx)
        
        while itr < max_iter:
            A_cl_ith_power = np.linalg.matrix_power(A_cl, itr)
            Omega_next = Omega + A_cl_ith_power @ W
            Omega_next.minHrep()
            
            if np.linalg.norm(A_cl_ith_power, ord=2) < 1e-2:
                print(f'Minimal robust invariant set computation converged after {itr} iterations.')
                break
            
            if itr == max_iter - 1:
                print(f'Minimal robust invariant set computation did NOT converge after {max_iter} iterations.')
            
            Omega = Omega_next
            itr += 1
        
        return Omega_next
    
    def _compute_max_invariant_set(self, A_cl: np.ndarray, X: Polyhedron, max_iter: int = 100) -> Polyhedron:
        """Compute maximal invariant set for autonomous system x+ = A_cl*x with x in X"""
        O = X
        itr = 1
        converged = False
        
        while itr <= max_iter:
            Oprev = O
            F, f = O.A, O.b
            # Compute the pre-set
            O = Polyhedron.from_Hrep(
                np.vstack((F, F @ A_cl)),
                np.vstack((f, f)).reshape((-1,))
            )
            O.minHrep()
            
            if O == Oprev:
                converged = True
                break
            
            itr += 1
        
        if converged:
            print(f'Maximum invariant set successfully computed after {itr} iterations.')
        else:
            print(f'Maximum invariant set did NOT converge after {max_iter} iterations.')
        
        return O

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

        # Convert absolute state to delta form for the optimization
        delta_x0 = x0 - self.xs
        
        # Set the initial state parameter (in delta form)
        self.x0_var.value = delta_x0
        
        # Solve the optimization problem
        self.ocp.solve(solver=cp.PIQP, verbose=False)
        
        # Check if the problem was solved successfully
        if self.ocp.status != cp.OPTIMAL:
            print(f"Warning: MPC solver returned status: {self.ocp.status}")
            # Return zero control if infeasible
            u0 = np.zeros(self.nu)
            x_traj = np.zeros((self.nx, self.N+1))
            u_traj = np.zeros((self.nu, self.N))
            return u0, x_traj, u_traj
        
        # Extract optimal nominal trajectories
        z_opt = self.z_var.value  # shape: (N+1, nx)
        v_opt = self.v_var.value  # shape: (N, nu)
        
        # Return nominal values (in delta form for Tube MPC to use)
        #   # First nominal input
        u0 = v_opt[0] # For nominal MPC, u0=v0. For Tube MPC, will be overridden.
        
        # Return trajectories (nominal trajectories in delta form)
        x_traj = z_opt.T  # shape: (nx, N+1)
        u_traj = v_opt.T  # shape: (nu, N)
        
        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj
        
