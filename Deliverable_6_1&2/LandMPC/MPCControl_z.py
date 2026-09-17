import numpy as np
import cvxpy as cp
from control import dlqr
from mpt4py import Polyhedron

from .MPCControl_base import MPCControl_base


class MPCControl_z(MPCControl_base):
    x_ids: np.ndarray = np.array([8, 11])
    u_ids: np.ndarray = np.array([2])

    # only useful for part 5 of the project
    d_estimate: np.ndarray
    d_gain: float

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE
        
        # Step 1: Compute LQR gain K_robust for A_cl = A + B*K_robust
        # This controller is used to define the tube (minimal robust invariant set E)
        # TUNING: More aggressive to reduce size of E
        Q_robust = np.diag([50.0, 200.0])  # High penalty on states for tight tube
        R_robust = 0.1 * np.eye(self.nu)  # Low penalty on control
        K_robust, _, _ = dlqr(self.A, self.B, Q_robust, R_robust)
        K_robust = -K_robust
        
        # Store K_robust for Tube MPC (used in get_u to compute actual input)
        self.K = K_robust
        
        A_cl_robust = self.A + self.B @ K_robust
        
        # Step 2: Define disturbance set W = [-15, 5]
        # The disturbance w is scalar and applied via B: x+ = A_cl*x + B*w
        # So the disturbance in state space is BW = {B*w | w in [-15, 5]}
        # W scalar: w in [-15, 5]
        W = Polyhedron.from_Hrep(
            A=np.array([[1.], [-1.]]),
            b=np.array([5., 15.])
        )
        # Map to state space: BW
        BW = self.B @ W
        
        # Step 3: Compute minimal robust invariant set E
        # For system: x+ = A_cl_robust*x + e, where e in BW
        E = self._compute_mRPI(A_cl_robust, BW)
        self.E = E
        
        # Step 4: Define original state and input constraints IN DELTA FORM
        # State: delta_x = x - xs, where x = [z_vel, z_pos]
        # Constraint: z_pos >= 0  =>  delta_z_pos + xs[1] >= 0  =>  delta_z_pos >= -xs[1]
        # No constraint on z_vel (x_max = None, x_min = None for velocity)
        # x_min = [None, -xs[1]]  i.e., [None, 0-xs[1]]
        # x_max = [None, None]
        
        # Only constraint: delta_z_pos >= -xs[1] (to ensure z_pos >= 0)
        # In Hrep: [0, -1] * [delta_z_vel; delta_z_pos] <= xs[1]
        X = Polyhedron.from_Hrep(
            A=np.array([[0., -1.]]),  # -delta_z_pos <= xs[1]  (i.e., delta_z_pos >= -xs[1])
            b=np.array([self.xs[1]])
        )
        
        # Input constraints: 40 <= P_avg <= 80 (percentage)
        # Delta inputs: u_delta = P_avg - us
        u_min_original = 40.0 - self.us[0]
        u_max_original = 80.0 - self.us[0]
        U = Polyhedron.from_Hrep(
            A=np.array([[1.], [-1.]]),
            b=np.array([u_max_original, -u_min_original])
        )
        
        # Step 5: Compute tightened constraints X_tilde = X - E, U_tilde = U - K_robust*E
        # Note: The tube controller u = v + K_robust*(x-z) implies v must satisfy U_tilde
        KE = K_robust @ E
        X_tilde = X - E
        U_tilde = U - KE
        
        # Store tightened constraints
        self.X_tilde = X_tilde
        self.U_tilde = U_tilde
        
        # Step 6: Compute Terminal Set and Cost using a separate LQR controller
        # This controller (K_mpc) is used for the terminal set invariance condition
        # TUNING: Can be different from robust controller, usually for performance
        Q_mpc = np.diag([10, 50.0]) 
        R_mpc =  1.0 * np.eye(self.nu)
        K_mpc, Qf_mpc, _ = dlqr(self.A, self.B, Q_mpc, R_mpc)
        K_mpc = -K_mpc
        
        self.Qf = Qf_mpc
        self.Q = Q_mpc
        self.R = R_mpc
        
        A_cl_mpc = self.A + self.B @ K_mpc

        # Step 7: Compute terminal set Xf_tilde
        # Xf_tilde must be positively invariant under u = K_mpc*x and satisfy X_tilde and U_tilde
        
        X_tilde_and_KU_tilde = X_tilde.intersect(
            Polyhedron.from_Hrep(U_tilde.A @ K_mpc, U_tilde.b)
        )
        
        # Check if non-empty by trying to compute vertices
        try:
            _ = X_tilde_and_KU_tilde.V
            Xf_tilde = self._compute_max_invariant_set(A_cl_mpc, X_tilde_and_KU_tilde)
        except Exception as e:
            print(f"  WARNING: Cannot compute Xf_tilde - {e}")
            print(f"  Using X_tilde as fallback...")
            Xf_tilde = X_tilde  # Fallback
        
        self.Xf_tilde = Xf_tilde
        
        # Call base class to setup the optimization problem
        MPCControl_base._setup_controller(self)
        
        # YOUR CODE HERE
        #################################################
    
    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Override base class get_u to apply Tube MPC control law."""
        # Call base class to solve the optimization problem and get v0, z, v trajectories
        u0_nominal, x_traj, u_traj = super().get_u(x0, x_target, u_target)
        
        
        
        # Compute actual control (in delta form)
        delta_u0 = u0_nominal + self.K @ (x0-self.xs - x_traj[:, 0])
        
        # Convert to absolute control
        u0 = delta_u0 + self.us
        
        # Also convert trajectories to absolute form
        x_traj_abs = x_traj + self.xs.reshape(-1, 1)
        u_traj_abs = u_traj + self.us.reshape(-1, 1)
        
        return u0, x_traj_abs, u_traj_abs

    def plot_sets(self):
        import matplotlib.pyplot as plt
        
        # Create a figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot E (Invariant Set)
        try:
            if self.E.V.size > 0:
                self.E.plot(ax=ax1, color='g', alpha=0.5)
                ax1.set_title("Minimal Robust Invariant Set (E)")
            else:
                ax1.text(0.5, 0.5, "E is empty or unbounded", ha='center')
        except Exception as e:
            print(f"Could not plot E: {e}")
            
        ax1.set_xlabel("z_vel")
        ax1.set_ylabel("z_pos")
        ax1.grid(True)
        
        # Plot Xf_tilde (Terminal Set) and X_tilde (Constraints)
        # X_tilde is likely unbounded (half-space), so we intersect with a box for plotting
        # Box: +/- 20 for both states
        try:
            box = Polyhedron.from_Hrep(
                np.vstack([np.eye(2), -np.eye(2)]),
                np.array([20, 20, 20, 20])
            )
            X_tilde_bounded = self.X_tilde.intersect(box)
            
            if X_tilde_bounded.V.size > 0:
                X_tilde_bounded.plot(ax=ax2, color='b', alpha=0.1, linestyle='--')
            else:
                print("X_tilde_bounded has no vertices")
                
            if self.Xf_tilde.V.size > 0:
                self.Xf_tilde.plot(ax=ax2, color='r', alpha=0.6)
            else:
                print("Xf_tilde has no vertices")
                
        except Exception as e:
            print(f"Could not plot X_tilde/Xf_tilde: {e}")
        
        ax2.set_title("Terminal Set (Red) & Constraints (Blue)")
        ax2.set_xlabel("z_vel")
        ax2.set_ylabel("z_pos")
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def print_sets(self):
        print("=== Minimal Robust Invariant Set (E) ===")
        print(f"A shape: {self.E.A.shape}, b shape: {self.E.b.shape}")
        print("A:\n", self.E.A)
        print("b:\n", self.E.b)
        print("\n")

        print("=== Tightened State Constraints (X_tilde) ===")
        print(f"A shape: {self.X_tilde.A.shape}, b shape: {self.X_tilde.b.shape}")
        print("A:\n", self.X_tilde.A)
        print("b:\n", self.X_tilde.b)
        print("\n")

        print("=== Tightened Input Constraints (U_tilde) ===")
        print(f"A shape: {self.U_tilde.A.shape}, b shape: {self.U_tilde.b.shape}")
        print("A:\n", self.U_tilde.A)
        print("b:\n", self.U_tilde.b)
        print("\n")

        print("=== Terminal Set (Xf_tilde) ===")
        print(f"A shape: {self.Xf_tilde.A.shape}, b shape: {self.Xf_tilde.b.shape}")
        print("A:\n", self.Xf_tilde.A)
        print("b:\n", self.Xf_tilde.b)
        print("\n")


