# Deliverable 6.1: Robust Tube MPC for Landing

## 1. Design Procedure and Tuning

To ensure a safe landing in the presence of disturbances (model mismatch, wind, etc.), we implemented a Robust Tube MPC controller for the z-subsystem. The core idea is to separate the control into two parts:
1.  **Nominal Controller (MPC):** Computes a trajectory $z$ and input $v$ for the nominal (undisturbed) system that satisfies tightened constraints.
2.  **Ancillary Controller (Feedback):** Keeps the actual state $x$ close to the nominal state $z$ by rejecting disturbances. The control law is $u = v + K_{robust}(x - z)$.

### Separation of Controllers ($K_{robust}$ vs $K_{mpc}$)

We explicitly separated the feedback gain used for disturbance rejection ($K_{robust}$) from the terminal cost/set gain used in the MPC optimization ($K_{mpc}$).

*   **$K_{robust}$ (Ancillary Controller):**
    *   **Purpose:** To define the "tube" (minimal robust invariant set $E$) and keep the system inside it.
    *   **Tuning:** We chose aggressive weights ($Q_{robust} = \text{diag}([50, 200])$, $R_{robust} = 0.1$) to generate a strong feedback gain. A "stiffer" controller results in a smaller invariant set $E$, which means less tightening of the constraints is required, leaving more room for the MPC to operate.
    
*   **$K_{mpc}$ (Terminal Controller):**
    *   **Purpose:** To define the terminal cost $Q_f$ and the terminal set $X_f$ for the nominal MPC problem.
    *   **Tuning:** We used weights focused on performance and stability ($Q_{mpc} = \text{diag}([10, 50])$, $R_{mpc} = 1.0$). This ensures the nominal trajectory converges smoothly to the target without being overly aggressive.

### Class Structure (`MPCControl_base`)

We structured the code using a base class `MPCControl_base` to handle the common MPC setup (optimization problem definition, discretization) and derived classes like `MPCControl_z` for specific subsystem logic.
*   **`MPCControl_z`**: Handles the robust specific steps: computing $K_{robust}$, the invariant set $E$, the tightened constraints $\tilde{X}, \tilde{U}$, and the terminal set $\tilde{X}_f$. It then calls `_setup_controller` from the base class.
*   **`MPCControl_base`**: Implements the generic Tube MPC optimization problem:
    *   Variables: Nominal state $z$ and input $v$.
    *   Initial Constraint: $x_0 \in z_0 \oplus E$ (implemented as $E.A(x_0 - z_0) \le E.b$).
    *   Dynamics: $z_{k+1} = A z_k + B v_k$.
    *   Tightened Constraints: $z_k \in \tilde{X}$, $v_k \in \tilde{U}$.
    *   Terminal Constraint: $z_N \in \tilde{X}_f$.

## 2. Invariant Sets and Constraints

*   **Minimal Robust Invariant Set ($E$):**
    This set represents the bound on the error $e = x - z$ under the ancillary controller. By tuning $K_{robust}$ aggressively, we kept this set small, ensuring that the "tube" around the nominal trajectory is tight.
    
*   **Tightened Constraints ($\tilde{U}$):**
    The input constraints for the nominal system must be tightened to reserve authority for the ancillary controller.
    *   Original Input: $40\% \le P_{avg} \le 80\%$
    *   **Tightened Input ($\tilde{U}$):** The available range for the nominal input $v$ is reduced by the maximum possible feedback action $K_{robust} e$ for $e \in E$. This ensures that even with the maximum correction applied, the total input $u = v + K_{robust}e$ remains within the physical limits $[40\%, 80\%]$.

*   **Terminal Set ($\tilde{X}_f$):**
    The terminal set is the maximal invariant set for the nominal system under the terminal controller $K_{mpc}$, subject to the tightened constraints. This ensures recursive feasibility.

## 3. Simulation Results

We simulated the landing maneuver from $z=10$m to $z=3$m under two disturbance scenarios:
1.  **Random Disturbance:** The controller successfully rejects the noise and tracks the reference. The state stays within the tube.
2.  **Extreme Disturbance:** Even with maximum disturbance, the robust controller maintains stability and constraint satisfaction (z position stays positive, input stays within bounds).

The settling time is well within the 4-second requirement for the random disturbance case, thanks to the efficient MPC tuning. In the extreme disturbance case, the convergence is slightly slower, but this is the necessary trade-off to robustly compensate for the full range of disturbances.
