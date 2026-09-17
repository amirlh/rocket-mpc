# Deliverable 6.2: Merged Controller for Landing

## 1. Design Procedure and Tuning

In this part, we merged the Robust Tube MPC controller for the z-subsystem (designed in Part 6.1) with three nominal MPC controllers for the x, y, and roll subsystems. The goal is to perform a complete landing maneuver from $x=3, y=2, z=10$ to $x=1, y=0, z=3$.

### Controller Architecture

We used the `MPCLandControl` class to coordinate the four independent controllers:
1.  **`MPCControl_z` (Robust Tube MPC):** Handles the vertical descent. It ensures robust constraint satisfaction ($z \ge 0$, $40\% \le P_{avg} \le 80\%$) despite disturbances.
2.  **`MPCControl_x`, `MPCControl_y`, `MPCControl_roll` (Nominal MPCs):** Handle the horizontal position and orientation. These are standard MPC controllers designed for the linearized subsystems.

### Tuning of Nominal Controllers

For the x, y, and roll controllers, we used the same `MPCControl_base` structure but configured for nominal MPC (no tube, $K_{robust}=0$, $E=\{0\}$).

*   **Constraints:**
    *   **State:** $|\alpha| \le 10^\circ$, $|\beta| \le 10^\circ$ (to keep linearization valid).
    *   **Input:** $|\delta_1| \le 15^\circ$, $|\delta_2| \le 15^\circ$, $|P_{diff}| \le 20\%$.
    *   **Soft Constraints:** We enabled soft constraints (`soft_constraints=True`) for the state constraints in these nominal controllers. This is crucial because the nonlinear reality might slightly violate the linear predictions, and we want to avoid infeasibility crashes in the solver.

*   **Weights:**
    *   **X/Y:** $Q = \text{diag}([10, 10, 10, 50])$ (Higher weight on position to ensure convergence), $R = 1.0$.
    *   **Roll:** $Q = \text{diag}([10, 50])$, $R = 1.0$.



## 2. Simulation Results

We simulated the full nonlinear system performing the landing maneuver.

*   **Performance:** The rocket successfully navigates from the starting point to the target. The z-controller robustly manages the descent, while the x and y controllers steer the rocket horizontally.
*   **Settling Time:** The settling time is within the 4-second requirement. The aggressive tuning of the z-controller ensures a fast descent, while the x/y controllers are tuned to be responsive enough to correct the initial position error within the same timeframe.
*   **Constraint Satisfaction:** The critical constraints (z-height and throttle limits) are respected thanks to the robust formulation of the z-controller. The orientation angles remain within the linear region limits.

## 3. Code Structure

*   **`MPCLandControl.py`**: The main coordinator class.
*   **`MPCControl_x.py`, `_y.py`, `_roll.py`**: Implement the nominal MPCs using the shared `MPCControl_base`. They set `self.K = 0` and `self.E = None` to disable the tube logic in the base class.
*   **`MPCControl_z.py`**: Implements the robust tube MPC (as described in 6.1).
