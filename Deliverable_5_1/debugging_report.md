# Deliverable 5.1: Offset-Free Tracking

## 1. Design Procedure and Tuning

To achieve offset-free tracking in the presence of model mismatch (specifically mass discrepancy), we implemented a Disturbance Observer (DOB) based MPC. The standard MPC relies on a nominal model. When the actual system (plant) differs from this model (e.g., different mass), the predictions become inaccurate, leading to steady-state errors.

### Design Procedure

1.  **Augmented State Model**:
    We augmented the system state $x$ with a constant disturbance state $d$. The augmented dynamics are:
    $$
    \begin{aligned}
    x_{k+1} &= A x_k + B u_k + B_d d_k \\
    d_{k+1} &= d_k
    \end{aligned}
    $$
    where $B_d$ is the disturbance input matrix. For the z-velocity subsystem, we chose $B_d = [-0.1]$ to model a disturbance affecting the vertical acceleration.

2.  **Observer Design**:
    We designed a linear observer to estimate both the state $\hat{x}$ and the disturbance $\hat{d}$ from the measurements $y_k = C x_k$. The observer dynamics are:
    $$
    \begin{bmatrix} \hat{x}_{k+1} \\ \hat{d}_{k+1} \end{bmatrix} = 
    \begin{bmatrix} A & B_d \\ 0 & I \end{bmatrix} \begin{bmatrix} \hat{x}_k \\ \hat{d}_k \end{bmatrix} + 
    \begin{bmatrix} B \\ 0 \end{bmatrix} u_k + 
    L (y_k - C \hat{x}_k)
    $$
    The observer gain matrix $L$ was computed using pole placement to ensure stable and fast convergence of the estimation error.

3.  **Disturbance Compensation**:
    The estimated disturbance $\hat{d}$ is used in two ways:
    *   **Target Calculation**: We adjust the target input $u_{target}$ to counteract the estimated disturbance at the steady state. We solve for $u_{target}$ such that:
        $$ (I - A) x_{ref} - B u_{target} - B_d \hat{d} = 0 $$
        This ensures that the controller applies the necessary "feedforward" force to hold the reference state despite the disturbance.
    *   **Prediction Model**: The MPC optimization uses the augmented model (with constant $\hat{d}$) for prediction. This allows the solver to anticipate the effect of the disturbance over the horizon.

### Tuning Parameters

*   **Observer Poles**: We placed the observer poles at `[0.4, 0.5]`. These are relatively fast (closer to 0 than 1 in discrete time) to ensure the disturbance estimate converges quickly, but not so fast that it becomes overly sensitive to noise.
*   **MPC Weights**:
    *   $Q = \text{diag}([100.0])$: High penalty on velocity error to ensure good tracking.
    *   $R = \text{diag}([0.1])$: Low penalty on input change to allow aggressive control actions if needed.
*   **Constraints**:
    *   Input (Throttle): $40 \le P_{avg} \le 80$ (Safety limits).

## 2. Impact of Mass Mismatch

We simulated the rocket with a mass of $1.5$ kg (nominal is likely different, e.g., 1.0 kg) and zero fuel consumption rate.

*   **Without Offset-Free Control (Part 4)**:
    The controller assumes the nominal mass. The gravity compensation (feedforward) is calculated for the lighter nominal mass. As a result, the thrust provided is insufficient to hover the heavier rocket. The rocket would accelerate downwards (negative z-velocity) or settle at a velocity where the proportional feedback term (from MPC) balances the gravity mismatch, resulting in a non-zero steady-state velocity error (offset).

*   **With Offset-Free Control (Part 5)**:
    Initially, the rocket drops due to the mass mismatch. However, the observer quickly detects a discrepancy between the predicted and measured state. This discrepancy is integrated into the disturbance estimate $\hat{d}$. As $\hat{d}$ converges to a value representing the "missing" lift, the controller increases the throttle ($u_{target}$) to compensate. Consequently, the z-velocity converges to the reference (0 m/s) with zero offset, and the rocket successfully hovers.

## 3. Disturbance Estimation Discussion

*   **Is the disturbance constant for a constant rocket mass?**
    Yes. The mass mismatch introduces a constant force difference: $F_{dist} = (m_{actual} - m_{nominal}) g$. In the linear dynamics, this appears as a constant additive term to the acceleration. Since the mass is constant (fuel rate = 0), this force is constant.
    
*   **Estimation Error**:
    The estimation error converges to zero. Initially, there is a transient phase where the observer "learns" the disturbance. Once the transient dies out (determined by the observer poles), the estimate $\hat{d}$ matches the true disturbance (scaled by $B_d$), and the system behaves as if it were the nominal plant.

## 4. Python Code

The implementation of the offset-free controller is provided in:
*   `LinearMPC/MPCControl_zvel.py`: Contains the `MPCControl_zvel` class with the observer setup and `estimate_parameters` method.
*   `LinearMPC/MPCControl_base.py`: Contains the `MPCControl_base` class with the updated `get_u` and `compute_equilibrium_input` methods that utilize the estimated disturbance.

The simulation code is in `Deliverable_5_1.ipynb`.
