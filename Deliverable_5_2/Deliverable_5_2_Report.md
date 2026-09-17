# Deliverable 5.2: Time-Varying Mass and Offset-Free Tracking

## 1. Simulation with Time-Varying Mass

We simulated the rocket with an initial mass of $2.0$ kg (nominal $1.0$ kg) and a fuel consumption rate of $0.1$ kg/s. This creates a scenario where the system parameters (mass) are continuously changing, introducing a time-varying disturbance.

## 2. Initial Tracking Offset

**Observation:**
In the first few seconds of the simulation, we observe a small but persistent tracking offset in the z-velocity, despite the presence of the disturbance estimator.

**Explanation:**
The disturbance observer we designed assumes a **constant disturbance** model:
$$ d_{k+1} = d_k $$
However, the linearly decreasing mass creates a **ramp-like disturbance** (the gravitational force difference changes linearly).
$$ F_{dist}(t) = (m(t) - m_{nom})g = (m_0 - \dot{m}t - m_{nom})g $$
A standard observer designed for constant disturbances will exhibit a steady-state error (lag) when tracking a ramp signal. The estimator is always "chasing" the true disturbance value.

**Minimizing the Offset:**
It is important to note that this offset is minimized if the estimator dynamics are fast. In our tuning (poles at `[0.4, 0.5]`), the estimator is relatively fast, which keeps this tracking error minimal, as observed in the simulation plots. The offset is present but small enough that the system remains stable and performs well.

## 3. Proposed Estimator Modification

To achieve true offset-free tracking for a linearly changing mass (ramp disturbance), we would need to augment the disturbance model to include the **rate of change** of the disturbance.

**Modified Model:**
We can introduce a second disturbance state $\delta$ representing the slope:
$$
\begin{aligned}
d_{k+1} &= d_k + T_s \cdot \delta_k \\
\delta_{k+1} &= \delta_k
\end{aligned}
$$
By estimating both the disturbance $d$ and its rate of change $\delta$, the MPC could anticipate the future evolution of the disturbance and compensate for the ramp without lag.

## 4. Trajectory Behaviors

Throughout the simulation, we observe distinct phases:
1.  **Initial Transient:** The rocket drops significantly due to the large initial mass mismatch (2.0 kg vs 1.0 kg). The estimator quickly identifies the large disturbance, and the controller increases thrust to recover.
2.  **Quasi-Steady State (Drift):** As the mass decreases linearly, the required thrust ($P_{avg}$) decreases linearly. The estimated disturbance $\hat{d}$ also trends downwards linearly. The system maintains a near-hover state, but with the small lag mentioned above.

## 5. End of Simulation Behavior

Towards the end of the simulation, we observe an unexpected behavior: **Uncontrollable Ascent due to Minimum Thrust Constraint**.

As the fuel is consumed, the total mass of the rocket decreases significantly. Eventually, the rocket becomes so light that the thrust produced at the **minimum throttle constraint** ($P_{avg} = 40\%$) exceeds the gravitational force acting on the rocket ($F_{thrust, min} > m(t) \cdot g$).

Since the controller is hard-constrained to keep $P_{avg} \ge 40\%$, it cannot reduce the thrust further to maintain a hover. Consequently, the rocket begins to accelerate upwards (take off) uncontrollably, violating the velocity setpoint because the physical/safety constraints prevent the necessary control action.
