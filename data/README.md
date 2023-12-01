# Datasets from Gymnasium

We made three datasets with Gymnasium + PPO Algorithm.
Our target environments are **Cart Pole**, **Mountain Car**, and **Pendulum**. 
## 1. CartPole
### Observation Space

| Num   | Observation              | Min                    | Max                  |
|-------|--------------------------|------------------------|----------------------|
|   0   |   **Cart Position**      |   -4.8                 |   4.8                |
|   1   |   **Cart Velocity**      |   -Inf                 |   Inf                | 
|   2   |   **Pole Angle**         |   ~ -0.418 rad (-24°)  |   ~ 0.418 rad (24°)  |
|   3   | **Pole Angular Velocity**|   -Inf                 |   Inf                |

### Action Space
The action is a `ndarray` with shape `(1,)`
- 0: Push cart to the left
- 1: Push cart to the right


## 2. Mountain Car
### Observation Space
| Num | Observation                              | Min   | Max  | Unit         |
|-----|------------------------------------------|-------|------|--------------|
| 0   | **position** of the car along the x-axis | -1.2  | 0.6  | position (m) |
| 1   | **velocity** of the car                  | -0.07 | 0.07 | velocity (v) |
### Action Space
There are 3 discrete deterministic actions:

- 0: Accelerate to the left
- 1: Don’t accelerate
- 2: Accelerate to the right

## 3. Pendulum
### Observation Space
| Num | Observation                | Min  | Max |
|-----|----------------------------|------|-----|
| 0   | $x = \mathrm{cos}(\theta)$ | -1.0 | 1.0 |
| 1   | $y = \mathrm{sin}(\theta)$ | -1.0 | 1.0 |
| 2   | **Angular Velocity**       | -8.0 | 8.0 |

### Action Space
| Num | Action     | Min  | Max |
|-----|------------|------|-----|
| 0   | **Torque** | -2.0 | 2.0 |