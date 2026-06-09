# Spinning-Wing Drone — Research Repository

A novel UAV design that uses **rotating wings for attitude control and aerodynamic lift augmentation**. This repository contains the simulation models, aerodynamic analysis, hardware designs, and experimental data accompanying the research paper.

---

## Repository Structure

```
spinning-wing-drone/
├── simulation/          # MATLAB dynamics model + MuJoCo physics simulation
├── aerodynamics/        # Airfoil polar data and selection analysis
├── hardware/            # CAD models (wings, frame, propellers) and firmware
└── experiments/         # Flight telemetry and thrust measurement data
```

---

## Project Overview

The drone achieves yaw control and attitude stabilization through spinning wing surfaces rather than conventional differential thrust alone. Key contributions:

- Analytical dynamics model with rotating-wing gyroscopic effects
- LQR and backstepping controller implementations in MuJoCo
- Airfoil selection via XFOIL polar sweep across multiple Reynolds numbers
- Hardware-in-the-loop validation on a custom quadrotor frame with PX4 (Holybro KakuteH7 Mini)

---

## Dependencies

| Component | Requirements |
|-----------|-------------|
| MATLAB simulation | MATLAB R2022b+ |
| MuJoCo simulation | Python 3.10+, `mujoco`, `numpy`, `scipy` |
| Aerodynamics analysis | Python 3.10+, `numpy`, `matplotlib`, `pandas` |
| Telemetry parsing | Python 3.10+, `pymavlink`, `pandas`, `matplotlib` |

---

## Subdirectory READMEs

Each folder has its own README with details on scripts, expected inputs/outputs, and what requires path fixes.
