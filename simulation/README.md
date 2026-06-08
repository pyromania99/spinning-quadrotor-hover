# Simulation

Two parallel simulation environments: a MATLAB analytical model and a MuJoCo physics simulation.

---

## MATLAB (`matlab/`)

Analytical dynamics model derived from first principles. Integrates the equations of motion via ODE solvers.

### Scripts (`matlab/scripts/`)

| File | Description |
|------|-------------|
| `UAV_Project.m` | Main entry point — sets up parameters and runs the simulation |
| `drone_rhs.m` | Right-hand side of the drone ODEs (equations of motion) |
| `drone_rhs_compensated.m` | ODE RHS with gyroscopic compensation from spinning wings |
| `drone_rhs_disturbed.m` | ODE RHS with disturbance injection for robustness testing |
| `position_control_pid.m` | PID position controller |
| `eigen_test.m` | Eigenvalue analysis of the linearised system |
| `eigen_sweep_r.m` | Sweep over spin rate `r` to study stability margin |
| `hold_tilt.m` / `change_tilt.m` / `compare_change_tilt.m` | Wing tilt angle studies |
| `No_tilt_all_clock_motor_fail.m` | Motor failure simulation (all-clockwise config) |
| `Without_wing_z_only.m` / `without_wings_pos.m` | Baseline without wings for comparison |
| `wings_pos.m` | Simulation with wings active |
| `yaw_diagram.m` | Yaw torque diagram generation |
| `hex_M_test.m` | Hexrotor mixer matrix test |

**To run:** Open `UAV_Project.m` in MATLAB and run. Other scripts are functions called by the main script or standalone studies.

### Results (`matlab/results/`)

Pre-generated PNG outputs. These are **not regenerated** unless you re-run the scripts. Key outputs:

- `Attitude_NED.png` — Attitude response in NED frame
- `Motor_fail_redundancy_config_3*.png` — Motor failure redundancy analysis
- `Trajectory_post_motor_failure.png` — Positional recovery after motor loss
- `Z_pos_tilt.png`, `motor_variation_tilt.png` — Effect of wing tilt on altitude and motor loads

> `.fig` and `.mat` files are excluded from this repo (MATLAB binary format, large).

---

## MuJoCo (`mujoco/`)

Physics simulation using DeepMind MuJoCo. Supports real-time viewer and headless batch runs.

### Models (`mujoco/models/`)

| File | Description |
|------|-------------|
| `quad.xml` | Standard quadrotor model |
| `quad_spinning.xml` | Quadrotor with spinning wing surfaces attached |

### Scripts (`mujoco/scripts/`)

| File | Description |
|------|-------------|
| `sim.py` | Basic PD controller simulation — good starting point |
| `LQR.py` | LQR controller; supports motor failure modes (`"none"`, `"one"`, `"two"`, `"three"`) |
| `backstepping_controller.py` | Backstepping controller class implementing the paper's control law |
| `sim_spinning_wing.py` | Full simulation with spinning wing dynamics |
| `tilt_sim.py` | Tilt angle sweep simulation |
| `sweep_tilt.py` | Batch sweep over tilt configurations |
| `eigen_Sim.py` / `eigen_so3.py` | Eigenvalue analysis in simulation |
| `anlysis.py` | Post-run trajectory and state analysis |
| `test_kinematics.py` / `test_rotation.py` | Unit-level kinematic tests |
| `interactive_transform.py` | Interactive SO(3) transform visualiser |
| `extract_values.py` | Extract specific state values from sim logs |
| `sim_backup.py` | Backup copy of sim.py (older version) |

**⚠️ Path fix required:** Scripts hardcode `mujoco\\quad.xml`. Change to the correct relative path before running:
```python
# Change this:
model = mujoco.MjModel.from_xml_path("mujoco\\quad.xml")
# To (from repo root):
model = mujoco.MjModel.from_xml_path("simulation/mujoco/models/quad.xml")
```

**To run a basic simulation:**
```bash
cd spinning-wing-drone
pip install mujoco numpy scipy
python simulation/mujoco/scripts/sim.py
```

### Results (`mujoco/results/`)

Pre-generated outputs from simulation runs. Includes:

- `sim_log.json`, `sim_log_so3.json`, `sim_log_spinning.json` — Raw simulation state logs
- `sweep_tilt_results.json` — Tilt sweep batch results
- `analysis_summary.csv` — Tabular summary of analysis runs
- `spinning_drone_bemt_analysis.png` — BEMT (Blade Element Momentum Theory) analysis
- `MUJOCO_LOG.TXT`, `CONTROL_LAW_EXPANSION.txt` — Run notes and derivation notes

> Dated subdirectories (`2026-02-18_06-09/`, etc.) contain timestamped run outputs.
