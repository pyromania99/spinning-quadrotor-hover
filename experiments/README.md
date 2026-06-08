# Experiments

Real-hardware flight data and bench measurements. This directory contains **data outputs and analysis scripts** — not code that can be re-run without the original flight hardware.

---

## Telemetry (`telemetry/`)

MAVLink telemetry logs captured from the PX4 flight controller over a companion computer connection. The drone streams attitude, rates, motor commands, and sensor data via MAVLink at ~50 Hz.

### Data (`telemetry/data/`)

| File/Pattern | Description |
|--------------|-------------|
| `mavlink_data_YYYYMMDD_HHMMSS.csv` | Parsed MAVLink logs per flight session |
| `Normal_hover_yaw_disp.xlsx` | Yaw displacement during normal hover baseline |
| `*.png` | Pre-generated attitude and signal plots |

Flight sessions included:
- `8_17_spin`, `8_22_spin` — Initial spin experiments
- `8_28_spin_1cant`, `8_29_spin_1cant_mkill` — Single-wing cant angle with motor kill
- `9_09_spin_1cant`, `11_09_spin_1cant` — Later spin/cant iterations

> Raw `.ulg` PX4 log files are not included (binary format, large). CSV files are pre-parsed exports.

### Scripts (`telemetry/scripts/`)

| Script | Description |
|--------|-------------|
| `telem.py` | Main MAVLink log parser — converts raw MAVLink stream to CSV |
| `servo_telem_feb_26.py` | Parses servo/actuator telemetry from Feb 2026 flights |
| `plot.py` | Standard telemetry plotter (attitude, rates, motor outputs) |
| `plot_gpt.py` | Alternative plot layout |
| `plot_test.py` | Quick test plot for a single CSV |
| `all_msgs.py` | Enumerate all MAVLink message types in a log |
| `calculate_yaw_rate.py` | Compute yaw rate from attitude quaternion data |
| `debug_mavlink.py` | Debug helper for MAVLink parsing issues |
| `test_animation.py` | Animated attitude visualisation |

**To parse a raw MAVLink log:**
```bash
pip install pymavlink pandas matplotlib
python experiments/telemetry/scripts/telem.py <path_to_log>
```

**⚠️ Path note:** Scripts reference input paths that no longer exist in this archive. Update the file path at the top of each script.

---

## Thrust Measurement (`thrust_measurement/`)

Static thrust bench tests measuring rotor thrust vs throttle command, used to calibrate the thrust coefficient `kf` in simulation.

### Data (`thrust_measurement/data/`)

| File | Description |
|------|-------------|
| `ani.xlsx`, `ani.2.xlsx`, `ani 3.xlsx` | Thrust measurement tables (different test sessions) |
| `mapping.xlsx` | PWM–to–thrust mapping calibration |
| `mappings.png` | Thrust curve plot |
| `hover_signals.png` | Motor signal values at hover thrust |

The `motor_mount.jpg` image documents the load cell fixture setup. The Arduino sketch used to read the load cell is in `rig_firmware/ld_cell2.ino`.

**Measured parameters:**
- Thrust coefficient `kf` ≈ 2.5×10⁻⁵ N/(rad/s)²
- Torque coefficient `km` ≈ 1.0×10⁻⁶ Nm/(rad/s)²
- Motor: Emax ECO Micro 1404 6000KV
- Propeller: Gemfan 3" 2-blade
