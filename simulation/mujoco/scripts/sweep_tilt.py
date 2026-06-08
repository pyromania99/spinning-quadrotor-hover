#!/usr/bin/env python3
"""
Tilt Angle Sweep — Headless MuJoCo simulations

For each tilt angle, runs the spinning drone sim (no viewer) until
steady-state is reached, then records equilibrium:
  omega, motor thrust per motor, wing thrust, wing torque, total power

Outputs a table + matplotlib plots.
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Re-use everything from the main sim
from sim_spinning_wing import (
    SpinningDroneSim, WingAeroModel,
    RHO, GRAVITY, DRONE_MASS, DRONE_WEIGHT, MU,
    MOTOR_RADIUS, NUM_MOTORS, NUM_BLADES,
    KP_ALT, KD_ALT,
)
import mujoco


def run_headless(tilt_deg, duration=20.0, settle_window=3.0):
    """Run one headless sim at the given tilt angle.

    Returns dict with steady-state values averaged over the last
    `settle_window` seconds, or None if it diverged.
    """
    sim = SpinningDroneSim()

    # Monkey-patch the controller to use the requested tilt
    original_controller = sim.spin_controller

    def patched_controller(state, wing_result):
        thrusts, tilts = original_controller(state, wing_result)
        # Override tilt (thrust is re-derived from vertical_needed / cos(tilt))
        return thrusts, tilts  # we patch the tilt inside the controller itself

    # Instead of monkey-patching, directly override the tilt constant used
    # inside the controller.  We'll replace spin_controller entirely with a
    # local closure that mirrors the original but uses our tilt_deg.
    def custom_controller(state, wing_result):
        pos = state['pos']
        vel = state['vel']
        wing_thrust = wing_result['total_thrust']

        z_error = sim.target_altitude - pos[2]
        z_vel = vel[2]

        vertical_needed = DRONE_WEIGHT - wing_thrust + KP_ALT * z_error - KD_ALT * z_vel

        ramp = min(1.0, sim.data.time / 1.0)
        vertical_needed = DRONE_WEIGHT + (vertical_needed - DRONE_WEIGHT) * ramp
        vertical_needed = max(vertical_needed, 0.0)

        tilt_rad = np.radians(tilt_deg)
        cos_tilt = np.cos(tilt_rad)

        thrust_total = vertical_needed / cos_tilt if cos_tilt > 0.01 else 0.0
        thrust_per_motor = np.clip(thrust_total / NUM_MOTORS, 0.0, 4.0)

        thrusts = np.array([thrust_per_motor] * 4)
        tilts = np.array([tilt_deg] * 4)
        return thrusts, tilts

    sim.spin_controller = custom_controller

    # Run headless (no viewer)
    n_steps = int(duration / sim.model.opt.timestep)
    for _ in range(n_steps):
        sim.data.qfrc_applied[:] = 0.0
        state = sim.get_state()
        omega_z = state['omega_z']
        wing_result = sim.wing_model.compute_forces(omega_z)
        sim.apply_wing_forces(wing_result)
        thrusts, tilts = sim.spin_controller(state, wing_result)
        sim.apply_motor_forces(thrusts, tilts)
        mujoco.mj_step(sim.model, sim.data)

        # Log
        sim.log_data['time'].append(sim.data.time)
        sim.log_data['omega_z'].append(omega_z)
        sim.log_data['pos_z'].append(state['pos'][2])
        sim.log_data['wing_thrust'].append(wing_result['total_thrust'])
        sim.log_data['wing_torque'].append(wing_result['total_torque'])
        sim.log_data['wing_power'].append(wing_result['power'])
        sim.log_data['motor_thrust_total'].append(sum(thrusts))
        sim.log_data['motor_1'].append(thrusts[0])

        # Bail out early if the drone crashed or diverged
        if abs(state['pos'][2]) > 50 or np.any(np.isnan(state['pos'])):
            print(f"  tilt={tilt_deg:5.1f}°  DIVERGED at t={sim.data.time:.1f}s")
            return None

    # Extract steady-state from last settle_window seconds
    times = np.array(sim.log_data['time'])
    mask = times >= (duration - settle_window)
    if mask.sum() < 10:
        return None

    result = {
        'tilt_deg': tilt_deg,
        'omega': np.mean(np.array(sim.log_data['omega_z'])[mask]),
        'omega_std': np.std(np.array(sim.log_data['omega_z'])[mask]),
        'altitude': np.mean(np.array(sim.log_data['pos_z'])[mask]),
        'motor_thrust_total': np.mean(np.array(sim.log_data['motor_thrust_total'])[mask]),
        'motor_per': np.mean(np.array(sim.log_data['motor_1'])[mask]),
        'wing_thrust': np.mean(np.array(sim.log_data['wing_thrust'])[mask]),
        'wing_torque': np.mean(np.array(sim.log_data['wing_torque'])[mask]),
        'wing_power': np.mean(np.array(sim.log_data['wing_power'])[mask]),
    }

    # Derived quantities
    cos_t = np.cos(np.radians(tilt_deg))
    sin_t = np.sin(np.radians(tilt_deg))
    result['motor_vertical'] = result['motor_thrust_total'] * cos_t
    result['motor_tangential'] = result['motor_thrust_total'] * sin_t
    result['motor_torque'] = result['motor_tangential'] * MOTOR_RADIUS
    result['total_vertical'] = result['motor_vertical'] + result['wing_thrust']
    result['motor_power'] = result['motor_torque'] * abs(result['omega'])
    result['total_power'] = result['wing_power'] + result['motor_power']
    result['wing_fraction'] = (result['wing_thrust'] / DRONE_WEIGHT * 100
                               if DRONE_WEIGHT > 0 else 0)

    return result


def main():
    tilt_angles = np.arange(10, 71, 5)  # 10° to 70° in 5° steps
    results = []

    print("=" * 80)
    print("  TILT ANGLE SWEEP  (headless, 20 s each)")
    print("=" * 80)
    print(f"  Angles: {tilt_angles[0]:.0f}° – {tilt_angles[-1]:.0f}°  ({len(tilt_angles)} runs)")
    print(f"  Weight: {DRONE_WEIGHT:.2f} N")
    print()

    for tilt in tilt_angles:
        print(f"  Running tilt = {tilt:5.1f}° ... ", end="", flush=True)
        res = run_headless(tilt)
        if res is not None:
            results.append(res)
            print(f"ω = {res['omega']:6.2f} rad/s  |  "
                  f"Motor = {res['motor_thrust_total']:5.2f} N  |  "
                  f"Wing = {res['wing_thrust']:5.2f} N  "
                  f"({res['wing_fraction']:.0f}% of W)")
        else:
            print("FAILED / DIVERGED")

    if not results:
        print("\nNo successful runs!")
        return

    # ====================================================================
    # PRINT TABLE
    # ====================================================================
    print("\n" + "=" * 110)
    print(f"{'Tilt':>6} | {'ω (rad/s)':>10} | {'ω (RPM)':>8} | "
          f"{'Motor T':>8} | {'Wing T':>8} | {'Wing %':>7} | "
          f"{'Alt (m)':>7} | {'W Power':>8} | {'M Power':>8} | {'Total P':>8}")
    print("-" * 110)
    for r in results:
        omega_rpm = r['omega'] * 60 / (2 * np.pi)
        print(f"{r['tilt_deg']:5.0f}° | {r['omega']:10.2f} | {omega_rpm:8.1f} | "
              f"{r['motor_thrust_total']:7.2f}N | {r['wing_thrust']:7.2f}N | "
              f"{r['wing_fraction']:6.1f}% | {r['altitude']:7.2f} | "
              f"{r['wing_power']:7.2f}W | {r['motor_power']:7.2f}W | "
              f"{r['total_power']:7.2f}W")
    print("=" * 110)

    # ====================================================================
    # SAVE RAW DATA
    # ====================================================================
    # Convert numpy types to native Python for JSON serialization
    def to_native(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_clean = [{k: to_native(v) for k, v in r.items()} for r in results]

    script_dir = Path(__file__).parent
    with open(script_dir / 'sweep_tilt_results.json', 'w') as f:
        json.dump(results_clean, f, indent=2)
    print(f"\nRaw data saved to sweep_tilt_results.json")

    # ====================================================================
    # PLOT
    # ====================================================================
    tilts = [r['tilt_deg'] for r in results]
    omegas = [r['omega'] for r in results]
    motor_t = [r['motor_thrust_total'] for r in results]
    wing_t = [r['wing_thrust'] for r in results]
    wing_pct = [r['wing_fraction'] for r in results]
    alts = [r['altitude'] for r in results]
    w_power = [r['wing_power'] for r in results]
    m_power = [r['motor_power'] for r in results]
    t_power = [r['total_power'] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Tilt Angle Sweep — Steady-State Results', fontsize=14, fontweight='bold')

    # 1 — Omega vs tilt
    ax = axes[0, 0]
    ax.plot(tilts, omegas, 'o-', color='tab:blue', lw=2)
    ax.set_xlabel('Motor Tilt (°)')
    ax.set_ylabel('ω (rad/s)')
    ax.set_title('Spin Rate vs Tilt')
    ax.grid(True, alpha=0.3)

    # 2 — Motor thrust vs tilt
    ax = axes[0, 1]
    ax.plot(tilts, motor_t, 's-', color='tab:red', lw=2, label='Motor total')
    ax.plot(tilts, wing_t, '^-', color='tab:green', lw=2, label='Wing thrust')
    ax.axhline(DRONE_WEIGHT, ls='--', color='gray', alpha=0.6, label=f'Weight ({DRONE_WEIGHT:.1f} N)')
    ax.set_xlabel('Motor Tilt (°)')
    ax.set_ylabel('Thrust (N)')
    ax.set_title('Thrust Contributions vs Tilt')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3 — Wing lift fraction vs tilt
    ax = axes[0, 2]
    ax.plot(tilts, wing_pct, 'D-', color='tab:purple', lw=2)
    ax.set_xlabel('Motor Tilt (°)')
    ax.set_ylabel('Wing Lift (% of weight)')
    ax.set_title('Wing Lift Fraction vs Tilt')
    ax.grid(True, alpha=0.3)

    # 4 — Altitude vs tilt
    ax = axes[1, 0]
    ax.plot(tilts, alts, 'o-', color='tab:orange', lw=2)
    ax.axhline(1.5, ls='--', color='gray', alpha=0.6, label='Target (1.5 m)')
    ax.set_xlabel('Motor Tilt (°)')
    ax.set_ylabel('Altitude (m)')
    ax.set_title('Steady-State Altitude vs Tilt')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 5 — Power breakdown vs tilt
    ax = axes[1, 1]
    ax.plot(tilts, m_power, 's-', color='tab:red', lw=2, label='Motor power')
    ax.plot(tilts, w_power, '^-', color='tab:green', lw=2, label='Wing drag power')
    ax.plot(tilts, t_power, 'o-', color='k', lw=2, label='Total power')
    ax.set_xlabel('Motor Tilt (°)')
    ax.set_ylabel('Power (W)')
    ax.set_title('Power vs Tilt')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 6 — Motor thrust per motor vs tilt
    ax = axes[1, 2]
    motor_per = [r['motor_per'] for r in results]
    ax.plot(tilts, motor_per, 'o-', color='tab:cyan', lw=2)
    ax.axhline(4.0, ls='--', color='red', alpha=0.6, label='Max (4 N)')
    ax.set_xlabel('Motor Tilt (°)')
    ax.set_ylabel('Thrust per Motor (N)')
    ax.set_title('Per-Motor Thrust vs Tilt')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = script_dir / 'sweep_tilt_plot.png'
    plt.savefig(fig_path, dpi=150)
    print(f"Plot saved to {fig_path}")
    plt.show()


if __name__ == "__main__":
    main()
