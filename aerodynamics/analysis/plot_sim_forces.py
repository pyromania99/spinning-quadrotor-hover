#!/usr/bin/env python3
"""
Spinning Drone Force & Power Analysis — Publication-Quality Figures
===================================================================

Loads sim_log_spinning.json and produces 9 publication-quality figures:

  1. Torque Balance (time-series: motor yaw torque vs wing drag torque)
  2. Lift Budget   (stacked area: wing + motor vertical vs weight)
  3. Wing Aero Performance (lift, drag torque, L/D ratio)
  4. State Convergence  (angular velocity + altitude vs time)
  5. Power Breakdown    (stacked bar chart comparison)
  5b. Cumulative Energy (energy consumption over time)
  6. Power Efficiency   (time-series: wing power, motor power, total)
  7. Induced Velocity   (v_i and disk loading vs time)
  8. Summary Dashboard  (compact multi-panel overview)

All plots saved to results/<date>_<time>/ at 300 dpi.
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from datetime import datetime

# ── Publication style ─────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 20,
    'axes.labelsize': 15,
    'axes.titlesize': 19,
    'legend.fontsize': 18,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.2,
    'text.usetex': False,
})

# Colour palette (colour-blind friendly, academic)
C = {
    'blue':      '#2166ac',
    'red':       '#b2182b',
    'green':     '#1b7837',
    'orange':    '#e08214',
    'purple':    '#7b3294',
    'grey':      '#636363',
    'lightblue': '#92c5de',
    'lightred':  '#fddbc7',
    'lightgreen':'#a6dba0',
}

# ── Constants (must match sim_spinning_wing.py & anlysis.py) ──────────────────
DRONE_WEIGHT_KG = 0.800
GRAVITY = 9.81
DRONE_WEIGHT_N = DRONE_WEIGHT_KG * GRAVITY  # 7.848 N

MOTOR_RADIUS = 0.40       # m
NUM_MOTORS = 4
MOTOR_KM = 0.015          # prop reaction-torque / thrust ratio (m)
MOTOR_PROP_DIAMETER = 0.10
MOTOR_EFFICIENCY = 0.65
RHO = 1.225               # kg/m³

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def load_sim_log(path='sim_log_spinning.json'):
    """Load the simulation log JSON produced by sim_spinning_wing.py."""
    script_dir = Path(__file__).parent
    with open(script_dir / path) as f:
        data = json.load(f)
    return {k: np.array(v) for k, v in data.items()}


def steady_state_slice(log, last_frac=0.15):
    """Return a boolean mask for the last `last_frac` of the sim (steady-state)."""
    t = log['time']
    return t >= t[-1] * (1 - last_frac)


def running_mean(x, N=50):
    """Causal running mean (no look-ahead) for smoothing noisy time-series."""
    cs = np.cumsum(np.insert(x, 0, 0))
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i + 1 - N)
        out[i] = (cs[i + 1] - cs[lo]) / (i + 1 - lo)
    return out


def motor_torque_series(log):
    """Compute the motor yaw torque time-series from thrust and tilt."""
    T = log['motor_thrust_total']
    tilt_rad = np.radians(log['motor_tilt_deg'])
    thrust_component = T * np.sin(tilt_rad) * MOTOR_RADIUS
    reaction_component = MOTOR_KM * T * np.cos(tilt_rad)
    return thrust_component, reaction_component, thrust_component + reaction_component


def make_output_dir():
    """Create results/<date>_<time>/ and return the Path."""
    now = datetime.now()
    name = now.strftime('%Y-%m-%d_%H-%M')
    out = Path(__file__).parent / 'results' / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def annotate_steady_state(ax, t, ss_mask, text='steady state', color='grey'):
    """Add a subtle vertical span marking the steady-state region."""
    t_ss_start = t[ss_mask][0]
    ax.axvspan(t_ss_start, t[-1], alpha=0.06, color=color, zorder=0)
    ax.axvline(t_ss_start, ls=':', lw=0.7, color=color, alpha=0.5)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Torque Balance Time-Series
#   Motor yaw torque (thrust + reaction) vs wing drag torque over time
# ═══════════════════════════════════════════════════════════════════════════════
def plot_torque_balance(log, out_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    t = log['time']
    ss = steady_state_slice(log)
    N_smooth = 80

    # Compute motor torque components
    tq_thrust, tq_reaction, tq_motor = motor_torque_series(log)
    tq_wing = log['wing_torque']

    # Top panel: torque overlay - wing drag vs negative motor torque (should cancel)
    ax1.plot(t, running_mean(tq_wing, N_smooth), color=C['orange'], lw=1.4,
             label='Wing drag torque')
    ax1.plot(t, running_mean(-tq_motor, N_smooth), color=C['blue'], lw=1.4,
             label='−Motor yaw torque')

    mean_wing = tq_wing[ss].mean()
    mean_motor = tq_motor[ss].mean()
    mean_sum = tq_wing[ss].mean() - tq_motor[ss].mean()
    ax1.axhline(mean_wing, ls='--', lw=0.8, color=C['orange'], alpha=0.6)
    ax1.axhline(-mean_motor, ls='--', lw=0.8, color=C['blue'], alpha=0.6)
    annotate_steady_state(ax1, t, ss)

    ax1.set_ylabel('Torque  (N·m)')
    ax1.set_title('Yaw Torque Balance')
    ax1.legend(loc='lower right', framealpha=0.9, fontsize=11)

    # Annotate steady-state averages
    ax1.annotate(
        f'SS avg:  wing = {mean_wing:.3f} N·m\n'
        f'         −motor = {-mean_motor:.3f} N·m',
        xy=(0.98, 0.96), xycoords='axes fraction', va='top', ha='right', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.1', fc='white', ec=C['grey'], alpha=0.8))

    # Bottom panel: residual (sum of wing drag and negative motor torque)
    residual = running_mean(tq_wing + tq_motor, N_smooth)
    ax2.plot(t, residual, color=C['grey'], lw=0.9)
    ax2.axhline(0, ls='-', lw=0.5, color='black', alpha=0.3)
    ax2.fill_between(t, 0, residual, alpha=0.15, color=C['grey'])
    annotate_steady_state(ax2, t, ss)

    mean_res = (tq_wing + tq_motor)[ss].mean()
    ax2.set_ylabel('Residual (N·m)')
    ax2.set_xlabel('Time  (s)')
    ax2.annotate(f'Mean residual = {mean_res:+.4f} N·m',
                 xy=(0.98, 0.92), xycoords='axes fraction',
                 ha='right', va='top', fontsize=11, color=C['grey'])

    fig.tight_layout()
    fig.savefig(out_dir / '1_torque_balance.png')
    print('  ✓ 1_torque_balance.png')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Lift Budget (Stacked Area)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_lift_budget(log, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t = log['time']
    ss = steady_state_slice(log)
    N_smooth = 80

    tilt_rad = np.radians(log['motor_tilt_deg'])
    motor_vert = log['motor_thrust_total'] * np.cos(tilt_rad)
    wing_lift = log['wing_thrust']
    total_lift = wing_lift + motor_vert

    wl = running_mean(wing_lift, N_smooth)
    mv = running_mean(motor_vert, N_smooth)
    tl = running_mean(total_lift, N_smooth)

    ax.fill_between(t, 0, wl, alpha=0.35, color=C['green'], label='Wing lift')
    ax.fill_between(t, wl, wl + mv, alpha=0.35, color=C['blue'],
                    label='Motor vertical')
    ax.plot(t, tl, lw=1.0, color=C['blue'], alpha=0.5)
    ax.axhline(DRONE_WEIGHT_N, color=C['red'], ls='--', lw=1.3,
               label=f'Weight = {DRONE_WEIGHT_N:.2f} N')

    annotate_steady_state(ax, t, ss)

    # Steady-state percentages
    mean_wing = wing_lift[ss].mean()
    mean_motor_v = motor_vert[ss].mean()
    wing_pct = mean_wing / DRONE_WEIGHT_N * 100
    motor_pct = mean_motor_v / DRONE_WEIGHT_N * 100
    ax.annotate(
        f'Wing:  {mean_wing:.2f} N  ({wing_pct:.0f}%)\n'
        f'Motor: {mean_motor_v:.2f} N  ({motor_pct:.0f}%)',
        xy=(0.98, 0.55), xycoords='axes fraction', ha='right', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=C['grey'], alpha=0.8))

    ax.set_xlabel('Time  (s)')
    ax.set_ylabel('Vertical Force  (N)')
    ax.set_title('Vertical Lift Budget')
    ax.legend(loc='lower right', framealpha=0.9, fontsize=11)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(out_dir / '2_lift_budget.png')
    print('  ✓ 2_lift_budget.png')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Wing Aerodynamic Performance
# ═══════════════════════════════════════════════════════════════════════════════
def plot_wing_aero(log, out_dir):
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    t = log['time']
    ss = steady_state_slice(log)
    N_smooth = 80

    wing_lift = log['wing_thrust']
    wing_torque = log['wing_torque']
    omega = np.abs(log['omega_z'])

    # Approximate drag force from torque: F_drag ≈ torque / r_mean
    r_mean = 0.5 * (0.35 + 0.65)  # mean wing radius
    wing_drag = wing_torque / r_mean  # approximate tangential drag force

    # L/D ratio (use torque-based drag to avoid radius ambiguity)
    # L/D ≈ lift / (drag_torque / r_mean), only valid when omega > 0
    valid = omega > 0.5
    ld_ratio = np.where(valid, wing_lift / np.maximum(wing_drag, 1e-6), 0)

    # Panel 1: Wing lift
    ax = axes[0]
    ax.plot(t, running_mean(wing_lift, N_smooth), color=C['green'], lw=1.3)
    mean_lift = wing_lift[ss].mean()
    ax.axhline(mean_lift, ls='--', lw=0.8, color=C['red'],
               label=f'SS avg = {mean_lift:.2f} N')
    ax.axhline(DRONE_WEIGHT_N, ls=':', lw=0.8, color=C['grey'],
               label=f'Weight = {DRONE_WEIGHT_N:.2f} N')
    annotate_steady_state(ax, t, ss)
    ax.set_ylabel('Wing Lift  (N)')
    ax.set_title('Wing Aerodynamic Performance', fontweight='bold')
    ax.legend(loc='lower right', fontsize=16, framealpha=0.9)

    # Panel 2: Wing drag torque
    ax = axes[1]
    ax.plot(t, running_mean(wing_torque, N_smooth), color=C['orange'], lw=1.3)
    mean_tq = wing_torque[ss].mean()
    ax.axhline(mean_tq, ls='--', lw=0.8, color=C['red'],
               label=f'SS avg = {mean_tq:.3f} N·m')
    annotate_steady_state(ax, t, ss)
    ax.set_ylabel('Drag Torque  (N·m)')
    ax.legend(loc='lower right', fontsize=16, framealpha=0.9)

    # Panel 3: Lift-to-drag ratio
    ax = axes[2]
    ax.plot(t, running_mean(ld_ratio, N_smooth * 2), color=C['purple'], lw=1.3)
    mean_ld = ld_ratio[ss].mean()
    ax.axhline(mean_ld, ls='--', lw=0.8, color=C['red'],
               label=f'SS avg L/D = {mean_ld:.1f}')
    annotate_steady_state(ax, t, ss)
    ax.set_ylabel('L/D Ratio')
    ax.set_xlabel('Time  (s)')
    ax.legend(loc='lower right', fontsize=16, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_dir / '3_wing_aero.png')
    print('  ✓ 3_wing_aero.png')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — State Convergence (ω and altitude)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_state_convergence(log, out_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    t = log['time']
    ss = steady_state_slice(log)
    N_smooth = 40

    # Panel 1: Angular velocity
    omega = np.abs(log['omega_z'])
    ax1.plot(t, running_mean(omega, N_smooth), color=C['blue'], lw=1.3)
    omega_ss = omega[ss].mean()
    omega_rpm = omega_ss * 60 / (2 * np.pi)
    ax1.axhline(omega_ss, ls='--', lw=0.8, color=C['red'],
                label=f'SS avg = {omega_ss:.2f} rad/s  ({omega_rpm:.0f} RPM)')
    annotate_steady_state(ax1, t, ss)
    ax1.set_ylabel('|ω|  (rad/s)')
    ax1.set_title('State Convergence')
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.9)

    # Panel 2: Altitude
    z = log['pos_z']
    ax2.plot(t, z, color=C['green'], lw=1.0,)
    ax2.axhline(1.5, ls='--', lw=0.8, color=C['red'], label='Target z = 1.5 m')
    z_ss = z[ss].mean()
    ax2.axhline(z_ss, ls=':', lw=0.8, color=C['blue'],
                label=f'SS avg = {z_ss:.3f} m')
    annotate_steady_state(ax2, t, ss)
    ax2.set_ylabel('Altitude  (m)')
    ax2.set_xlabel('Time  (s)')
    ax2.legend(loc='lower right', fontsize=11, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_dir / '4_state_convergence.png')
    print('  ✓ 4_state_convergence.png')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Power Breakdown (Waterfall Chart)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_power_waterfall(log, out_dir):
    ss = steady_state_slice(log)
    omega_ss = np.abs(log['omega_z'][ss].mean())
    T_total = log['motor_thrust_total'][ss].mean()
    tilt_deg = log['motor_tilt_deg'][ss].mean()
    tilt_rad = np.radians(tilt_deg)

    # Motor mechanical power components
    tq_thrust, tq_reaction, tq_motor_total = motor_torque_series(log)
    motor_yaw_power_mech = tq_motor_total[ss].mean() * omega_ss
    
    motor_disk_area = np.pi * (MOTOR_PROP_DIAMETER / 2) ** 2 * NUM_MOTORS
    v_i_motor = np.sqrt(max(T_total, 0) / (2 * RHO * motor_disk_area))
    motor_thrust_power_mech = T_total * v_i_motor
    
    # Motor electrical power (convert mechanical to electrical via efficiency)
    motor_total_mech = motor_thrust_power_mech + motor_yaw_power_mech
    motor_total_elec = motor_total_mech / MOTOR_EFFICIENCY
    
    # For display: show components
    motor_thrust_power_elec = motor_thrust_power_mech / MOTOR_EFFICIENCY
    motor_yaw_power_elec = motor_yaw_power_mech / MOTOR_EFFICIENCY

    # Total system electrical power
    total_system_power = motor_total_elec

    # Motor-only baseline
    motor_only_v_i = np.sqrt(DRONE_WEIGHT_N / (2 * RHO * motor_disk_area))
    motor_only_power = DRONE_WEIGHT_N * motor_only_v_i / MOTOR_EFFICIENCY

    # --- Stacked Bar Chart ---
    fig, ax = plt.subplots(figsize=(7, 5.5))

    categories = ['Motor Power\n(Thrust + Torque)', 'Motor-Only\nBaseline']
    x_pos = np.arange(len(categories))
    
    # First bar: stacked (yaw on bottom, thrust on top)
    bar1_bottom = ax.bar(x_pos[0], motor_yaw_power_elec, color=C['purple'], 
                         edgecolor='white', linewidth=1.5, width=0.5, zorder=3,
                         label='Motor Yaw Power')
    bar1_top = ax.bar(x_pos[0], motor_thrust_power_elec, 
                      bottom=motor_yaw_power_elec, color=C['blue'],
                      edgecolor='white', linewidth=1.5, width=0.5, zorder=3,
                      label='Motor Thrust Power')
    
    # Second bar: motor-only baseline
    bar2 = ax.bar(x_pos[1], motor_only_power, color=C['red'], 
                  edgecolor='white', linewidth=1.5, width=0.5, zorder=3,
                  label='Baseline (no wing)')
    
    # Cover the bottom white edge by drawing colored rectangles at the base
    from matplotlib.patches import Rectangle
    bar_width = 0.49
    edge_cover_height = 2.0  # Should be slightly larger than linewidth
    
    # Cover bottom edge of first bar (purple)
    rect1 = Rectangle((x_pos[0] - bar_width/2, 0), bar_width, edge_cover_height,
                      facecolor=C['purple'], edgecolor='none', zorder=4)
    ax.add_patch(rect1)
    
    # Cover bottom edge of second bar (red)
    rect2 = Rectangle((x_pos[1] - bar_width/2, 0), bar_width, edge_cover_height,
                      facecolor=C['red'], edgecolor='none', zorder=4)
    ax.add_patch(rect2)

    # Value annotations
    # Yaw power component
    ax.text(x_pos[0], motor_yaw_power_elec / 2, 
            f'{motor_yaw_power_elec:.1f} W\n(yaw)', 
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='white')
    
    # Thrust power component
    ax.text(x_pos[0], motor_yaw_power_elec + motor_thrust_power_elec / 2,
            f'{motor_thrust_power_elec:.1f} W\n(thrust)',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='white')
    
    # Total on first bar
    ax.text(x_pos[0], total_system_power + max(total_system_power, motor_only_power) * 0.02,
            f'Total: {total_system_power:.1f} W',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Motor-only value
    ax.text(x_pos[1], motor_only_power + max(total_system_power, motor_only_power) * 0.02,
            f'{motor_only_power:.1f} W',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    
    # Add visual appeal with subtle shadow/3D effect
    ax.set_facecolor('#fafafa')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    
    # Savings annotation - more prominent and cleaner
    savings_pct = (motor_only_power - total_system_power) / motor_only_power * 100

  
    ax.set_ylabel('Power  (W)', fontsize=11, fontweight='bold')
    ax.set_title(
        f'Power Budget Comparison at Steady State  '
        f'(ω = {omega_ss:.1f} rad/s,  tilt = {tilt_deg:.0f}°)',
        fontweight='bold', fontsize=12)

    fig.tight_layout()
    fig.savefig(out_dir / '5_power_waterfall.png')
    print('  ✓ 5_power_waterfall.png')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5B — Cumulative Energy Consumption
# ═══════════════════════════════════════════════════════════════════════════════
def plot_cumulative_energy(log, out_dir):
    """Plot cumulative energy consumption over time for spinning wing vs motor-only."""
    fig, ax = plt.subplots(figsize=(8, 5))
    t = log['time']
    ss = steady_state_slice(log)
    
    omega = np.abs(log['omega_z'])
    
    # Calculate instantaneous power for spinning wing system
    _, _, tq_motor = motor_torque_series(log)
    motor_yaw_power_mech = tq_motor * omega
    
    T = log['motor_thrust_total']
    motor_disk_area = np.pi * (MOTOR_PROP_DIAMETER / 2) ** 2 * NUM_MOTORS
    v_i_motors = np.sqrt(np.maximum(T, 0) / (2 * RHO * motor_disk_area))
    motor_thrust_power_mech = T * v_i_motors
    
    motor_total_mech = motor_thrust_power_mech + motor_yaw_power_mech
    motor_total_elec = motor_total_mech / MOTOR_EFFICIENCY
    
    # Calculate motor-only baseline power (constant)
    motor_only_v_i = np.sqrt(DRONE_WEIGHT_N / (2 * RHO * motor_disk_area))
    motor_only_power = DRONE_WEIGHT_N * motor_only_v_i / MOTOR_EFFICIENCY
    motor_only_power_series = np.full_like(t, motor_only_power)
    
    # Compute cumulative energy (integrate power over time)
    # Energy in Joules (Watt-seconds)
    dt = np.diff(t, prepend=t[0])
    cumulative_spinning = np.cumsum(motor_total_elec * dt)
    cumulative_motor_only = np.cumsum(motor_only_power_series * dt)
    
    # Convert to Watt-hours for better readability
    cumulative_spinning_wh = cumulative_spinning / 3600
    cumulative_motor_only_wh = cumulative_motor_only / 3600
    
    # Plot cumulative energy
    ax.plot(t, cumulative_motor_only_wh, color=C['red'], lw=2.5, 
            label='Motor-Only Baseline', linestyle='--', alpha=0.8)
    ax.plot(t, cumulative_spinning_wh, color=C['blue'], lw=2.5,
            label='Spinning Wing System')
    
    # Fill area between curves to show savings
    ax.fill_between(t, cumulative_spinning_wh, cumulative_motor_only_wh,
                   where=(cumulative_motor_only_wh >= cumulative_spinning_wh),
                   alpha=0.2, color=C['green'], label='Energy Saved')
    
    annotate_steady_state(ax, t, ss)
    
    # Final energy values
    final_spinning = cumulative_spinning_wh[-1]
    final_motor_only = cumulative_motor_only_wh[-1]
    energy_saved = final_motor_only - final_spinning
    savings_pct = (energy_saved / final_motor_only) * 100
    
    # Annotation with final values
    note = (f'Total Energy Consumed:\n'
            f'  Spinning Wing: {final_spinning:.2f} Wh\n'
            f'  Motor-Only:    {final_motor_only:.2f} Wh\n'
            f'\n'
            f'Energy Saved: {energy_saved:.2f} Wh ({savings_pct:+.1f}%)')
    ax.annotate(note, xy=(0.98, 0.05), xycoords='axes fraction',
               ha='right', va='bottom', fontsize=20,
               bbox=dict(boxstyle='round,pad=0.5', fc='white', 
                        ec=C['grey'], lw=1.5, alpha=0.95))
    
    ax.set_xlabel('Time  (s)', fontsize=22, fontweight='bold')
    ax.set_ylabel('Cumulative Energy  (Wh)', fontsize=22, fontweight='bold')
    ax.set_title('Cumulative Energy Consumption Comparison', 
                fontweight='bold', fontsize=24)
    ax.legend(loc='upper left', fontsize=20, framealpha=0.95)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    fig.tight_layout()
    fig.savefig(out_dir / '5b_cumulative_energy.png')
    print('  ✓ 5b_cumulative_energy.png')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Power Time-Series
# ═══════════════════════════════════════════════════════════════════════════════
def plot_power_timeseries(log, out_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    t = log['time']
    ss = steady_state_slice(log)
    N_smooth = 80

    omega = np.abs(log['omega_z'])

    # Motor mechanical power components
    _, _, tq_motor = motor_torque_series(log)
    motor_yaw_power_mech = tq_motor * omega
    
    T = log['motor_thrust_total']
    motor_disk_area = np.pi * (MOTOR_PROP_DIAMETER / 2) ** 2 * NUM_MOTORS
    v_i_motors = np.sqrt(np.maximum(T, 0) / (2 * RHO * motor_disk_area))
    motor_thrust_power_mech = T * v_i_motors
    
    # Motor electrical power
    motor_total_mech = motor_thrust_power_mech + motor_yaw_power_mech
    motor_total_elec = motor_total_mech / MOTOR_EFFICIENCY
    
    # For display
    motor_yaw_power_elec = motor_yaw_power_mech / MOTOR_EFFICIENCY
    motor_thrust_power_elec = motor_thrust_power_mech / MOTOR_EFFICIENCY
    
    total_power = motor_total_elec

    # Motor-only baseline (constant)
    motor_only_v_i = np.sqrt(DRONE_WEIGHT_N / (2 * RHO * motor_disk_area))
    motor_only_power = DRONE_WEIGHT_N * motor_only_v_i / MOTOR_EFFICIENCY

    # Top panel: power curves
    ax1.plot(t, running_mean(motor_yaw_power_elec, N_smooth), color=C['purple'], lw=1.3,
             label='Motor yaw power (elec)')
    ax1.plot(t, running_mean(motor_thrust_power_elec, N_smooth), color=C['blue'], lw=1.3,
             label='Motor thrust power (elec)')
    ax1.plot(t, running_mean(total_power, N_smooth), color=C['orange'], lw=1.5,
             label='Total motor power (elec)')
    ax1.axhline(motor_only_power, ls='--', lw=1.2, color=C['red'],
                label=f'Motor-only baseline = {motor_only_power:.1f} W')
    annotate_steady_state(ax1, t, ss)

    ax1.set_ylabel('Power  (W)')
    ax1.set_title('Power over Time', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax1.set_ylim(bottom=0)

    # Bottom panel: instantaneous savings %
    savings = (motor_only_power - total_power) / motor_only_power * 100
    ax2.plot(t, running_mean(savings, N_smooth * 2), color=C['purple'], lw=1.1)
    ax2.axhline(0, ls='-', lw=0.5, color='black', alpha=0.3)
    mean_savings = savings[ss].mean()
    ax2.axhline(mean_savings, ls='--', lw=0.8, color=C['red'],
                label=f'SS avg = {mean_savings:+.1f}%')
    annotate_steady_state(ax2, t, ss)
    ax2.set_ylabel('Power Saving  (%)')
    ax2.set_xlabel('Time  (s)')
    ax2.legend(loc='lower right', fontsize=16, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_dir / '6_power_timeseries.png')
    print('  ✓ 6_power_timeseries.png')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Induced Velocity & Disk Loading
# ═══════════════════════════════════════════════════════════════════════════════
def plot_induced_velocity(log, out_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    t = log['time']
    ss = steady_state_slice(log)
    N_smooth = 60

    # Induced velocity
    vi = log['v_induced']
    ax1.plot(t, running_mean(vi, N_smooth), color=C['blue'], lw=1.3)
    vi_ss = vi[ss].mean()
    ax1.axhline(vi_ss, ls='--', lw=0.8, color=C['red'],
                label=f'SS avg = {vi_ss:.3f} m/s')
    annotate_steady_state(ax1, t, ss)
    ax1.set_ylabel('Induced Velocity  (m/s)')
    ax1.set_title('Wing Induced Flow', fontweight='bold')
    ax1.legend(loc='lower right', fontsize=16, framealpha=0.9)

    # Disk loading = Thrust / Disk area
    wing_thrust = log['wing_thrust']
    r_tip = 0.65
    A_disk = np.pi * r_tip**2
    disk_loading = wing_thrust / A_disk  # N/m²
    ax2.plot(t, running_mean(disk_loading, N_smooth), color=C['green'], lw=1.3)
    dl_ss = disk_loading[ss].mean()
    ax2.axhline(dl_ss, ls='--', lw=0.8, color=C['red'],
                label=f'SS avg = {dl_ss:.1f} N/m²')
    annotate_steady_state(ax2, t, ss)
    ax2.set_ylabel('Disk Loading  (N/m²)')
    ax2.set_xlabel('Time  (s)')
    ax2.legend(loc='lower right', fontsize=16, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_dir / '7_induced_velocity.png')
    print('  ✓ 7_induced_velocity.png')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Summary Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
def plot_dashboard(log, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle('Spinning Drone Simulation — Steady-State Summary',
                 fontweight='bold', fontsize=26, y=0.98)
    t = log['time']
    ss = steady_state_slice(log)
    N_smooth = 80

    omega = np.abs(log['omega_z'])
    tilt_rad = np.radians(log['motor_tilt_deg'])
    T = log['motor_thrust_total']
    wing_lift = log['wing_thrust']
    wing_torque = log['wing_torque']
    motor_vert = T * np.cos(tilt_rad)
    vi = log['v_induced']

    # Steady-state values
    omega_ss = omega[ss].mean()
    T_ss = T[ss].mean()
    tilt_ss = log['motor_tilt_deg'][ss].mean()

    # (0,0) Omega
    ax = axes[0, 0]
    ax.plot(t, running_mean(omega, N_smooth), color=C['blue'], lw=1.0)
    ax.axhline(omega_ss, ls='--', lw=0.7, color=C['red'])
    ax.set_ylabel('ω  (rad/s)')
    ax.set_title(f'ω = {omega_ss:.1f} rad/s', fontsize=20)

    # (0,1) Altitude
    ax = axes[0, 1]
    z = log['pos_z']
    ax.plot(t, z, color=C['green'], lw=0.8)
    ax.axhline(1.5, ls='--', lw=0.7, color=C['red'])
    ax.set_ylabel('z  (m)')
    ax.set_title(f'z_ss = {z[ss].mean():.3f} m', fontsize=20)

    # (0,2) Lift pie chart
    ax = axes[0, 2]
    mean_wing = wing_lift[ss].mean()
    mean_motor = motor_vert[ss].mean()
    sizes = [mean_wing, mean_motor]
    labels = [f'Wing\n{mean_wing:.2f} N', f'Motor\n{mean_motor:.2f} N']
    wedges, texts = ax.pie(sizes, labels=labels,
                           colors=[C['green'], C['blue']],
                           startangle=90,
                           wedgeprops=dict(edgecolor='white', linewidth=1.5))
    ax.set_title(f'Lift Share ({mean_wing/DRONE_WEIGHT_N*100:.0f}% / '
                 f'{mean_motor/DRONE_WEIGHT_N*100:.0f}%)', fontsize=20)

    # (1,0) Wing torque
    ax = axes[1, 0]
    ax.plot(t, running_mean(wing_torque, N_smooth), color=C['orange'], lw=1.0)
    tq_ss = wing_torque[ss].mean()
    ax.axhline(tq_ss, ls='--', lw=0.7, color=C['red'])
    ax.set_ylabel('torque (N·m)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'τ_drag = {tq_ss:.3f} N·m', fontsize=20)

    # (1,1) Power bar
    ax = axes[1, 1]
    
    # Motor mechanical power components
    _, _, tq_motor_ss = motor_torque_series(log)
    motor_yaw_power_mech = tq_motor_ss[ss].mean() * omega_ss
    
    motor_disk_area = np.pi * (MOTOR_PROP_DIAMETER / 2) ** 2 * NUM_MOTORS
    v_i_mot = np.sqrt(max(T_ss, 0) / (2 * RHO * motor_disk_area))
    motor_thrust_power_mech = T_ss * v_i_mot
    
    # Motor electrical power
    motor_total_mech = motor_thrust_power_mech + motor_yaw_power_mech
    motor_total_elec = motor_total_mech / MOTOR_EFFICIENCY
    
    # For display
    motor_yaw_P = motor_yaw_power_mech / MOTOR_EFFICIENCY
    motor_thrust_P = motor_thrust_power_mech / MOTOR_EFFICIENCY
    
    baseline_v_i = np.sqrt(DRONE_WEIGHT_N / (2 * RHO * motor_disk_area))
    baseline_P = DRONE_WEIGHT_N * baseline_v_i / MOTOR_EFFICIENCY

    labels_p = ['Yaw', 'Thrust', 'Total', 'Baseline']
    vals = [motor_yaw_P, motor_thrust_P, motor_total_elec, baseline_P]
    cols = [C['purple'], C['blue'], C['orange'], C['red']]
    bars = ax.bar(labels_p, vals, color=cols, edgecolor='white', width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{v:.1f}', ha='center', fontsize=16, fontweight='bold')
    ax.set_ylabel('Power (W)')
    ax.set_xlabel('')
    savings = (baseline_P - motor_total_elec) / baseline_P * 100
    ax.set_title(f'Motor Power (elec)  ({savings:+.0f}% saving)', fontsize=20)

    # (1,2) Key parameters table
    ax = axes[1, 2]
    ax.axis('off')
    table_data = [
        ['Weight', f'{DRONE_WEIGHT_N:.2f} N'],
        ['ω (steady)', f'{omega_ss:.2f} rad/s'],
        ['Motor tilt', f'{tilt_ss:.0f}°'],
        ['Motor thrust', f'{T_ss:.2f} N'],
        ['Wing lift', f'{mean_wing:.2f} N ({mean_wing/DRONE_WEIGHT_N*100:.0f}%)'],
        ['Wing drag τ', f'{tq_ss:.3f} N·m'],
        ['v_induced', f'{vi[ss].mean():.3f} m/s'],
        ['Motor yaw P', f'{motor_yaw_P:.1f} W'],
        ['Motor thrust P', f'{motor_thrust_P:.1f} W'],
        ['Total motor P', f'{motor_total_elec:.1f} W'],
        ['Baseline', f'{baseline_P:.1f} W'],
        ['Saving', f'{savings:+.1f}%'],
    ]
    table = ax.table(cellText=table_data, colLabels=['Parameter', 'Value'],
                     loc='center', cellLoc='left',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    table.scale(1, 1.25)
    # Style header
    for j in range(2):
        table[0, j].set_facecolor(C['blue'])
        table[0, j].set_text_props(color='white', fontweight='bold')
    ax.set_title('Key Parameters', fontsize=20)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_dir / '8_dashboard.png')
    print('  ✓ 8_dashboard.png')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print('Loading sim_log_spinning.json …')
    log = load_sim_log()
    print(f'  {len(log["time"])} timesteps, {log["time"][-1]:.1f} s total')

    out = make_output_dir()
    print(f'Output → {out}\n')

    plot_torque_balance(log, out)
    plot_lift_budget(log, out)
    plot_wing_aero(log, out)
    plot_state_convergence(log, out)
    plot_power_waterfall(log, out)
    plot_cumulative_energy(log, out)
    plot_power_timeseries(log, out)
    plot_induced_velocity(log, out)
    plot_dashboard(log, out)

    print(f'\nDone — 9 figures in {out}')


if __name__ == '__main__':
    main()
