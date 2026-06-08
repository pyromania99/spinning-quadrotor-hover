import json
import numpy as np

# Load simulation data
with open('sim_log_spinning.json', 'r') as f:
    d = json.load(f)

# Steady-state mask (last 15% of simulation)
t = np.array(d['time'])
m = t >= t[-1] * 0.85

print('=== ACCURATE STEADY-STATE VALUES ===')
omega = np.abs(np.array(d['omega_z'])[m].mean())
print(f'omega: {omega:.3f} rad/s ({omega*60/(2*np.pi):.1f} RPM)')

T = np.array(d['motor_thrust_total'])[m].mean()
print(f'motor_thrust_total: {T:.3f} N')

tilt_deg = np.array(d['motor_tilt_deg'])[m].mean()
print(f'motor_tilt: {tilt_deg:.2f} deg')

tilt_rad = np.radians(tilt_deg)
motor_vert = T * np.cos(tilt_rad)
print(f'motor_vert: {motor_vert:.3f} N')

wing_lift = np.array(d['wing_thrust'])[m].mean()
print(f'wing_lift: {wing_lift:.3f} N')
print(f'wing_lift_pct: {wing_lift/7.85*100:.1f}%')
print(f'motor_vert_pct: {motor_vert/7.85*100:.1f}%')

wing_torque = np.array(d['wing_torque'])[m].mean()
print(f'wing_torque: {wing_torque:.4f} N·m')

wing_power = np.array(d['wing_power'])[m].mean()
print(f'wing_power: {wing_power:.2f} W')

v_induced = np.array(d['v_induced'])[m].mean()
print(f'v_induced: {v_induced:.3f} m/s')

altitude = np.array(d['pos_z'])[m].mean()
print(f'altitude: {altitude:.3f} m')

# L/D ratio calculation
r_mean = 0.5  # mean radius in meters
wing_drag_approx = wing_torque / r_mean
ld_ratio = wing_lift / wing_drag_approx
print(f'L/D_approx: {ld_ratio:.2f}')

print()
print('=== POWER (ELECTRICAL) ===')
tq_motor = T * np.sin(tilt_rad) * 0.4 + 0.015 * T * np.cos(tilt_rad)
print(f'motor_yaw_torque: {tq_motor:.4f} N·m')

motor_yaw_mech = tq_motor * omega
print(f'motor_yaw_power_mech: {motor_yaw_mech:.2f} W')

motor_yaw_elec = motor_yaw_mech / 0.65
print(f'motor_yaw_power_elec: {motor_yaw_elec:.2f} W')

vi_motor = np.sqrt(T / (2 * 1.225 * np.pi * (0.05)**2 * 4))
print(f'v_i_motor: {vi_motor:.3f} m/s')

motor_thrust_mech = T * vi_motor
print(f'motor_thrust_power_mech: {motor_thrust_mech:.2f} W')

motor_thrust_elec = motor_thrust_mech / 0.65
print(f'motor_thrust_power_elec: {motor_thrust_elec:.2f} W')

total_elec = motor_yaw_elec + motor_thrust_elec
print(f'total_motor_elec: {total_elec:.2f} W')

baseline_vi = np.sqrt(7.85 / (2 * 1.225 * np.pi * (0.05)**2 * 4))
baseline_elec = 7.85 * baseline_vi / 0.65
print(f'baseline_elec: {baseline_elec:.2f} W')

savings = (baseline_elec - total_elec) / baseline_elec * 100
print(f'savings: {savings:.2f}%')

print()
print('=== CONVERGENCE TIMES ===')
# Find when omega reaches 95% of steady-state value
omega_all = np.abs(np.array(d['omega_z']))
omega_ss = omega
t95_idx = np.where(omega_all >= 0.95 * omega_ss)[0]
if len(t95_idx) > 0:
    t95 = t[t95_idx[0]]
    print(f'omega_95%_time: {t95:.2f} s')

# Find torque balance time (when residual < 5%)
motor_torque_all = T * np.sin(tilt_rad) * 0.4 + 0.015 * T * np.cos(tilt_rad)
wing_torque_all = np.array(d['wing_torque'])
residual = np.abs(motor_torque_all - wing_torque_all) / wing_torque
balance_idx = np.where(residual < 0.05)[0]
if len(balance_idx) > 0:
    t_balance = t[balance_idx[0]]
    print(f'torque_balance_time: {t_balance:.2f} s')
