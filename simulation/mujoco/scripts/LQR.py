import mujoco
import mujoco.viewer
import numpy as np
from scipy.linalg import solve_continuous_are
# FAILURE_MODE = "none"  # Options: "none", "one", "two", "three"

# ==========================================================
# Model
# ==========================================================
model = mujoco.MjModel.from_xml_path("mujoco\\quad.xml")
data = mujoco.MjData(model)

quad_body_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "quad"
)

# ==========================================================
# Physical parameters (match your model)
# ==========================================================
mass = 1.2
g = 9.81

kf = 2.5e-5
km = 1.0e-6

# Aerodynamic yaw damping coefficient (Nm/(rad/s))
# Models drag torque from spinning propellers
C_yaw_drag = 0.01

I_T_xx = 0.01
I_T_yy = 0.01
I_T_zz = 0.02
I_P_zz = 1.5e-5

l = 0.2
gamma = 2.75e-3
tau_mot = 1e-6  # Effectively zero lag for perfect actuators in simulation

# ==========================================================
# Control / scheduling parameters
# ==========================================================
K_UPDATE_DT = 0.02   # 50 Hz LQR update
MAX_R_SCHED = 24.0  # rad/s clamp

prev_time = 0.0
last_K_time = -np.inf
K_current = None

# ==========================================================
# Equilibrium (used only for linearization constants)
# ==========================================================
def calculate_equilibrium():
    # nominal thrust split
    f = mass * g / 4
    omega = np.sqrt(f / kf)
    return {
        "f": np.array([f, f, f, f]),
        "omega": np.array([omega, omega, omega, omega]),
        "body_rates": np.zeros(3),
        "n": np.array([0.0, 0.0, 1.0]),
    }

equilibrium = calculate_equilibrium()

# ==========================================================
# Linearization utilities (from Mueller–D’Andrea)
# ==========================================================
def compute_a_bar(r):
    return (
        ((I_T_xx - I_T_zz) / I_T_xx) * r
    )

def build_A_matrix(r, eq):
    nx, ny, nz = eq["n"]
    a_bar = compute_a_bar(r)

    A = np.array([
        [0,      a_bar,   0,      0,      0,          0],
        [-a_bar, 0,       0,      0,      0,          0],
        [-nz,    0,       0,      r,      0,          0],
        [0,      nz,     -r,      0,      0,          0],
        [0,      0,       0,      0,     -1/tau_mot,  0],
        [0,      0,       0,      0,      0,         -1/tau_mot],
    ])
    return A

def build_B_matrix():
    return np.array([
        [0,          l/I_T_xx],
        [l/I_T_xx,   0],
        [0,          0],
        [0,          0],
        [1/tau_mot,  0],
        [0,          1/tau_mot],
    ])

B = build_B_matrix()

# Reduced Q gains to keep commanded forces physically realizable
# Previous [1, 1, 80, 80] caused commands of ±200 N (impossible!)
Q = np.diag([0.1, 0.1, 10.0, 10.0, 0.0, 0.0])
R = 0.01 * np.eye(2)

# ==========================================================
# Gain-Scheduled LQR Setup
# ==========================================================

# Yaw rate schedule points (rad/s)
R_SCHEDULE = np.array([0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0, 22.5, 25.0])

# Pre-compute K gains for each scheduled yaw rate
print("Pre-computing gain schedule...")
K_SCHEDULE = []
# Equilibrium at hover: level orientation, zero rates
equilibrium = {
    "p": 0, "q": 0, "r": 0,
    "n": np.array([0, 0, 1]),
    "mot": np.array([0, 0])
}
for r_sched in R_SCHEDULE:
    A_sched = build_A_matrix(r_sched, equilibrium)
    P_sched = solve_continuous_are(A_sched, B, Q, R)
    K_sched = np.linalg.inv(R) @ B.T @ P_sched
    K_SCHEDULE.append(K_sched)
    print(f"  r={r_sched:5.1f} rad/s: K computed")

print("Gain schedule ready!\n")

def interpolate_K(r_current):
    """Linearly interpolate K gains based on current yaw rate"""
    # Clip to schedule range
    r_clamped = np.clip(abs(r_current), R_SCHEDULE[0], R_SCHEDULE[-1])
    
    # Find bracketing indices
    idx_upper = np.searchsorted(R_SCHEDULE, r_clamped)
    if idx_upper == 0:
        return K_SCHEDULE[0]
    if idx_upper >= len(R_SCHEDULE):
        return K_SCHEDULE[-1]
    
    idx_lower = idx_upper - 1
    
    # Linear interpolation weight
    r_lower = R_SCHEDULE[idx_lower]
    r_upper = R_SCHEDULE[idx_upper]
    alpha = (r_clamped - r_lower) / (r_upper - r_lower)
    
    # Interpolate K
    K_interp = (1 - alpha) * K_SCHEDULE[idx_lower] + alpha * K_SCHEDULE[idx_upper]
    return K_interp

# ==========================================================
# Geometry helpers
# ==========================================================
def get_primary_axis_body():
    Rmat = np.zeros(9)
    mujoco.mju_quat2Mat(Rmat, data.qpos[3:7])
    Rmat = Rmat.reshape(3, 3)
    return Rmat.T @ np.array([0, 0, 1])

def tilt_axis_from_n(n):
    tilt_angle = np.arccos(np.clip(n[2], -1.0, 1.0))
    axis = np.array([n[1], -n[0], 0.0])
    if np.linalg.norm(axis) > 1e-6:
        axis /= np.linalg.norm(axis)
    return axis, tilt_angle

# ==========================================================
# Force application
# ==========================================================
def apply_forces(u_sq):
    thrusts = kf * u_sq
    torques = km * u_sq
    Rb = data.xmat[quad_body_id].reshape(3, 3)

    # Apply motor forces and torques
    for i in range(4):
        f_world = Rb @ np.array([0, 0, thrusts[i]])
        tau_world = Rb @ np.array([0, 0, torques[i]])
        mujoco.mj_applyFT(
            model, data,
            f_world, tau_world,
            data.site_xpos[i],
            quad_body_id,
            data.qfrc_applied
        )
    
    # Add aerodynamic yaw damping (drag from spinning)
    r = data.qvel[5]  # Yaw rate in world frame
    yaw_damping_torque = -C_yaw_drag * r * abs(r)  # Quadratic drag
    tau_damping_world = np.array([0, 0, yaw_damping_torque])
    mujoco.mj_applyFT(
        model, data,
        np.array([0, 0, 0]), tau_damping_world,
        data.xpos[quad_body_id],
        quad_body_id,
        data.qfrc_applied
    )

# ==========================================================
# Controller (Option 1: gain-scheduled LQR)
# ==========================================================
def controller(target_pos):
    global prev_time, last_K_time, K_current

    pos = data.qpos[:3]
    vel = data.qvel[:3]
    ang = data.qvel[3:6]

    # Body rates in paper convention
    p = ang[1]
    q = ang[0]
    r = ang[2]

    # Get scheduled LQR gain based on current yaw rate
    K_current = interpolate_K(r)

    # === Translational PD (reduced to prevent saturation) ===
    kp_pos, kd_pos = 0.5,1.0
    kp_z, kd_z = 15.0, 8.0

    pos_err = target_pos - pos
    acc_x = kp_pos * pos_err[0] - kd_pos * vel[0]
    acc_y = kp_pos * pos_err[1] - kd_pos * vel[1]
    acc_z = kp_z   * pos_err[2] - kd_z   * vel[2]

    f_total = mass * (g + acc_z)

    # Desired primary axis in WORLD frame (direction of thrust vector)
    n_des_world_x = -acc_x/g 
    n_des_world_y = -acc_y/g 
    
    # Limit desired tilt to ±15° (≈0.26 rad) to prevent motor saturation
    max_tilt = 0.26  # tan(15°) ≈ 0.268
    n_des_world_x = np.clip(n_des_world_x, -max_tilt, max_tilt)
    n_des_world_y = np.clip(n_des_world_y, -max_tilt, max_tilt)
    
    n_des_world_z = np.sqrt(max(0, 1 - n_des_world_x**2 - n_des_world_y**2))
    n_des_world = np.array([n_des_world_x, n_des_world_y, n_des_world_z])
    n_des_world = n_des_world / np.linalg.norm(n_des_world)
    
    # Get current yaw angle to rotate n_des to body frame
    quat = data.qpos[3:7]
    qw, qx, qy, qz = quat
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    # Rotate n_des from world to body frame (rotation about Z-axis)
    # Body frame rotates by -yaw relative to world frame
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    n_des_body_x = cos_yaw * n_des_world[0] - sin_yaw * n_des_world[1]
    n_des_body_y = sin_yaw * n_des_world[0] + cos_yaw * n_des_world[1]

    # Current primary axis (already in body frame)
    n = get_primary_axis_body()
    nx, ny, _ = n
    
    # State for LQR (error state in body frame)
    x = np.array([
        p,
        q,
        nx - n_des_body_x,
        ny - n_des_body_y,
        0,
        0
    ])

    # LQR control gives torque commands
    u = -K_current @ x
    tau_pitch = -u[0]  # u[0] = pitch torque
    tau_roll = u[1]   # u[1] = roll torque
    
    # Debug output every 100 steps with full causal chain
    if int(data.time * 1000) % 100 == 0:
        # Get angular velocity in body frame
        omega_body = data.qvel[3:6]  # [q, p, r] in MuJoCo convention
        omega_body_reordered = np.array([omega_body[1], omega_body[0], omega_body[2]])  # [p, q, r]
        omega_mag = np.linalg.norm(omega_body_reordered)
        
        # Spin axis (normalized angular velocity vector in body frame)
        if omega_mag > 0.01:
            spin_axis = omega_body_reordered / omega_mag
            # Angle between spin axis and n3 (body Z / thrust axis)
            n3_body = np.array([0, 0, 1])
            cos_angle = np.dot(spin_axis, n3_body)
            spin_alignment_deg = np.rad2deg(np.arccos(np.clip(cos_angle, -1, 1)))
        else:
            spin_alignment_deg = 0.0
            
        print(f"[DEBUG t={data.time:.2f}]")
        print(f"  pos_err=[{pos_err[0]:6.3f}, {pos_err[1]:6.3f}] | vel=[{vel[0]:6.3f}, {vel[1]:6.3f}]")
        print(f"  acc_des=[{acc_x:6.3f}, {acc_y:6.3f}] → n_des_world=[{n_des_world[0]:6.3f}, {n_des_world[1]:6.3f}]")
        print(f"  yaw={np.rad2deg(yaw):5.1f}° → n_des_body=[{n_des_body_x:6.3f}, {n_des_body_y:6.3f}]")
        print(f"  n_cur=[{nx:6.3f}, {ny:6.3f}] | τ=[pitch:{tau_pitch:6.2f}, roll:{tau_roll:6.2f}]")
        print(f"  ω_body=[{omega_body_reordered[0]:5.2f}, {omega_body_reordered[1]:5.2f}, {omega_body_reordered[2]:5.2f}] |ω|={omega_mag:5.2f} rad/s")
        print(f"  Spin axis ∠ n3: {spin_alignment_deg:5.1f}° (0°=perfect alignment)")

    # === MOTOR MIXING & SATURATION PREVENTION ===
    MAX_TAU = f_total * l / 4
    
    tau_pitch = np.clip(tau_pitch, -MAX_TAU, MAX_TAU)
    tau_roll = np.clip(tau_roll, -MAX_TAU, MAX_TAU)
    f1 = f_total/4 - tau_roll/(2*l) - tau_pitch/(2*l)
    f2 = f_total/4 + tau_roll/(2*l) - tau_pitch/(2*l)
    f3 = f_total/4 + tau_roll/(2*l) + tau_pitch/(2*l)
    f4 = f_total/4 - tau_roll/(2*l) + tau_pitch/(2*l)
   
    # CRITICAL: Tight clamp prevents saturation-induced cross-coupling
    # With [0.5, 6.0], saturation at [0.5, 0.5, 6.0, 6.0] creates 11 N × 0.2 m ≈ 2.2 Nm pitch torque!
    forces_raw = np.array([f1, f2, f3, f4])
    # forces = np.clip(forces_raw, 0, forces_raw)
    
    # torque_error_pitch = ((forces[2] + forces[3]) - (forces[0] + forces[1])) * l
    # torque_error_roll = ((forces[1] + forces[2]) - (forces[0] + forces[3])) * l
    # print(f"[SATURATION t={data.time:.2f}]")
    # print(f"  Commanded: {forces_raw}")
    # print(f"  Clamped:   {forces}")
    # print(f"  Unintended τ: pitch={torque_error_pitch:.2f} Nm, roll={torque_error_roll:.2f} Nm")

    return np.clip(forces_raw / kf, 0, 5e6)

# ==========================================================
# Main loop
# ==========================================================
def main():
    global prev_time
    target = np.array([1.0, 0.0, 1.5])
    data.qpos[2] = 1.5
    prev_time = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0, 0, 1.5]
        viewer.cam.distance = 5.0

        step = 0
        while viewer.is_running():
            data.qfrc_applied[:] = 0.0
            u = controller(target)
            apply_forces(u)
            mujoco.mj_step(model, data)

            if step % 16 == 0:
                viewer.sync()

            if step % 500 == 0:
                pos = data.qpos[:3]
                vel = data.qvel[:3]
                pos_err = target - pos
                n = get_primary_axis_body()
                axis, angle = tilt_axis_from_n(n)
                
                print(f"\n{'='*80}")
                print(f"t={data.time:.2f}s | Step {step}")
                print(f"{'-'*80}")
                print(f"Position:    x={pos[0]:7.3f}, y={pos[1]:7.3f}, z={pos[2]:7.3f}")
                print(f"Target:      x={target[0]:7.3f}, y={target[1]:7.3f}, z={target[2]:7.3f}")
                print(f"Error:       x={pos_err[0]:7.3f}, y={pos_err[1]:7.3f}, z={pos_err[2]:7.3f}")
                print(f"Velocity:    x={vel[0]:7.3f}, y={vel[1]:7.3f}, z={vel[2]:7.3f}")
                print(f"{'-'*80}")
                print(f"Body Rates:  p={data.qvel[4]:7.3f}, q={data.qvel[3]:7.3f}, r={data.qvel[5]:7.3f} rad/s")
                print(f"Tilt Vector: n=[{n[0]:7.3f}, {n[1]:7.3f}, {n[2]:7.3f}]")
                print(f"Tilt Angle:  {np.rad2deg(angle):5.1f}° about {axis}")
                print(f"{'-'*80}")
                # Show last computed motor commands (omega^2 values)
                motor_speeds = np.sqrt(u)  # Convert from omega^2 to omega
                motor_forces = kf * u  # Thrusts in Newtons
                print(f"Motor ω²:    [{u[0]:.0f}, {u[1]:.0f}, {u[2]:.0f}, {u[3]:.0f}]")
                print(f"Motor F(N):  [{motor_forces[0]:.3f}, {motor_forces[1]:.3f}, {motor_forces[2]:.3f}, {motor_forces[3]:.3f}]")
                print(f"Total Thrust: {np.sum(motor_forces):.3f} N (weight={mass*g:.3f} N)")


            step += 1

if __name__ == "__main__":
    main()
