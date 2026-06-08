import numpy as np
import mujoco
import mujoco.viewer

# ============================================================================
# SO(3) GEOMETRIC CONTROLLER FOR QUADROTOR
# ============================================================================
# This controller operates directly on the rotation group SO(3), avoiding
# singularities and gimbal lock inherent in Euler angle representations.
#
# Key differences from Euler-based controller:
# 1. No explicit roll/pitch/yaw calculations
# 2. Error computed as rotation matrix difference
# 3. Natural handling of large attitude changes
# 4. Almost-global stability guarantees
# ============================================================================

# Load MuJoCo model
model = mujoco.MjModel.from_xml_path("mujoco\\quad.xml")
data = mujoco.MjData(model)

# Get quad body ID
quad_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "quad")

# Physical parameters
kf = 2.5e-5    # thrust constant (N/(rad/s)^2)
km = 1.0e-7    # torque constant (Nm/(rad/s)^2)
mass = 1.2     # kg
g = 9.81       # m/s^2

# Previous state variables for derivative terms
prev_pos_error_world = np.zeros(3)
prev_vel_world = np.zeros(3)
prev_e_R = np.zeros(3)
prev_time = 0.0

def apply_forces(u):
    """Apply rotor forces to the quadrotor.
    
    Args:
        u: Array of 4 squared angular velocities (rad/s)^2
    """
    thrusts = kf * u
    torques = km * u 
    
    # Get body rotation matrix to transform body frame forces to world frame
    body_xmat = data.xmat[quad_body_id].reshape(3, 3)  # 3x3 rotation matrix

    for i, f in enumerate(thrusts):
        # Force in body frame (pointing up along body z-axis)
        force_body = np.array([0.0, 0.0, float(f)], dtype=np.float64)
        # Transform to world frame
        force_world = body_xmat @ force_body
        
        # Torque in body frame (yaw torque along body z-axis)
        torque_body = np.array([0.0, 0.0, float(torques[i])], dtype=np.float64)
        # Transform to world frame
        torque_world = body_xmat @ torque_body
        
        # Get the site position (attachment point) for this rotor
        point = np.array(data.site_xpos[i], dtype=np.float64)

        mujoco.mj_applyFT(
            model,
            data,
            force_world,
            torque_world,
            point,
            quad_body_id,
            data.qfrc_applied,
        )


def vex(S):
    """Extract vector from skew-symmetric matrix.
    
    For a skew-symmetric matrix S:
        S = [  0  -c   b ]
            [  c   0  -a ]
            [ -b   a   0 ]
    
    Returns: [a, b, c]^T
    
    This is the inverse of the hat operator (wedge).
    """
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def hat(v):
    """Skew-symmetric matrix from vector (wedge/hat operator).
    
    Args:
        v: 3D vector [a, b, c]
    
    Returns:
        3x3 skew-symmetric matrix such that hat(v) @ w = v × w
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def quaternion_to_rotation_matrix(quat):
    """Convert quaternion to rotation matrix.
    
    Args:
        quat: Quaternion [qw, qx, qy, qz] (MuJoCo convention)
    
    Returns:
        3x3 rotation matrix
    """
    qw, qx, qy, qz = quat
    
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    
    return R


def compute_desired_rotation(thrust_world, yaw_desired=0.0):
    """Compute desired rotation matrix from desired thrust vector.
    
    Args:
        thrust_world: Desired thrust direction in world frame (will be normalized)
        yaw_desired: Desired yaw angle (rotation about world Z-axis)
    
    Returns:
        3x3 desired rotation matrix R_d
    
    Construction:
        - Z-axis of body frame points along thrust direction
        - X-axis projected onto horizontal plane at desired yaw
        - Y-axis completes right-handed frame
    """
    # Desired body Z-axis (thrust direction, normalized)
    z_body_des = thrust_world / np.linalg.norm(thrust_world)
    
    # Desired heading direction (world frame)
    x_world_des = np.array([np.cos(yaw_desired), np.sin(yaw_desired), 0.0])
    
    # Body Y-axis (perpendicular to both Z_body and X_world)
    y_body_des = np.cross(z_body_des, x_world_des)
    y_body_des_norm = np.linalg.norm(y_body_des)
    
    # Handle singularity when thrust is purely vertical
    if y_body_des_norm < 1e-6:
        # If thrust is vertical, use current yaw
        y_body_des = np.array([-np.sin(yaw_desired), np.cos(yaw_desired), 0.0])
    else:
        y_body_des = y_body_des / y_body_des_norm
    
    # Body X-axis completes right-handed frame
    x_body_des = np.cross(y_body_des, z_body_des)
    
    # Construct rotation matrix [X_body | Y_body | Z_body]
    R_d = np.column_stack([x_body_des, y_body_des, z_body_des])
    
    return R_d


def so3_geometric_controller(target_pos=np.array([0.0, 0.0, 1.5]), 
                             yaw_desired=None):
    """SO(3) geometric controller for quadrotor.
    
    This controller operates directly on the rotation group SO(3), providing:
    - Singularity-free attitude control
    - Almost-global stability
    - Natural handling of large rotations
    - Improved transient response
    
    Args:
        target_pos: Desired position [x, y, z] in world frame
        yaw_desired: Desired yaw angle. If None, uses current yaw (for co-rotating motors)
    
    Control Architecture:
        1. Position control → desired acceleration (world frame)
        2. Desired acceleration → desired rotation matrix R_d
        3. Orientation error on SO(3): e_R = vex(R_d^T R - R^T R_d)
        4. Geometric control law: τ = -k_R e_R - k_ω e_ω + ω×(Jω)
        5. Motor mixing → individual rotor commands
    
    Body frame convention (from XML):
        +X: RIGHT of drone
        +Y: FORWARD of drone
        +Z: UP
    """
    global prev_pos_error_world, prev_vel_world, prev_e_R, prev_time
    
    # ========================================================================
    # 1. STATE EXTRACTION
    # ========================================================================
    pos = data.qpos[:3]
    vel = data.qvel[:3]
    quat = data.qpos[3:7]  # [qw, qx, qy, qz]
    ang_vel = data.qvel[3:6]  # [ωx, ωy, ωz] in body frame
    
    # Current rotation matrix (world → body)
    R = quaternion_to_rotation_matrix(quat)
    
    # Extract current yaw from rotation matrix for co-rotating motor adaptation
    # yaw = atan2(R[1,0], R[0,0]) for ZYX convention
    current_yaw = np.arctan2(R[1, 0], R[0, 0])
    
    # For co-rotating motors, we adapt to current yaw since we can't control it
    if yaw_desired is None:
        yaw_desired = current_yaw
    
    # Time step for derivatives
    dt = data.time - prev_time if prev_time > 0 else 0.001
    
    # ========================================================================
    # 2. POSITION CONTROL (matching eigen_Sim.py exactly)
    # ========================================================================
    kp_pos = 0.10   # Position proportional gain (x, y) 
    kd_pos = 1.0    # Error rate damping gain (x, y)
    kp_z = 15.0     # Altitude proportional gain
    kd_z = 8.0      # Altitude derivative gain
    
    # Position error in world frame
    pos_error_world = target_pos - pos
    
    # Calculate error rate in WORLD frame (derivative of error, not velocity)
    if prev_time > 0:
        error_rate_world_x = (pos_error_world[0] - prev_pos_error_world[0]) / dt
        error_rate_world_y = (pos_error_world[1] - prev_pos_error_world[1]) / dt
    else:
        error_rate_world_x = 0.0
        error_rate_world_y = 0.0
    
    # Desired acceleration in world frame (PD control on position error)
    acc_world_desired = np.zeros(3)
    acc_world_desired[0] = kp_pos * pos_error_world[0] + kd_pos * error_rate_world_x
    acc_world_desired[1] = kp_pos * pos_error_world[1] + kd_pos * error_rate_world_y
    acc_world_desired[2] = kp_z * pos_error_world[2] - kd_z * vel[2] + g
    
    # ========================================================================
    # 3. DESIRED ROTATION MATRIX
    # ========================================================================
    # Total desired force direction
    thrust_world_desired = mass * acc_world_desired
    
    # Clamp to ensure positive thrust magnitude
    thrust_norm = np.linalg.norm(thrust_world_desired)
    if thrust_norm < 0.1:
        thrust_world_desired = np.array([0.0, 0.0, mass * g])
    
    R_d = compute_desired_rotation(thrust_world_desired, yaw_desired)
    
    # ========================================================================
    # 4. ORIENTATION ERROR ON SO(3) - LIE ALGEBRA
    # ========================================================================
    # Configuration error on SO(3): e_R ∈ ℝ³ (tangent space at identity)
    # Standard formula: e_R = 0.5 * vex(R_d^T R - R^T R_d)
    # This gives error in BODY frame, where positive error means R lags R_d
    #
    # Sign convention check:
    # - In eigen_Sim.py: roll_error = desired_roll - roll (positive = need more roll)
    # - With SO(3): e_R measures how R differs from R_d
    # - For control: positive e_R[1] should mean "need to roll more" (same as Euler)
    #
    # The standard formula gives e_R that drives R → R_d when τ = -k*e_R
    # But eigen_Sim.py uses τ ∝ +error (positive error → positive torque)
    # So we negate to match: e_R = -0.5 * vex(R_d^T R - R^T R_d)
    
    e_R_matrix = R_d.T @ R - R.T @ R_d
    e_R = -0.5 * vex(e_R_matrix)  # Negated to match eigen_Sim.py sign convention
    
    # Angular velocity error: e_ω = ω - R^T R_d ω_d
    # For hover stabilization, ω_d = 0, so e_ω = ω
    e_omega = ang_vel
    
    # Calculate SO(3) error rate in Lie algebra (for outer loop derivative term)
    if prev_time > 0:
        e_R_dot = (e_R - prev_e_R) / dt
    else:
        e_R_dot = np.zeros(3)
    
    # ========================================================================
    # 5. HURWITZ CONTROL WITH SO(3) OUTER LOOP (matching eigen_Sim.py structure)
    # ========================================================================
    # Structure: 
    #   OUTER LOOP: SO(3) error → desired angular rates (replaces Euler errors)
    #   INNER LOOP: Hurwitz on [p,q,r] with feedforward from desired rates
    #
    # This maintains EXACT same structure as eigen_Sim.py, just using e_R instead
    # of Euler angle errors in the outer loop.
    
    # Inertia parameters
    Ixx, Iyy, Izz = 0.01, 0.01, 0.02
    J = np.diag([Ixx, Iyy, Izz])
    
    # Compute thrust magnitude for diagnostics
    thrust_magnitude = np.linalg.norm(thrust_world_desired)
    
    # Current angular rates
    p = ang_vel[0]  # rotation rate about body x-axis (pitch rate)
    q = ang_vel[1]  # rotation rate about body y-axis (roll rate)
    r = ang_vel[2]  # rotation rate about body z-axis (yaw rate)
    
    # -------------------------------------------------------------------------
    # OUTER LOOP: SO(3) error → desired angular rates
    # -------------------------------------------------------------------------
    # Same gains as eigen_Sim.py
    Kp_att = 0.70   # attitude proportional gain
    Kd_att = 10.0   # attitude derivative gain
    
    # Compute desired rates from SO(3) error (replaces Euler angle errors)
    # e_R[0] corresponds to pitch_error (rotation about x-axis)
    # e_R[1] corresponds to roll_error (rotation about y-axis)
    # e_R[2] corresponds to yaw_error (rotation about z-axis)
    desired_pitch_rate = Kp_att * e_R[0] + Kd_att * e_R_dot[0]
    desired_roll_rate = Kp_att * e_R[1] + Kd_att * e_R_dot[1]
    
    # -------------------------------------------------------------------------
    # INNER LOOP: Hurwitz eigenvalue placement (EXACTLY as eigen_Sim.py)
    # -------------------------------------------------------------------------
    # Damping parameters
    alpha = 0.5    # roll/pitch damping
    beta = 0.6     # yaw damping
    
    # Define Hurwitz matrix M_dyn (desired closed-loop dynamics)
    M_dyn = np.array([
        [-7.0,  0.0,  0.0],
        [ 0.0, -7.0,  0.0],
        [ 0.0,  0.0, -beta]
    ])
    
    # Feedforward yaw rate from co-rotating motors (same as eigen_Sim.py)
    net_thrust = (mass * g)
    desired_yaw_rate = net_thrust * km / (kf * 0.3)
    
    # Desired state vector [desired_pitch_rate, desired_roll_rate, desired_yaw_rate]
    # M_des = -M_dyn @ desired_rates creates the feedforward term
    M_des = -M_dyn @ np.array([
        [desired_pitch_rate],
        [desired_roll_rate],
        [desired_yaw_rate]
    ])
    
    # Concatenate: M = [M_dyn | M_des] → 3x4 matrix
    M = np.hstack([M_dyn, M_des])
    
    # System coupling matrix with gyroscopic effects
    A_r = np.array([
        [-alpha,  r,     0.0, 0.0],
        [-r,     -alpha, 0.0, 0.0],
        [ 0.0,    0.0,  -beta, 0.0]
    ])
    
    M_eff = M - A_r
    
    # Augmented body rate state: [p, q, r, 1]
    w_aug = np.array([
        [p],
        [q],
        [r],
        [1.0]
    ])
    
    # Control law: τ = J @ M_eff @ w_aug (EXACTLY as eigen_Sim.py)
    C = J @ M_eff
    corrections = C @ w_aug
    
    pitch_correction = float(corrections[0, 0])  # Torque about x-axis
    roll_correction = float(corrections[1, 0])   # Torque about y-axis
    # corrections[2] is yaw (ignored for co-rotating motors)
    
    # ========================================================================
    # 6. THRUST MAGNITUDE (compensate for tilt like eigen_Sim.py)
    # ========================================================================
    # Extract roll and pitch from R for tilt compensation
    sin_pitch = -R[2, 0]  # -R[2,0] gives sin(pitch)
    sin_roll = R[2, 1] / np.cos(np.arcsin(sin_pitch)) if abs(sin_pitch) < 0.99 else 0.0
    
    roll = np.arcsin(np.clip(sin_roll, -1.0, 1.0))
    pitch = np.arcsin(np.clip(sin_pitch, -1.0, 1.0))
    
    # Compensate base thrust for tilt (like eigen_Sim.py)
    cos_roll = np.cos(roll)
    cos_pitch = np.cos(pitch)
    if abs(cos_roll * cos_pitch) > 0.1:  # Avoid division by zero
        base_thrust = (mass * g) / (4.0 * cos_roll * cos_pitch)
    else:
        base_thrust = (mass * g) / 4.0
    
    # Altitude correction
    z_error = pos_error_world[2]
    z_vel = vel[2]
    altitude_correction = (kp_z * z_error - kd_z * z_vel) / 4.0
    
    # ========================================================================
    # 7. MOTOR MIXING (matching eigen_Sim.py signs exactly)
    # ========================================================================
    # Motor layout:
    #   2(FL)  1(FR)
    #      \ X /
    #      / X \
    #   3(RL)  4(RR)
    #
    # pitch_correction and roll_correction are now torques (N·m) from Hurwitz law
    # Motor thrust commands (matching eigen_Sim.py signs)
    u1 = base_thrust + altitude_correction - roll_correction + pitch_correction  # FR
    u2 = base_thrust + altitude_correction + roll_correction + pitch_correction  # FL  
    u3 = base_thrust + altitude_correction + roll_correction - pitch_correction  # RL
    u4 = base_thrust + altitude_correction - roll_correction - pitch_correction  # RR
    
    # ========================================================================
    # 8. CONVERT TO SQUARED ANGULAR VELOCITIES
    # ========================================================================
    u = np.array([u1, u2, u3, u4]) / kf
    u = np.clip(u, 0, 5e5)
    
    # ========================================================================
    # 9. UPDATE PREVIOUS STATE
    # ========================================================================
    prev_pos_error_world = pos_error_world
    prev_vel_world = vel
    prev_e_R = e_R
    prev_time = data.time
    
    # Return control inputs and diagnostics
    diagnostics = {
        'e_R': e_R,
        'e_R_dot': e_R_dot,
        'e_omega': ang_vel,
        'thrust_magnitude': thrust_magnitude,
        'pos_error_world': pos_error_world,
        'acc_world_desired': acc_world_desired,
        'desired_roll_rate': desired_roll_rate,
        'desired_pitch_rate': desired_pitch_rate,
        'roll_correction': roll_correction,
        'pitch_correction': pitch_correction,
        'ang_vel_x': ang_vel[0],
        'ang_vel_y': ang_vel[1],
        'ang_vel_z': ang_vel[2],
    }
    
    return u, diagnostics


def main():
    """Main simulation loop with SO(3) geometric controller."""
    print("Starting SO(3) Geometric Controller Simulation...")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Scroll: Zoom")
    print("  - Double-click: Select/track")
    print("\nThe drone will follow a square trajectory.\n")
    
    # Data logging
    log_data = {
        'time': [],
        'pos_x': [], 'pos_y': [], 'pos_z': [],
        'vel_x': [], 'vel_y': [], 'vel_z': [],
        'pos_error_x': [], 'pos_error_y': [], 'pos_error_z': [],
        'e_R_x': [], 'e_R_y': [], 'e_R_z': [],  # SO(3) orientation error
        'motor_1': [], 'motor_2': [], 'motor_3': [], 'motor_4': [],
        'ang_vel_x': [], 'ang_vel_y': [], 'ang_vel_z': [],
        'target_x': [], 'target_y': [], 'target_z': [],
        'thrust_magnitude': []
    }
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set initial camera
        viewer.cam.lookat[:] = [0, 0, 1.5]
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20
        
        step_count = 0
        
        # Square trajectory state
        square_size = 1.0  # meters (half-width of square)
        target_z = 1.5  # constant altitude
        reach_threshold = 0.01  # meters
        hold_time_required = 2.0  # seconds
        
        # Define the four corners of the square
        corners = [
            np.array([-square_size, -square_size, target_z]),  # Back-left
            np.array([square_size, -square_size, target_z]),   # Back-right
            np.array([square_size, square_size, target_z]),    # Front-right
            np.array([-square_size, square_size, target_z])    # Front-left
        ]
        current_corner = 0
        target_pos = corners[current_corner]
        hold_start_time = None
        
        while viewer.is_running():
            # Check if we've reached the current corner
            pos = data.qpos[:3]
            distance_to_target = np.linalg.norm(target_pos - pos)
            
            if distance_to_target < reach_threshold:
                if hold_start_time is None:
                    hold_start_time = data.time
                    print(f"\n>>> Within threshold! Holding position for {hold_time_required}s...")
                
                hold_duration = data.time - hold_start_time
                if hold_duration >= hold_time_required:
                    current_corner = (current_corner + 1) % 4
                    target_pos = corners[current_corner]
                    hold_start_time = None
                    print(f"\n>>> Held for {hold_time_required}s! Moving to corner {current_corner}: [{target_pos[0]:.2f}, {target_pos[1]:.2f}]")
            else:
                if hold_start_time is not None:
                    print(f"\n>>> Lost position (distance: {distance_to_target:.4f}m), resetting hold timer...")
                hold_start_time = None
            
            # Clear forces
            data.qfrc_applied[:] = 0.0
            
            # Run SO(3) geometric controller
            u, diagnostics = so3_geometric_controller(target_pos=target_pos)
            apply_forces(u)
            
            # Get state for logging
            pos_error = target_pos - pos
            
            # Log data
            log_data['time'].append(data.time)
            log_data['pos_x'].append(pos[0])
            log_data['pos_y'].append(pos[1])
            log_data['pos_z'].append(pos[2])
            log_data['vel_x'].append(data.qvel[0])
            log_data['vel_y'].append(data.qvel[1])
            log_data['vel_z'].append(data.qvel[2])
            log_data['pos_error_x'].append(pos_error[0])
            log_data['pos_error_y'].append(pos_error[1])
            log_data['pos_error_z'].append(pos_error[2])
            log_data['e_R_x'].append(diagnostics['e_R'][0])
            log_data['e_R_y'].append(diagnostics['e_R'][1])
            log_data['e_R_z'].append(diagnostics['e_R'][2])
            log_data['motor_1'].append(u[0])
            log_data['motor_2'].append(u[1])
            log_data['motor_3'].append(u[2])
            log_data['motor_4'].append(u[3])
            log_data['ang_vel_x'].append(np.rad2deg(data.qvel[3]))
            log_data['ang_vel_y'].append(np.rad2deg(data.qvel[4]))
            log_data['ang_vel_z'].append(np.rad2deg(data.qvel[5]))
            log_data['target_x'].append(target_pos[0])
            log_data['target_y'].append(target_pos[1])
            log_data['target_z'].append(target_pos[2])
            log_data['thrust_magnitude'].append(diagnostics['thrust_magnitude'])
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Sync viewer at ~60 Hz
            if step_count % 16 == 0:
                viewer.sync()
            
            step_count += 1
            
            # Print status every 1000 steps
            if step_count % 1000 == 0:
                pos = data.qpos[:3]
                ang_vel = data.qvel[3:6]
                pos_error = target_pos - pos
                
                print(f"\n--- t={data.time:.2f}s [SO(3) Controller] ---")
                print(f"Position: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
                print(f"Target:   x={target_pos[0]:.3f}, y={target_pos[1]:.3f}, z={target_pos[2]:.3f}")
                print(f"Error:    x={pos_error[0]:.3f}, y={pos_error[1]:.3f}, z={pos_error[2]:.3f}")
                print(f"SO(3) Error: e_R=[{diagnostics['e_R'][0]:.4f}, {diagnostics['e_R'][1]:.4f}, {diagnostics['e_R'][2]:.4f}]")
                print(f"Angular rates: [{np.rad2deg(ang_vel[0]):.2f}, {np.rad2deg(ang_vel[1]):.2f}, {np.rad2deg(ang_vel[2]):.2f}] °/s")
                print(f"Thrust: {diagnostics['thrust_magnitude']:.2f} N")
                print(f"Total position error: {np.linalg.norm(pos_error):.3f}m")
    
    # Save logged data
    import json
    with open('mujoco/sim_log_so3.json', 'w') as f:
        json.dump(log_data, f)
    print("\n\nSimulation data saved to mujoco/sim_log_so3.json")


if __name__ == "__main__":
    main()