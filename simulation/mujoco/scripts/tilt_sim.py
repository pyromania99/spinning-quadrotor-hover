"""
Quadrotor simulation with Backstepping Controller for attitude/altitude control.

This implements the control law from the paper:
    u_{d,1} = B_2^{-1}(-K_2 * x̃_2 - A_2 * x_2 - D_2 - (A_1^T - K̇) * x̃_1 + G_3 * x_1)

State definitions:
    x_1 = [n_{3,x}, n_{3,y}]^T  (3rd column of R_eb, x and y components)
    x_2 = [v_z, p, q]^T         (vertical velocity, roll rate, pitch rate)
"""

import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("mujoco\\quad.xml")
data = mujoco.MjData(model)

# Get quad body ID
quad_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "quad")

# Physical parameters
kf = 2.5e-5    # thrust constant (N/(rad/s)^2)
km = 0.0e-7    # torque constant (Nm/(rad/s)^2)
mass = 1.2     # kg
g = 9.81       # m/s^2

# Inertia from quad.xml: diaginertia="2.0 2.0 4.0"
Jx = 2.0
Jy = 2.0
Jz = 4.0

# Arm length (distance from center to rotor)
L = 0.2  # meters


class BacksteppingController:
    """Backstepping controller for drone altitude and attitude control."""
    
    def __init__(self, mass, Jx, Jy, Jz, g=9.81):
        """Initialize controller with drone parameters."""
        self.m = mass
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.g = g
        
        # Controller gains
        self.k_n3 = 10.0      # Gain for K matrix (x_1 tracking)
        self.k_vz = 1.0     # Gain for v_z in K_2
        self.k_omega = 0.00   # Gain for p, q in K_2
        
        # Position control gains - reduced for stability with backstepping
        self.kp_pos = 0.0001  # Further reduced for stability
        self.kd_pos = 0.0  # Moderate damping
        self.kp_z = 1.0
        self.kd_z = 0.0
        # Build constant matrices
        self.K_2 = np.diag([self.k_vz, self.k_omega, self.k_omega])
        # D_2 represents disturbance terms in dynamics: ẋ_2 = A_2*x_2 + B_2*u + D_2
        # For v̇_z: v̇_z = (F_z*n3z)/m - g, so D_2[0] = -g
        self.D_2 = np.array([-self.g, 0.0, 0.0])
    
    def _build_B2_inv(self, n3z):
        """Build B_2^{-1} matrix with singularity protection.
        
        Dynamics: v̇_z = (F_z * n3z) / m - g
        So: B_2[0,0] * F_z = (n3z/m) * F_z
        Therefore: B_2[0,0] = n3z/m  =>  B_2^{-1}[0,0] = m/n3z
        
        But the control computes the DESIRED acceleration, not the force directly.
        The relationship is: F_z = m * (desired_acc_z + g) / n3z
        """
        eps = 1e-6
        n3z_safe = n3z if abs(n3z) > eps else eps * np.sign(n3z + eps)
        return np.array([
            [self.m / n3z_safe, 0.0, 0.0],
            [0.0, self.Jx, 0.0],
            [0.0, 0.0, self.Jy]
        ])
    
    def _build_A1(self, C, D, A, B):
        """Build A_1 matrix."""
        return np.array([[0.0, -C, A], [0.0, -D, B]])
    
    def _build_A2(self, r):
        """Build A_2 matrix with gyroscopic coupling."""
        return np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -(self.Jz - self.Jy) / self.Jx * r],
            [0.0, -(self.Jx - self.Jz) / self.Jy * r, 0.0]
        ])
    
    def _build_K(self, A, B, C, D):
        """Build K matrix."""
        return self.k_n3 * np.array([[0.0, 0.0], [-B, A], [-D, C]])
    
    def _build_G3(self, n3z):
        """Build G_3 matrix."""
        G = -n3z * self.k_n3 * np.eye(2)
        G_3 = np.zeros((3, 3))
        G_3[1:3, 1:3] = G
        return G_3
    
    def _compute_K_dot(self, A, B, C, D, p, q, r, n3x, n3y):
        """Compute K̇ using rotation kinematics."""
        A_dot = C * r - q * n3x
        B_dot = D * r - q * n3y
        C_dot = -A * r + p * n3x
        D_dot = -B * r + p * n3y
        return self.k_n3 * np.array([[0.0, 0.0], [-B_dot, A_dot], [-D_dot, C_dot]])
    
    def compute_control(self, R_eb, v_z, omega, x1_d, v_z_d):
        """Compute the backstepping control input."""
        A, B, C, D = R_eb[0, 0], R_eb[1, 0], R_eb[0, 1], R_eb[1, 1]
        n3x, n3y, n3z = R_eb[0, 2], R_eb[1, 2], R_eb[2, 2]
        p, q, r = omega
        
        x_1 = np.array([n3x, n3y])
        x_2 = np.array([v_z, p, q])
        x_1_tilde = x_1 - x1_d
        
        # Compute x2_d according to paper: x2_d = K*x_tilde_1 + [vz_d, 0, 0]^T
        K = self._build_K(A, B, C, D)
        x2_d = K @ x_1_tilde + np.array([v_z_d, 0.0, 0.0])
        
        x_2_tilde = x_2 - x2_d
        
        A1 = self._build_A1(C, D, A, B)
        A2 = self._build_A2(r)
        B2_inv = self._build_B2_inv(n3z)
        K_dot = self._compute_K_dot(A, B, C, D, p, q, r, n3x, n3y)
        G3 = self._build_G3(n3z)
        
        term1 = -self.K_2 @ x_2_tilde
        term2 = -A2 @ x_2
        term3 = -self.D_2
        term4 = -(A1.T - K_dot) @ x_1_tilde
        term5 = G3 @ x_2
        
        return B2_inv @ (term1 + term2 + term3 + term4 + term5) 
    
    def compute_desired_states(self, target_pos, current_pos, current_vel):
        """Compute desired states for position tracking.
        
        Correctly computes n3_d as a unit vector using:
            a_total = a_desired + g*e_z
            n3_d = a_total / ||a_total||
            F_d = m * ||a_total||
        
        This ensures:
        1. x'' = (F/m) * n3x (not g * n3x)
        2. ||n3_d|| = 1 (unit vector constraint)
        """
        pos_error = target_pos - current_pos
        
        # Desired accelerations in world frame from position PD control
        ax_d = self.kp_pos * pos_error[0] - self.kd_pos * current_vel[0]
        ay_d = self.kp_pos * pos_error[1] - self.kd_pos * current_vel[1]
        az_d = self.kp_z * pos_error[2] - self.kd_z * current_vel[2]
        
        # Total acceleration vector including gravity compensation
        # The thrust must produce: a_d + g*e_z (to cancel gravity and achieve desired acc)
        a_total = np.array([ax_d, ay_d, az_d + self.g])
        
        # Limit maximum tilt angle (e.g., 60 degrees -> n3z_min ≈ 0.5)
        max_tilt = 0.5  # corresponds to ~60° max tilt
        n3z_min = np.cos(np.arcsin(max_tilt))  # ≈ 0.866 for 30° max, 0.5 for 60°
        
        # Compute thrust magnitude and direction
        a_norm = np.linalg.norm(a_total)
        
        if a_norm < 1e-6:
            # Edge case: zero acceleration request -> hover
            n3_d = np.array([0.0, 0.0, 1.0])
            v_z_des = az_d
        else:
            # Desired thrust direction (unit vector)
            n3_d = a_total / a_norm
            
            # Enforce minimum n3z to limit tilt angle
            if n3_d[2] < n3z_min:
                # Scale horizontal components to respect tilt limit
                n3_horiz = np.sqrt(n3_d[0]**2 + n3_d[1]**2)
                if n3_horiz > 1e-6:
                    # Maximum allowed horizontal component for given n3z_min
                    n3_horiz_max = np.sqrt(1 - n3z_min**2)
                    scale = n3_horiz_max / n3_horiz
                    n3_d = np.array([n3_d[0] * scale, n3_d[1] * scale, n3z_min])
                    # Re-normalize to ensure unit vector
                    n3_d = n3_d / np.linalg.norm(n3_d)
                else:
                    n3_d = np.array([0.0, 0.0, 1.0])
            
            v_z_des = az_d
        
        return np.array([n3_d[0], n3_d[1]]), v_z_des


# Initialize backstepping controller
controller = BacksteppingController(mass, Jx, Jy, Jz, g)


def quat_to_rotation_matrix(quat):
    """Convert quaternion [w, x, y, z] to rotation matrix (body to world)."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])


def quat_to_euler(quat):
    """Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw] (ZYX convention).
    
    Uses atan2 for all angles to ensure correct quadrant handling for any yaw value.
    Handles gimbal lock (pitch = ±90°) gracefully.
    
    Returns:
        roll, pitch, yaw in radians
    """
    w, x, y, z = quat
    
    # Compute rotation matrix elements needed for Euler extraction
    # R = [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]
    r11 = 1 - 2*(y*y + z*z)
    r12 = 2*(x*y - w*z)
    r13 = 2*(x*z + w*y)
    r21 = 2*(x*y + w*z)
    r22 = 1 - 2*(x*x + z*z)
    r23 = 2*(y*z - w*x)
    r31 = 2*(x*z - w*y)
    r32 = 2*(y*z + w*x)
    r33 = 1 - 2*(x*x + y*y)
    
    # ZYX Euler angles (yaw-pitch-roll, aerospace convention)
    # pitch = atan2(-r31, sqrt(r11^2 + r21^2)) for numerical stability
    sy = np.sqrt(r11*r11 + r21*r21)
    
    if sy > 1e-6:  # Not at gimbal lock
        roll = np.arctan2(r32, r33)
        pitch = np.arctan2(-r31, sy)
        yaw = np.arctan2(r21, r11)
    else:  # Gimbal lock: pitch ≈ ±90°
        roll = np.arctan2(-r23, r22)
        pitch = np.arctan2(-r31, sy)
        yaw = 0.0  # Yaw is undefined at gimbal lock, set to 0
    
    return roll, pitch, yaw


def control_allocation(F_z, tau_x, tau_y):
    """Convert desired thrust and torques to individual motor commands.
    
    X-config motor layout (motors at 45° from body axes):
        Motor 1: (+L, +L) front-right
        Motor 2: (-L, +L) front-left
        Motor 3: (-L, -L) rear-left
        Motor 4: (+L, -L) rear-right
    
    For X-config, each motor contributes to BOTH roll and pitch:
        tau_x (roll):  +T1 -T2 -T3 +T4 (right side up)
        tau_y (pitch): +T1 +T2 -T3 -T4 (nose up)
    
    Solving: T = F/4 + (tau_x ± tau_y) / (4*L)
    """
    f_base = F_z / 4.0
    
    # For X-config, combine roll and pitch contributions
    # tau_x contribution per motor: ±tau_x / (4*L)
    # tau_y contribution per motor: ±tau_y / (4*L)
    tau_x_contrib = tau_x / (4 * L)
    tau_y_contrib = tau_y / (4 * L)
    
    # Clip combined contributions to prevent negative thrust
    max_torque_contrib = f_base
    scale = 1.0
    max_combined = max(abs(tau_x_contrib + tau_y_contrib), 
                       abs(tau_x_contrib - tau_y_contrib),
                       abs(-tau_x_contrib + tau_y_contrib),
                       abs(-tau_x_contrib - tau_y_contrib))
    if max_combined > max_torque_contrib:
        scale = max_torque_contrib / max_combined
    
    tau_x_contrib *= scale
    tau_y_contrib *= scale
    
    # X-config allocation
    T1 = f_base + tau_x_contrib + tau_y_contrib  # front-right: +roll, +pitch
    T2 = f_base - tau_x_contrib + tau_y_contrib  # front-left:  -roll, +pitch
    T3 = f_base - tau_x_contrib - tau_y_contrib  # rear-left:   -roll, -pitch
    T4 = f_base + tau_x_contrib - tau_y_contrib  # rear-right:  +roll, -pitch
    
    thrusts = np.maximum(np.array([T1, T2, T3, T4]), 0)
    return np.clip(thrusts / kf, 0, 5e6)


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

        # FIXED: Correct mj_applyFT signature - arrays should not be reshaped
        mujoco.mj_applyFT(
            model,
            data,
            force_world,      # 1D array of 3 elements
            torque_world,     # 1D array of 3 elements
            point,            # 1D array of 3 elements
            quad_body_id,
            data.qfrc_applied,
        )

def hover_controller(target_pos=np.array([0.0, 0.0, 1.5])):
    """Backstepping controller for position and attitude control.
    
    Args:
        target_pos: Target position [x, y, z] in world frame
        
    Returns:
        u: Motor commands (omega^2)
        diagnostics: Dictionary of diagnostic values
    """
    # Get state
    pos = data.qpos[:3]
    vel = data.qvel[:3]
    quat = data.qpos[3:7]
    ang_vel = data.qvel[3:6]  # [p, q, r] body angular velocities
    
    # Get rotation matrix DIRECTLY from MuJoCo (body to world)
    R_eb = data.xmat[quad_body_id].reshape(3, 3)
    
    # Vertical velocity in world frame
    v_z = vel[2]
    
    # Body angular velocities
    omega = ang_vel  # [p, q, r]
    
    # Compute desired states based on position tracking
    x1_d, v_z_d = controller.compute_desired_states(target_pos, pos, vel)
    
    # Compute backstepping control
    u_backstepping = controller.compute_control(R_eb, v_z, omega, x1_d, v_z_d)
    
    # Extract thrust and torques
    F_z = u_backstepping[0] 
    tau_x = u_backstepping[1]
    tau_y = u_backstepping[2]
    
    # Allocate to motors
    u = control_allocation(F_z, tau_x, tau_y)
    
    # Diagnostics
    pos_error = target_pos - pos
    diagnostics = {
        'F_z': F_z,
        'tau_x': tau_x,
        'tau_y': tau_y,
        'desired_n3x': x1_d[0],
        'desired_n3y': x1_d[1],
        'actual_n3x': R_eb[0, 2],
        'actual_n3y': R_eb[1, 2],
        'desired_vz': v_z_d,
        'desired_roll': np.arcsin(np.clip(x1_d[0], -1, 1)),
        'desired_pitch': np.arcsin(np.clip(x1_d[1], -1, 1)),
        'acc_world_x_desired': controller.kp_pos * pos_error[0] - controller.kd_pos * vel[0],
        'acc_world_y_desired': controller.kp_pos * pos_error[1] - controller.kd_pos * vel[1],
        'error_rate_world_x': -controller.kd_pos * vel[0],
        'error_rate_world_y': -controller.kd_pos * vel[1],
        'dt': model.opt.timestep,
        'pos_error_world_x': pos_error[0],
        'pos_error_world_y': pos_error[1]
    }
    
    return u, diagnostics


def main():
    print("Starting quadrotor simulation with Backstepping Controller...")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Scroll: Zoom")
    print("  - Double-click: Select/track")
    print(f"\nController: Backstepping (k_n3={controller.k_n3}, k_vz={controller.k_vz}, k_omega={controller.k_omega})")
    print(f"Position gains: kp={controller.kp_pos}, kd={controller.kd_pos}\n")
    
    # Data logging
    log_data = {
        'time': [],
        'pos_x': [], 'pos_y': [], 'pos_z': [],
        'vel_x': [], 'vel_y': [], 'vel_z': [],
        'pos_error_x': [], 'pos_error_y': [], 'pos_error_z': [],
        'roll': [], 'pitch': [], 'yaw': [],
        'desired_roll': [], 'desired_pitch': [],
        'roll_error': [], 'pitch_error': [],
        'acc_x_desired': [], 'acc_y_desired': [],
        'motor_1': [], 'motor_2': [], 'motor_3': [], 'motor_4': [],
        'ang_vel_x': [], 'ang_vel_y': [], 'ang_vel_z': [],
        'target_x': [], 'target_y': [], 'target_z': [],
        'F_z': [], 'tau_x': [], 'tau_y': [],
        'n3x': [], 'n3y': [], 'n3z': [],
        'desired_n3x': [], 'desired_n3y': []
    }
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set initial camera
        viewer.cam.lookat[:] = [0, 0, 1.5]
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20
        
        step_count = 0
        
        # Square trajectory state
        square_size = 1.0
        target_z = 1.5
        reach_threshold = 0.05
        hold_time_required = 2.0
        
        # Define the four corners of the square
        corners = [
            np.array([-square_size, -square_size, target_z]),
            np.array([square_size, -square_size, target_z]),
            np.array([square_size, square_size, target_z]),
            np.array([-square_size, square_size, target_z])
        ]
        current_corner = 0
        target_pos = corners[current_corner]
        hold_start_time = None  # Track when we started holding position
        
        while viewer.is_running():
            # Check if we've reached the current corner
            pos = data.qpos[:3]
            distance_to_target = np.linalg.norm(target_pos - pos)
            
            if distance_to_target < reach_threshold:
                # Within threshold - start or continue holding
                if hold_start_time is None:
                    hold_start_time = data.time
                    print(f"\n>>> Within threshold! Holding position for {hold_time_required}s...")
                
                # Check if we've held long enough
                hold_duration = data.time - hold_start_time
                if hold_duration >= hold_time_required:
                    # Move to next corner
                    current_corner = (current_corner + 1) % 4
                    target_pos = corners[current_corner]
                    hold_start_time = None
                    print(f"\n>>> Held for {hold_time_required}s! Moving to corner {current_corner}: [{target_pos[0]:.2f}, {target_pos[1]:.2f}]")
            else:
                # Outside threshold - reset hold timer
                if hold_start_time is not None:
                    print(f"\n>>> Lost position (distance: {distance_to_target:.4f}m), resetting hold timer...")
                hold_start_time = None
            
            # Clear forces
            data.qfrc_applied[:] = 0.0
            
            # Run position controller
            u, diagnostics = hover_controller(target_pos=target_pos)
            apply_forces(u)
            
            # Get state for logging
            quat = data.qpos[3:7]
            
            # Calculate current roll, pitch, and yaw using robust extraction
            roll, pitch, yaw = quat_to_euler(quat)
            
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
            log_data['roll'].append(np.rad2deg(roll))
            log_data['pitch'].append(np.rad2deg(pitch))
            log_data['yaw'].append(np.rad2deg(yaw))
            log_data['desired_roll'].append(np.rad2deg(diagnostics['desired_roll']))
            log_data['desired_pitch'].append(np.rad2deg(diagnostics['desired_pitch']))
            log_data['roll_error'].append(np.rad2deg(diagnostics['desired_roll'] - roll))
            log_data['pitch_error'].append(np.rad2deg(diagnostics['desired_pitch'] - pitch))
            log_data['acc_x_desired'].append(diagnostics['acc_world_x_desired'])
            log_data['acc_y_desired'].append(diagnostics['acc_world_y_desired'])
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
            
            # Backstepping-specific logging
            R_eb = quat_to_rotation_matrix(quat)
            log_data['F_z'].append(diagnostics['F_z'])
            log_data['tau_x'].append(diagnostics['tau_x'])
            log_data['tau_y'].append(diagnostics['tau_y'])
            log_data['n3x'].append(R_eb[0, 2])
            log_data['n3y'].append(R_eb[1, 2])
            log_data['n3z'].append(R_eb[2, 2])
            log_data['desired_n3x'].append(diagnostics['desired_n3x'])
            log_data['desired_n3y'].append(diagnostics['desired_n3y'])
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Sync viewer at ~60 Hz
            if step_count % 16 == 0:  # 1000Hz sim / 16 = ~62.5 Hz render
                viewer.sync()
            
            step_count += 1
            
            # Print status every 1000 steps
            if step_count % 1000 == 0:
                pos = data.qpos[:3]
                vel = data.qvel[:3]
                quat = data.qpos[3:7]
                
                # Calculate current roll, pitch, and yaw using robust extraction
                roll, pitch, yaw = quat_to_euler(quat)
                
                # Calculate errors
                pos_error = np.linalg.norm(target_pos - pos)
                x_error = target_pos[0] - pos[0]
                y_error = target_pos[1] - pos[1]
                z_error = target_pos[2] - pos[2]
                
                print(f"\n--- t={data.time:.2f}s ---")
                print(f"Position: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
                print(f"Target:   x={target_pos[0]:.3f}, y={target_pos[1]:.3f}, z={target_pos[2]:.3f}")
                print(f"Error:    x={x_error:.3f}, y={y_error:.3f}, z={z_error:.3f}")
                print(f"Attitude: roll={np.rad2deg(roll):.2f}°, pitch={np.rad2deg(pitch):.2f}°")
                print(f"Control:  F_z={diagnostics['F_z']:.2f}N, τx={diagnostics['tau_x']:.3f}Nm, τy={diagnostics['tau_y']:.3f}Nm")
                print(f"n3:       actual=[{diagnostics['actual_n3x']:.3f}, {diagnostics['actual_n3y']:.3f}], desired=[{diagnostics['desired_n3x']:.3f}, {diagnostics['desired_n3y']:.3f}]")
                print(f"Total position error: {pos_error:.3f}m")
    
    # Save logged data
    import json
    with open('mujoco/sim_log.json', 'w') as f:
        json.dump(log_data, f)
    print("\n\nSimulation data saved to mujoco/sim_log.json")


if __name__ == "__main__":
    main()