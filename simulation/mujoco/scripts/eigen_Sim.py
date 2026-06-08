import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("mujoco\\quad.xml")
data = mujoco.MjData(model)

# Get quad body ID
quad_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "quad")

# Physical parameters
kf = 2.5e-5    # thrust constant (N/(rad/s)^2) - increased for realistic quad
km = 6.0e-7    # torque constant (Nm/(rad/s)^2)
mass = 1.2     # kg
g = 9.81       # m/s^2

# Controller gains
kp_z = 15.0
kd_z = 8.0
kp_pos = 0.10  # position control gain (x, y)
kd_pos = 1.0   # error rate damping gain (x, y)

# Attitude control damping and inertia parameters
alpha = 0.5    # roll/pitch damping term
beta = 0.6     # yaw damping term
Ixx = 0.01     # roll inertia (kg*m^2)
Iyy = 0.01     # pitch inertia (kg*m^2)
Izz = 0.02     # yaw inertia (kg*m^2)

# Error tracking for derivative term
prev_error_body_x = 0.0
prev_error_body_y = 0.0
prev_roll_error = 0.0
prev_pitch_error = 0.0
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
    """PD controller for position and attitude control.
    
    Args:
        target_pos: Target position [x, y, z] in world frame
    
    Body frame convention (from XML):
        +x: RIGHT of drone
        +y: FORWARD of drone
        +z: UP
    
    Angular velocity mapping:
        ang_vel[0] = omega_x = pitch rate (rotation about x/right axis)
        ang_vel[1] = omega_y = roll rate (rotation about y/forward axis)
        ang_vel[2] = omega_z = yaw rate
    """
    global prev_error_body_x, prev_error_body_y, prev_roll_error, prev_pitch_error, prev_time
    
    # Get state
    pos = data.qpos[:3]
    vel = data.qvel[:3]
    quat = data.qpos[3:7]
    ang_vel = data.qvel[3:6]
    
    # MuJoCo quaternion order is (w, x, y, z)
    qw, qx, qy, qz = quat
    
    # Convert quaternion to Euler angles
    # BODY FRAME CONVENTION:
    #   roll = rotation about body Y-axis (forward axis) - tilts left/right
    #   pitch = rotation about body X-axis (right axis) - tilts nose up/down
    # This is SWAPPED from standard 3-2-1 aerospace convention!
    
    # yaw (z-axis rotation) - same for both conventions
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    # For our body frame:
    # roll (about Y-axis) - what 3-2-1 calls "pitch"
    sinp = np.sqrt(1.0 + 2.0 * (qw * qy - qx * qz))
    cosp = np.sqrt(1.0 - 2.0 * (qw * qy - qx * qz))
    roll = 2.0 * np.arctan2(sinp, cosp) - np.pi / 2.0
    
    # pitch (about X-axis) - what 3-2-1 calls "roll"  
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    pitch = np.arctan2(sinr_cosp, cosr_cosp)
    

    # Position error in world frame
    pos_error_world = target_pos - pos
    vel_world = vel[:3]
    
    # Calculate error rate in WORLD frame (inertial frame)
    dt = data.time - prev_time if prev_time > 0 else 0.001
    error_rate_world_x = (pos_error_world[0] - prev_error_body_x) / dt
    error_rate_world_y = (pos_error_world[1] - prev_error_body_y) / dt
    
    # Update previous values (storing world frame errors now)
    prev_error_body_x = pos_error_world[0]
    prev_error_body_y = pos_error_world[1]

    acc_world_x_desired = kp_pos * pos_error_world[0] + kd_pos * error_rate_world_x
    acc_world_y_desired = kp_pos * pos_error_world[1] + kd_pos * error_rate_world_y 
    
    desired_roll = ( np.cos(yaw)*acc_world_x_desired
                + np.sin(yaw)*acc_world_y_desired ) / g

    desired_pitch   = ( np.sin(yaw)*acc_world_x_desired
                - np.cos(yaw)*acc_world_y_desired ) / g


    z_error = pos_error_world[2]
    z_vel = vel_world[2]

    # Compensate base thrust for tilt (roll, pitch)
    base_thrust = (mass * g) / (4.0 * np.cos(roll) * np.cos(pitch))
    altitude_correction = (kp_z * z_error - kd_z * z_vel) / 4.0

    # Attitude control: track desired angles (OUTER LOOP - KEEP AS IS)
    roll_error = desired_roll - roll 
    pitch_error = desired_pitch - pitch

    # Get current angular rates
    p = ang_vel[0]  # pitch rate (rotation about body x-axis)
    q = ang_vel[1]  # roll rate (rotation about body y-axis)
    r = ang_vel[2]  # yaw rate (rotation about body z-axis)

    # INNER LOOP: Rate-based control law
    # Outer loop produces desired rates from attitude errors
    Kp_att = 0.70  # attitude proportional gain
    Kd_att = 10.0  # rate damping gain
    
    # Calculate error rates by discrete differentiation
    pitch_error_rate = (pitch_error - prev_pitch_error) / dt
    roll_error_rate = (roll_error - prev_roll_error) / dt
    
    desired_pitch_rate = Kp_att * pitch_error + Kd_att * pitch_error_rate
    desired_roll_rate = Kp_att * roll_error + Kd_att * roll_error_rate

    prev_pitch_error = pitch_error
    prev_roll_error = roll_error
    prev_time = data.time

    # Define Hurwitz matrix M_dyn (desired closed-loop dynamics)
    M_dyn = np.array([
        [-7.0,  0.0,  0.0],
        [ 0.0, -7.0,  0.0],
        [ 0.0,  0.0, -beta]  # beta for yaw
    ])

    # Feedforward yaw rate from co-rotating motors
    net_thrust = base_thrust * 4.0
    desired_yaw_rate = net_thrust * km / (kf * 0.3)  # beta = 0.3

    # Desired state vector [desired_pitch_rate, desired_roll_rate, desired_yaw_rate]
    M_des = -M_dyn @ np.array([
        [desired_pitch_rate],
        [desired_roll_rate],
        [desired_yaw_rate]
    ])

    # Concatenate: M = [M_dyn | M_des] → 3x4 matrix
    M = np.hstack([M_dyn, M_des])

    # System coupling matrix with gyroscopic effects
    A_r = np.array([
        [-alpha,  r,    0.0, 0.0],  # alpha = -0.5
        [-r,   -alpha,  0.0, 0.0],
        [ 0.0,  0.0, -beta, 0.0]   # beta = 0.6
    ])

    M_eff = M - A_r

    # Inertia matrix
    Ixx, Iyy, Izz = 0.01, 0.01, 0.02
    B_inv = np.diag([1.0/Ixx, 1.0/Iyy, 1.0/Izz])
    B = np.linalg.inv(B_inv)

    # Augmented state: [p, q, r, 1]
    w_aug = np.array([
        [p],    # pitch rate
        [q],    # roll rate
        [r],    # yaw rate
        [1.0]   # feedforward
    ])

    # Control law: τ = B * (M - A) * w_aug
    C = B @ M_eff
    corrections = C @ w_aug

    pitch_correction = float(corrections[0, 0])  # Torque about x-axis
    roll_correction = float(corrections[1, 0])   # Torque about y-axis
    # corrections[2] is yaw (feedforward only, ignored since motors co-rotate)

    # Motor mixing (KEEP YOUR CURRENT SIGNS - THEY WORK!)
    u1 = base_thrust + altitude_correction - roll_correction + pitch_correction  # FR
    u2 = base_thrust + altitude_correction + roll_correction + pitch_correction  # FL
    u3 = base_thrust + altitude_correction + roll_correction - pitch_correction  # RL
    u4 = base_thrust + altitude_correction - roll_correction - pitch_correction  # RR
    # Convert thrusts to squared angular velocities
    # T = kf * omega^2  =>  omega^2 = T / kf
    u = np.array([u1, u2, u3, u4]) / kf
    u = np.clip(u, 0, 5e5)  
    
    # Return control inputs and diagnostics
    diagnostics = {
        'acc_world_x_desired': acc_world_x_desired,
        'acc_world_y_desired': acc_world_y_desired,
        'desired_roll': desired_roll,
        'desired_pitch': desired_pitch,
        'error_rate_world_x': error_rate_world_x,
        'error_rate_world_y': error_rate_world_y,
        'dt': dt,
        'pos_error_world_x': pos_error_world[0],
        'pos_error_world_y': pos_error_world[1],
        'ang_vel_roll': ang_vel[1],
        'ang_vel_pitch': ang_vel[0],
        'ang_vel_yaw': ang_vel[2],
        'desired_roll_rate': desired_roll_rate,
        'desired_pitch_rate': desired_pitch_rate,
        'roll_correction': roll_correction,
        'pitch_correction': pitch_correction
    }
    return u, diagnostics

def main():
    print("Starting quadrotor simulation...")
    print("Controls:")
    print("  - Mouse: Rotate view")
    print("  - Scroll: Zoom")
    print("  - Double-click: Select/track")
    print("\nThe drone will follow a sine wave altitude command.\n")
    
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
        'target_x': [], 'target_y': [], 'target_z': []
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
        reach_threshold = 0.01  # meters - how close to get before counting as reached
        hold_time_required = 2.0  # seconds - how long to hold position before switching
        
        # Define the four corners of the square
        corners = [
            np.array([-square_size, -square_size, target_z]),  # Back-left
            np.array([square_size, -square_size, target_z]),   # Back-right
            np.array([square_size, square_size, target_z]),    # Front-right
            np.array([-square_size, square_size, target_z])    # Front-left
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
            qw, qx, qy, qz = quat
            
            # Calculate current roll, pitch, and yaw
            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            
            sinp = np.sqrt(1.0 + 2.0 * (qw * qy - qx * qz))
            cosp = np.sqrt(1.0 - 2.0 * (qw * qy - qx * qz))
            roll = 2.0 * np.arctan2(sinp, cosp) - np.pi / 2.0
            
            sinr_cosp = 2.0 * (qw * qx + qy * qz)
            cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
            pitch = np.arctan2(sinr_cosp, cosr_cosp)
            
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
                qw, qx, qy, qz = quat
                
                # Calculate current roll, pitch, and yaw
                pitch = np.arcsin(2.0 * (qw*qx - qy*qz))
                roll = np.arctan2(2.0 * (qw*qy + qx*qz), 1.0 - 2.0 * (qx*qx + qy*qy))
                yaw = np.arctan2(2.0 * (qw*qz + qx*qy), 1.0 - 2.0 * (qy*qy + qz*qz))
                
                # Calculate errors
                pos_error = np.linalg.norm(target_pos - pos)
                x_error = target_pos[0] - pos[0]
                y_error = target_pos[1] - pos[1]
                z_error = target_pos[2] - pos[2]
                
                print(f"\n--- t={data.time:.2f}s ---")
                print(f"Position: x={pos[0]:.3f}, y={pos[1]:.3f}, z={pos[2]:.3f}")
                print(f"Target:   x={target_pos[0]:.3f}, y={target_pos[1]:.3f}, z={target_pos[2]:.3f}")
                print(f"Error:    x={x_error:.3f}, y={y_error:.3f}, z={z_error:.3f}")
                print(f"Attitude: roll={np.rad2deg(roll):.2f}°, pitch={np.rad2deg(pitch):.2f}°, yaw={np.rad2deg(yaw):.2f}°")
                print(f"Desired:  roll={np.rad2deg(diagnostics['desired_roll']):.2f}°, pitch={np.rad2deg(diagnostics['desired_pitch']):.2f}°")
                print(f"Angular rates: roll={np.rad2deg(diagnostics['ang_vel_roll']):.2f}°/s, pitch={np.rad2deg(diagnostics['ang_vel_pitch']):.2f}°/s, yaw={np.rad2deg(diagnostics['ang_vel_yaw']):.2f}°/s")
                print(f"Desired rates: roll={np.rad2deg(diagnostics['desired_roll_rate']):.2f}°/s, pitch={np.rad2deg(diagnostics['desired_pitch_rate']):.2f}°/s")
                print(f"Corrections:   roll={diagnostics['roll_correction']:.4f}, pitch={diagnostics['pitch_correction']:.4f}")
                print(f"Accel:    x={diagnostics['acc_world_x_desired']:.3f}m/s², y={diagnostics['acc_world_y_desired']:.3f}m/s²")
                print(f"Total position error: {pos_error:.3f}m")
    
    # Save logged data
    import json
    with open('mujoco/sim_log.json', 'w') as f:
        json.dump(log_data, f)
    print("\n\nSimulation data saved to mujoco/sim_log.json")


if __name__ == "__main__":
    main()