clear;
clc;
close all;

global m g alpha time_arr psi_counter delta_x_cg delta_y_cg delta_z_cg
time_arr = [];
psi_counter = 1;

%% ============ HEXACOPTER PARAMETERS ============
m = 2;              % kg
g = 9.81;           % m/s^2
Ixx = 7.5e-3;
Iyy = 7.5e-3;
Izz = 2.3e-2;

Ct = 0.0027;
rho = 1.225;
Kd = 1.58e-4; % 
w_init = 2612.5;

l = 0.05;           % arm length (m) - reduced from 0.25m
d = 4.1;            % propeller diameter
pitch = 4.1;
kT_est = 1.2e-5;    % thrust coefficient (estimated)
kM_est = 2.047e-8;  % yaw torque coefficient

%% ============ CG UNCERTAINTY PARAMETERS (ADJUSTABLE) ============
% Set these to simulate CG offset during flight
delta_x_cg = -0.01;   % CG offset in x-direction (m) - adjustable ±0.03
delta_y_cg = 0.0;   % CG offset in y-direction (m) - adjustable ±0.03
delta_z_cg = 0.00;   % CG offset in z-direction (m) - adjustable ±0.02

% CG uncertainty bounds for robust allocation
delta_x_max = 0.03;  % Maximum expected CG offset in x (m)
delta_y_max = 0.03;  % Maximum expected CG offset in y (m)
delta_cg_bar = max(delta_x_max, delta_y_max);  % Max horizontal offset

% Enable/disable robust allocation
USE_ROBUST_ALLOCATION = true;  % Set to false for nominal allocation only

frequency = 100; %Controller Frequency in Hertz (increased for speed)
timePeriod = 50; % Simulation time
tspan = [0 1/frequency];
% stINIT now has 18 states (6 motors)
stINIT = [ 0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 , w_init, w_init, w_init, w_init, w_init, w_init];
%stIN = [x, y, z, phi, theta, psi, u, v, w, p, q, r, M1...M6];
options = odeset('RelTol',1e-4,'AbsTol',1e-6);

% Initialize arrays for storing states and time values
max_iterations = timePeriod * frequency;
all_states = zeros(max_iterations * 50, 18); % columns = 18 for hex
all_times = zeros(max_iterations * 50, 1);
states_counter = 1;

i = 1;

% Desired positions - simple trajectory
X_des = 1;  
Y_des = 1;  
Z_des = -1; 

% PID Coefficients for position control
Kp_pos_xy = 0.6;  Ki_pos_xy = 0; Kd_pos_xy = 1;  % Fixed position gains
Kp_Z = 500;       Ki_Z = 0;          Kd_Z = 100;      % Z position control gains

% PID Coefficients for angle control (scaled by 5x for l=0.05m)
Kp_ang = 0.005; Ki_ang = 0; Kd_ang = 0.05;

% Initialize error integrals and previous errors
X_err_integral = 0; X_err_prev = X_des;
Y_err_integral = 0; Y_err_prev = Y_des;
Z_err_integral = 0; Z_err_prev = -Z_des;

Phi_err_integral = 0; Phi_err_prev = 0;
Theta_err_integral = 0; Theta_err_prev = 0;

% Define maximum allowable angles (in radians)
phi_max = deg2rad(30); % Maximum roll angle limit
theta_max = deg2rad(30); % Maximum pitch angle limit

% Initialize trajectory storage for real-time plotting
traj_x = [];
traj_y = [];
traj_z = [];

% Pre-allocate arrays for better performance
T_des_arr = zeros(1, max_iterations);
Phi_des_arr = zeros(1, max_iterations);
Theta_des_arr = zeros(1, max_iterations);
Phi_actual_arr = zeros(1, max_iterations);
Theta_actual_arr = zeros(1, max_iterations);
X_error_arr = zeros(1, max_iterations);
Y_error_arr = zeros(1, max_iterations);
time_des_arr = zeros(1, max_iterations);

% Arrays for CG uncertainty analysis
eps_phi_arr = zeros(1, max_iterations);
eps_theta_arr = zeros(1, max_iterations);
alloc_error_arr = zeros(1, max_iterations);
stability_margin_arr = zeros(1, max_iterations);

global psiarr ttarr
psiarr = zeros(1, max_iterations * 50);
ttarr = zeros(1, max_iterations * 50);
motor_speed = zeros(1, max_iterations);

% Set alpha permanently to 0
alpha = 0;

% Motor angles (deg) for hex: 30, 90, 150, 210, 270, 330
% This configuration matches the matrix with ±1/2 and ±√3/2 coefficients
motor_angles_deg = [30, 90, 150, 210, 270, 330];
motor_angles_rad = deg2rad(motor_angles_deg);

% Rotor spin directions for yaw torque (1 = CCW, -1 = CW) alternating
rot_dir = [1, -1, 1, -1, 1, -1];

% Refine kT_est using actual motor model (optional, uses fallback if motor model fails)
% use V0=0 and aoa=0 for a rough linear estimate
approx_thrust_at_winit = MotorForce(w_init, d, pitch, 0, 0);
kT_est_refined = approx_thrust_at_winit / (w_init^2 + eps);

% Safety: use refined estimate if valid, otherwise keep initial estimate
if isfinite(kT_est_refined) && kT_est_refined > 0
    kT_est = kT_est_refined;
end

while i < (timePeriod * frequency)
    current_time = tspan(1);
    % alpha permanently 0: do not change it here
    
    if current_time > 200
        w_init = 3499.5;
        Kp_Z = 700;    Ki_Z = 1;   Kd_Z = 1500;
        Kp_pos_xy = 1; Kd_pos_xy = 100; 
        Kp_ang = 5; Kd_ang = 0.5;  % Scaled by 5x for l=0.05m
    end
    
    [t, states] = ode45(@hex_dynamics, tspan, stINIT, options);

    % Store states
    new_data_points = length(t);
    all_states(states_counter:states_counter+new_data_points-1, :) = states;
    all_times(states_counter:states_counter+new_data_points-1) = t;
    states_counter = states_counter + new_data_points;
    
    % Update initial conditions for the next iteration
    stINIT = states(end,:);
    
    % Current state
    x_current = states(end,1);
    y_current = states(end,2);
    z_current = states(end,3);
    psi_current = states(end,6);  % Current yaw angle
    
    dt = 1/frequency;
    
    % Position errors
    x_error = X_des - x_current;
    y_error = Y_des - y_current;
    z_error = z_current - Z_des;
    
    % Position error derivatives
    x_error_dot = (x_error - X_err_prev) / dt;
    y_error_dot = (y_error - Y_err_prev) / dt;
    z_error_dot = (z_error - Z_err_prev) / dt;
    
    % Position error integrals
    X_err_integral = X_err_integral + x_error * dt;
    Y_err_integral = Y_err_integral + y_error * dt;
    Z_err_integral = Z_err_integral + z_error * dt;
    
    % Position control outputs (desired accelerations in world frame)
    ax_des_world = -(Kp_pos_xy * x_error + Ki_pos_xy * X_err_integral + Kd_pos_xy * x_error_dot);
    ay_des_world = -(Kp_pos_xy * y_error + Ki_pos_xy * Y_err_integral + Kd_pos_xy * y_error_dot);
    
    % Transform desired accelerations from world frame to body frame considering yaw
    cos_psi = cos(psi_current);
    sin_psi = sin(psi_current);
    
    ax_des_body = cos_psi * ax_des_world + sin_psi * ay_des_world;
    ay_des_body = -sin_psi * ax_des_world + cos_psi * ay_des_world;
    
    % Convert desired body accelerations to desired roll and pitch angles
    phi_des = -ay_des_body / g;  % Desired roll angle
    theta_des = ax_des_body / g; % Desired pitch angle
    
    % phi_des = deg2rad(10);
    % theta_des = 0;
    % Limit desired angles to maximum allowable values
    phi_des = max(-phi_max, min(phi_max, phi_des));
    theta_des = max(-theta_max, min(theta_max, theta_des));

    % Current angles
    phi_current = states(end,4);
    theta_current = states(end,5);

    % Angle errors
    phi_error = phi_des - phi_current;
    theta_error = theta_des - theta_current;
    psi_error = psi_current;

    % Angle error derivatives
    phi_error_dot = (phi_error - Phi_err_prev) / dt;
    theta_error_dot = (theta_error - Theta_err_prev) / dt;
    
    % Angle error integrals
    Phi_err_integral = Phi_err_integral + phi_error * dt;
    Theta_err_integral = Theta_err_integral + theta_error * dt;
    
    % Altitude control (Z-axis) - use Z-only method (keeps your previous approach)
    z_ctrl_adj = Kp_Z * z_error + Ki_Z * Z_err_integral + Kd_Z * z_error_dot;
    
    upper_limit = 6000; % rad/s
    lower_limit = 0;
    
    % Base motor speed for altitude control (used to form an initial T_f estimate)
    w_base = max(lower_limit, min(w_init + z_ctrl_adj, upper_limit));
    
    % Attitude control outputs (we'll treat these as desired moments)
    phi_ctrl = Kp_ang * phi_error + Ki_ang * Phi_err_integral + Kd_ang * phi_error_dot;
    theta_ctrl = Kp_ang * theta_error + Ki_ang * Theta_err_integral + Kd_ang * theta_error_dot;
    yaw_ctrl = Kp_ang * psi_error;

    % Build desired moment vector M_des = [tau_phi; tau_theta; tau_psi; T_f]
    tau_phi = phi_ctrl;
    tau_theta = theta_ctrl;
    tau_psi = yaw_ctrl; 
    
    % Desired total thrust: start from equal-distribution estimate using w_base
    T_f_des = 6 * kT_est * (w_base^2);
    
    M_des = [tau_phi; tau_theta; tau_psi; T_f_des];
    
    % --- Build NOMINAL control allocation matrix G_c (no CG offset) ---
    G_c = zeros(4,6);
    G_c(1, :) = -l * sin(motor_angles_rad) * kT_est;  % Roll
    G_c(2, :) =  l * cos(motor_angles_rad) * kT_est;  % Pitch
    G_c(3, :) = kM_est * rot_dir;                      % Yaw
    G_c(4, :) = kT_est * ones(1, 6);                   % Thrust
    
    % --- Build ACTUAL system matrix G_a with CG offset ---
    G_a = G_c;
    for motor_idx = 1:6
        alpha_i = motor_angles_rad(motor_idx);
        
        % Moment arm uncertainties from CG offset (eq 3.21)
        % ∆l_i,φ ≈ -∆y_cg * cos(α_i) + ∆x_cg * sin(α_i)
        % ∆l_i,θ ≈ ∆x_cg * cos(α_i) + ∆y_cg * sin(α_i)
        delta_l_phi = -delta_y_cg * cos(alpha_i) + delta_x_cg * sin(alpha_i);
        delta_l_theta = delta_x_cg * cos(alpha_i) + delta_y_cg * sin(alpha_i);
        
        % Add uncertainty to roll/pitch rows
        G_a(1, motor_idx) = G_a(1, motor_idx) + delta_l_phi * kT_est;
        G_a(2, motor_idx) = G_a(2, motor_idx) + delta_l_theta * kT_est;
    end
    
    % --- Compute robustness margins (eq 3.24-3.25) ---
    G_c_pinv = pinv(G_c);
    
    % Calculate maximum achievable torques based on physical limits
    % Maximum motor speed (rad/s) - using w_init as reference max speed
    w_max = max(3500, w_init);  % Use current w_init or 3500 rad/s, whichever is higher
    
    % Maximum thrust per motor at w_max
    F_max = kT_est * w_max^2;
    
    % Maximum roll torque: occurs when motors on opposite sides run at different extremes
    % For hexacopter: max(|G_c(1,:)|) gives the largest moment arm coefficient
    tau_phi_max = max(abs(G_c(1,:))) * w_max^2;  % Max roll torque (N·m)
    tau_theta_max = max(abs(G_c(2,:))) * w_max^2;  % Max pitch torque (N·m)
    
    % Calculate robustness margins with safety bounds
    eps_phi = min(0.4, kT_est * delta_cg_bar * norm(G_c_pinv(1,:), 1) / tau_phi_max);
    eps_theta = min(0.4, kT_est * delta_cg_bar * norm(G_c_pinv(2,:), 1) / tau_theta_max);
    eps_T = 0.05;  % Thrust uncertainty margin (typically small)
    
    % --- Choose allocation strategy ---
    if USE_ROBUST_ALLOCATION
        motor_failed = (current_time > 20);
        
        if motor_failed
            % Increase robustness margins due to reduced control authority (eq 3.30)
            % After motor failure, remaining motors have higher workload and less redundancy
            G_c_reduced = G_c(:, [1 3 4 5 6]);  % 4x5 matrix without motor 2
            G_c_reduced_pinv = pinv(G_c_reduced);
            
            % Recalculate margins with reduced system
            eps_phi = min(0.5, kT_est * delta_cg_bar * norm(G_c_reduced_pinv(1,:), 1) / tau_phi_max);
            eps_theta = min(0.5, kT_est * delta_cg_bar * norm(G_c_reduced_pinv(2,:), 1) / tau_theta_max);
            eps_T = 0.10;  % Increased thrust uncertainty with reduced motors
        end
        
        % Weighted robust allocation (eq 3.30)
        % W_r,i = diag(w_φ(1-ε_φ,i), w_θ(1-ε_θ,i), 0, w_T(1-ε_T,i))
        w_phi = 1.0;    % Roll weight
        w_theta = 1.0;  % Pitch weight
        w_T = 1.0;      % Thrust weight
        
        W_r = diag([w_phi * (1 - eps_phi), w_theta * (1 - eps_theta), 0, w_T * (1 - eps_T)]);  
        G_alloc = W_r * G_c;
        M_des_weighted = M_des;
        
        % --- Adjust for motor failure (t > 200) ---
        if motor_failed
            % Motor 2 failed - remove its column and disable yaw control
            G_alloc = G_alloc(:, [1 3 4 5 6]);      % 4x5 matrix (motor 2 removed)
            nu_sq_reduced = pinv(G_alloc) * M_des_weighted;
            
            % Reconstruct full 6-motor vector
            nu_sq = zeros(6,1);
            nu_sq([1 3 4 5 6]) = nu_sq_reduced;   % failed motor = 0
        else
            % Normal operation - all 6 motors
            nu_sq = pinv(G_alloc) * M_des_weighted;
        end
    else
        % Standard nominal allocation - no motor failure handling, no robustness
        G_alloc = G_c;
        M_des_weighted = M_des;
        nu_sq = pinv(G_alloc) * M_des_weighted;
    end
    
    % --- Stability analysis ---
    E = eye(4) - G_a * G_c_pinv;
    lambda_max = max(abs(eig(E)));
    stability_margin = 1 - lambda_max;
    
    % Calculate allocation error (actual vs desired)
    T_actual = G_a * max(nu_sq, 0);
    T_error = T_actual - M_des;
    alloc_error_norm = norm(T_error); 
    % Clamp negative values to zero and apply motor bounds (squared)
    nu_sq = max(nu_sq, 0);
    nu_sq = min(nu_sq, upper_limit^2);
    
    % Compute motor speeds
    nu = sqrt(nu_sq);
    
    % Store motor speed for plotting (store base for comparison)
    if i <= max_iterations
        motor_speed(i) = w_base;
    end
    
    % Apply fade factor to second motor (you had that earlier); apply to matching indices
    fade_factor = (1 - 1/(1 + exp(-4*(current_time - 21))));  
    stINIT(13) = nu(1);
    stINIT(14) = nu(2) * fade_factor; 
    stINIT(15) = nu(3);
    stINIT(16) = nu(4);
    stINIT(17) = nu(5);
    stINIT(18) = nu(6);
    
    % Store previous errors for next iteration
    X_err_prev = x_error;
    Y_err_prev = y_error;
    Z_err_prev = z_error;
    Phi_err_prev = phi_error;
    Theta_err_prev = theta_error;
    
    % Store data for plotting
    if i <= max_iterations
        Phi_des_arr(i) = phi_des;
        Theta_des_arr(i) = theta_des;
        Phi_actual_arr(i) = phi_current;
        Theta_actual_arr(i) = theta_current;
        X_error_arr(i) = x_error;
        Y_error_arr(i) = y_error;
        time_des_arr(i) = current_time;
        
        % Store CG analysis data
        eps_phi_arr(i) = eps_phi;
        eps_theta_arr(i) = eps_theta;
        alloc_error_arr(i) = alloc_error_norm;
        stability_margin_arr(i) = stability_margin;
    end

    tspan = tspan + 1/frequency;
    i = i + 1;
end

% Trim pre-allocated arrays to actual size
all_states = all_states(1:states_counter-1, :);
all_times = all_times(1:states_counter-1);
psiarr = psiarr(1:psi_counter-1);
ttarr = ttarr(1:psi_counter-1);

% Trim plotting arrays
actual_iterations = min(i-1, max_iterations);
Phi_des_arr = Phi_des_arr(1:actual_iterations);
Theta_des_arr = Theta_des_arr(1:actual_iterations);
Phi_actual_arr = Phi_actual_arr(1:actual_iterations);
Theta_actual_arr = Theta_actual_arr(1:actual_iterations);
X_error_arr = X_error_arr(1:actual_iterations);
Y_error_arr = Y_error_arr(1:actual_iterations);
time_des_arr = time_des_arr(1:actual_iterations);
eps_phi_arr = eps_phi_arr(1:actual_iterations);
eps_theta_arr = eps_theta_arr(1:actual_iterations);
alloc_error_arr = alloc_error_arr(1:actual_iterations);
stability_margin_arr = stability_margin_arr(1:actual_iterations);

fprintf('Position control simulation complete. Using fixed gains: Kp_ang=%.3f, Ki_ang=%.3f, Kd_ang=%.3f\n', Kp_ang, Ki_ang, Kd_ang);
fprintf('\n=== CG OFFSET SIMULATION ===\n');
fprintf('CG Offset: x=%.1f mm, y=%.1f mm, z=%.1f mm\n', delta_x_cg*1000, delta_y_cg*1000, delta_z_cg*1000);
fprintf('Robust Allocation: %s\n', string(USE_ROBUST_ALLOCATION));
fprintf('Mean allocation error: %.6f N·m\n', mean(alloc_error_arr));
fprintf('Max allocation error: %.6f N·m\n', max(alloc_error_arr));
fprintf('Min stability margin: %.4f\n', min(stability_margin_arr));
%% PLOT RESULTS
% Create consistent time array for control data
if ~isempty(all_times)
    time_control = linspace(0, all_times(end), actual_iterations);
else
    time_control = linspace(0,1,actual_iterations);
end

figure(4);

subplot(4,1,1);
if ~isempty(all_states) && size(all_states, 1) > 0
    invZ = (-1.*all_states(:,3));
    plot(all_times,invZ, 'b-', 'LineWidth', 1.5); hold on;
    plot([0 all_times(end)], [-Z_des -Z_des], 'r--', 'LineWidth', 1.5); hold off;
    minScale = 1;
    ylim([min([invZ; -Z_des])-minScale, max([invZ; -Z_des])+minScale]);
    legend('Actual', 'Desired', 'Location', 'best');
else
    plot([0 1], [0 0], 'b-');
    title('Position: Z - No Data Available');
end
title('Position: Z');
ylabel('m');

subplot(4,1,2);
if ~isempty(all_states) && size(all_states, 1) > 0
    plot(all_times,rad2deg(all_states(:,6)))
    minScale = 1;
    ylim([min(rad2deg(all_states(:,6)))-minScale, max(rad2deg(all_states(:,6)))+minScale]);
else
    plot([0 1], [0 0], 'b-');
    title('Angle: Psi - No Data Available');
end
title('Angle: Psi');
ylabel('deg');

subplot(4,1,3);
if ~isempty(all_states) && size(all_states, 1) > 0
    invW = (-1.*all_states(:,9));
    plot(all_times,invW)
    minScale = 1;
    ylim([min(invW)-minScale, max(invW)+minScale]);
else
    plot([0 1], [0 0], 'b-');
    title('Velocity(z): W - No Data Available');
end
title('Velocity(z): W');
ylabel('m/s');

subplot(4,1,4);
if ~isempty(all_states) && size(all_states, 1) > 0
    plot(all_times,all_states(:,12))
    minScale = 1;
    ylim([min(all_states(:,12))-minScale, max(all_states(:,12))+minScale]);
else
    plot([0 1], [0 0], 'b-');
    title('Angular Rate: R - No Data Available');
end
title('Angular Rate: R');
ylabel('rad/s');

figure(2);

if ~isempty(all_states) && size(all_states, 1) > 0
    startColor = [0, 0, 1];     % Start color (blue)
    endColor = [1, 0, 0];       % End color (red)
    numPoints = size(all_states, 1);
    colorMap = [linspace(startColor(1), endColor(1), numPoints)', ...
                linspace(startColor(2), endColor(2), numPoints)', ...
                linspace(startColor(3), endColor(3), numPoints)'];

    scatter3(all_states(:,1),all_states(:,2),-1.*all_states(:,3),10,colorMap); hold on;
    scatter3(X_des, Y_des, -Z_des, 200, 'r*', 'LineWidth', 3); hold off;

    minScale = 1;
    xlim([min(all_states(:,1))-minScale, max(all_states(:,1))+minScale]);
    ylim([min(all_states(:,2))-minScale, max(all_states(:,2))+minScale]);
    zlim([min(-1.*all_states(:,3))-minScale, max(-1.*all_states(:,3))+minScale]);
    grid on;
    title('3D Trajectory (Blue to Red) with Desired Position');
    legend('Trajectory', 'Desired Position', 'Location', 'best');
else
    scatter3(0, 0, 0, 10, 'b'); hold on;
    scatter3(X_des, Y_des, -Z_des, 200, 'r*', 'LineWidth', 3); hold off;
    title('3D Trajectory - No Data Available');
    legend('Start', 'Desired Position', 'Location', 'best');
    grid on;
end

figure(3);
if ~isempty(time_arr) && ~isempty(ttarr) && length(time_arr) == length(ttarr)
    plot(time_arr, ttarr);
    title('Total Thrust Over Time');
    xlabel('Time (s)');
    ylabel('Thrust (N)');
    grid on;
else
    title('Total Thrust Over Time - No Data Available');
    xlabel('Time (s)');
    ylabel('Thrust (N)');
    grid on;
end

% Motor speeds plot (now for 6 motors) - IMPROVED READABILITY
figure(7);
if size(all_states, 1) > 0
    % Use more distinguishable colors and line styles
    plot(all_times, all_states(:, 13), '-', 'Color', [0.8 0.2 0.2], 'LineWidth', 1.5, 'DisplayName', 'Motor 1'); hold on;
    plot(all_times, all_states(:, 14), '-', 'Color', [0.2 0.6 0.2], 'LineWidth', 1.5, 'DisplayName', 'Motor 2');
    plot(all_times, all_states(:, 15), '-', 'Color', [0.2 0.2 0.8], 'LineWidth', 1.5, 'DisplayName', 'Motor 3');
    plot(all_times, all_states(:, 16), '-', 'Color', [0.8 0.5 0.0], 'LineWidth', 1.5, 'DisplayName', 'Motor 4');
    plot(all_times, all_states(:, 17), '-', 'Color', [0.6 0.2 0.6], 'LineWidth', 1.5, 'DisplayName', 'Motor 5');
    plot(all_times, all_states(:, 18), '-', 'Color', [0.2 0.7 0.7], 'LineWidth', 1.5, 'DisplayName', 'Motor 6');
    hold off;
    
    xlabel('Time (s)', 'FontSize', 11);
    ylabel('Motor Speed (rad/s)', 'FontSize', 11);
    title('Motor Speeds Over Time (6 motors)', 'FontSize', 12, 'FontWeight', 'bold');
    legend('Location', 'eastoutside', 'FontSize', 10);
    grid on;
    set(gca, 'FontSize', 10);
else
    title('Motor Speeds Over Time - No Data Available');
    xlabel('Time (s)');
    ylabel('Motor Speeds (rad/s)');
    grid on;
end

% Add figure for position errors
figure(5);

subplot(3,1,1);
if length(X_error_arr) > 0 && length(time_control) == length(X_error_arr)
    plot(time_control, X_error_arr, 'r-', 'LineWidth', 1.5);
else
    title('X Position Error - Data Size Mismatch');
end
title('X Position Error');
ylabel('m');
grid on;

subplot(3,1,2);
if length(Y_error_arr) > 0 && length(time_control) == length(Y_error_arr)
    plot(time_control, Y_error_arr, 'r-', 'LineWidth', 1.5);
else
    title('Y Position Error - Data Size Mismatch');
end
title('Y Position Error');
ylabel('m');
grid on;

subplot(3,1,3);
if ~isempty(all_states) && size(all_states, 1) > 0
    z_error_arr = all_states(:,3) - Z_des;  % Match Z-only convention
    plot(all_times, z_error_arr, 'r-', 'LineWidth', 1.5);
else
    plot([0 1], [0 0], 'r-');
    title('Z Position Error - No Data Available');
end
title('Z Position Error');
ylabel('m');
xlabel('Time (s)');
grid on;

% IMPROVED FIGURE 1 - Roll and Pitch angles together, XY positions together
figure(1);

subplot(3,1,1);
if ~isempty(all_states) && size(all_states, 1) > 0
    plot(all_times,all_states(:,1), 'b-', 'LineWidth', 1.5); hold on;
    plot([0 all_times(end)], [X_des X_des], 'b--', 'LineWidth', 1.5);
    plot(all_times,all_states(:,2), 'r-', 'LineWidth', 1.5);
    plot([0 all_times(end)], [Y_des Y_des], 'r--', 'LineWidth', 1.5);
    hold off;
    minScale = 1;
    all_pos = [all_states(:,1); all_states(:,2); X_des; Y_des];
    ylim([min(all_pos)-minScale, max(all_pos)+minScale]);
    legend('X Actual', 'X Target', 'Y Actual', 'Y Target', 'Location', 'best');
    title('X and Y Position');
    ylabel('Position (m)');
    grid on;
else
    plot([0 1], [X_des X_des], 'b--', 'LineWidth', 1.5); hold on;
    plot([0 1], [Y_des Y_des], 'r--', 'LineWidth', 1.5); hold off;
    title('X and Y Position - No Data Available');
    ylabel('Position (m)');
    legend('X Target', 'Y Target', 'Location', 'best');
    grid on;
end

subplot(3,1,2);
if ~isempty(all_states) && size(all_states, 1) > 0
    plot(all_times,rad2deg(all_states(:,4)), 'b-', 'LineWidth', 1.5); hold on;
    if length(Phi_des_arr) > 0 && length(time_control) == length(Phi_des_arr)
        plot(time_control, rad2deg(Phi_des_arr), 'b--', 'LineWidth', 1.5);
    end
    plot(all_times,rad2deg(all_states(:,5)), 'r-', 'LineWidth', 1.5);
    if length(Theta_des_arr) > 0 && length(time_control) == length(Theta_des_arr)
        plot(time_control, rad2deg(Theta_des_arr), 'r--', 'LineWidth', 1.5);
    end
    hold off;
    minScale = 5;
    all_angles = [rad2deg(all_states(:,4)); rad2deg(all_states(:,5))];
    ylim([min(all_angles)-minScale, max(all_angles)+minScale]);
    legend('Roll (φ) Actual', 'Roll Desired', 'Pitch (θ) Actual', 'Pitch Desired', 'Location', 'best');
    title('Roll and Pitch Angles');
    ylabel('Angle (deg)');
    grid on;
else
    plot([0 1], [0 0], 'b-');
    title('Roll and Pitch Angles - No Data Available');
    ylabel('Angle (deg)');
    grid on;
end

subplot(3,1,3);
if ~isempty(all_states) && size(all_states, 1) > 0
    plot(all_times,rad2deg(all_states(:,10)), 'b-', 'LineWidth', 1.5); hold on;
    plot(all_times,rad2deg(all_states(:,11)), 'r-', 'LineWidth', 1.5); hold off;
    minScale = 5;
    all_rates = [rad2deg(all_states(:,10)); rad2deg(all_states(:,11))];
    ylim([min(all_rates)-minScale, max(all_rates)+minScale]);
    legend('Roll Rate (P)', 'Pitch Rate (Q)', 'Location', 'best');
    title('Roll and Pitch Rates');
    ylabel('Angular Rate (deg/s)');
    xlabel('Time (s)');
    grid on;
else
    plot([0 1], [0 0], 'b-');
    title('Roll and Pitch Rates - No Data Available');
    ylabel('Angular Rate (deg/s)');
    xlabel('Time (s)');
    grid on;
end

% Additional simple XY plot
figure(6);
if ~isempty(all_states) && size(all_states, 1) > 0
    plot(all_times, all_states(:,1), 'b-', 'LineWidth', 1.5); hold on;
    plot(all_times, all_states(:,2), 'r-', 'LineWidth', 1.5); hold off;
    title('X and Y Position vs Time');
    xlabel('Time (s)');
    ylabel('Position (m)');
    legend('X Position', 'Y Position', 'Location', 'best');
    grid on;
else
    plot([0 1], [0 0], 'b-'); hold on;
    plot([0 1], [0 0], 'r-'); hold off;
    title('X and Y Position vs Time - No Data Available');
    xlabel('Time (s)');
    ylabel('Position (m)');
    legend('X Position', 'Y Position', 'Location', 'best');
    grid on;
end

%% CG UNCERTAINTY ANALYSIS PLOTS

% Figure 8: Robustness Margins Over Time
figure(8);
if length(time_control) == length(eps_phi_arr)
    subplot(2,1,1);
    plot(time_control, eps_phi_arr, 'r-', 'LineWidth', 1.5); hold on;
    plot(time_control, eps_theta_arr, 'b-', 'LineWidth', 1.5); hold off;
    ylabel('Robustness Margin');
    title('Robustness Margins (ε) Over Time');
    legend('ε_φ (roll)', 'ε_θ (pitch)', 'Location', 'best');
    grid on;
    ylim([0 0.5]);
    
    subplot(2,1,2);
    plot(time_control, alloc_error_arr, 'k-', 'LineWidth', 1.5);
    xlabel('Time (s)');
    ylabel('Allocation Error Norm (N·m)');
    title('Control Allocation Error Over Time');
    grid on;
    
    sgtitle(sprintf('CG Offset: Δx=%.1fmm, Δy=%.1fmm | Robust: %s', ...
        delta_x_cg*1000, delta_y_cg*1000, string(USE_ROBUST_ALLOCATION)));
else
    title('Robustness Margins - Data Size Mismatch');
end

% Figure 9: Stability Margin Over Time
figure(9);
if length(time_control) == length(stability_margin_arr)
    plot(time_control, stability_margin_arr, 'g-', 'LineWidth', 2);
    hold on;
    yline(0, 'r--', 'LineWidth', 1.5, 'Label', 'Unstable Threshold');
    yline(0.1, 'y--', 'LineWidth', 1, 'Label', 'Marginal');
    hold off;
    xlabel('Time (s)');
    ylabel('Stability Margin (1 - λ_{max})');
    title(sprintf('Closed-Loop Stability Under CG Uncertainty\nΔx=%.1fmm, Δy=%.1fmm', ...
        delta_x_cg*1000, delta_y_cg*1000));
    grid on;
    legend('Stability Margin', 'Location', 'best');
    ylim([min(stability_margin_arr)-0.1, 1]);
else
    title('Stability Margin - Data Size Mismatch');
end

%% trimmed state arrays for final lines
xarr = states(:,1);
yarr = states(:,2);
zarr = -1.*states(:,3);

Motor_Speeds = [];

%% Dynamics for the hexacopter
function st_dot = hex_dynamics(t,states)

    global m g alpha psiarr ttarr time_arr psi_counter delta_x_cg delta_y_cg delta_z_cg;
    
    % Nominal inertias (at geometric center)
    Ixx_nominal = 7.5e-3;
    Iyy_nominal = 7.5e-3;
    Izz_nominal = 2.3e-2;
    
    % Adjust inertia using parallel axis theorem: I = I_nom + m*d^2
    % where d is the perpendicular distance from the new axis
    Ixx = Ixx_nominal + m * (delta_y_cg^2 + delta_z_cg^2);
    Iyy = Iyy_nominal + m * (delta_x_cg^2 + delta_z_cg^2);
    Izz = Izz_nominal + m * (delta_x_cg^2 + delta_y_cg^2);

    phi = states(4);
    theta = states(5);
    psi = states(6);
    u = states(7);
    v = states(8);
    w = states(9);
    p = states(10);
    q = states(11);
    r = states(12);
    w1 = states(13);
    w2 = states(14);
    w3 = states(15);
    w4 = states(16);
    w5 = states(17);
    w6 = states(18);

    % kM used in allocation approximate; actual yaw torque used below is Kd
    Kd = 2.047e-8;

    l = 0.05; % arm length (m) - reduced from 0.25m
    tipV = l*r; % m/s

    % J and inv_J logic kept (was in your quad code)
    J = tipV/(0.25*(w1 + 0.1));
    inv_J = max(0,(1-J)); % fixes Kd as the velocity changes

    % Motor thrusts (MotorForce kept same signature)
    MF = [
        MotorForce(w1,4.1,4.1,tipV,alpha)
        MotorForce(w2,4.1,4.1,tipV,alpha)
        MotorForce(w3,4.1,4.1,tipV,alpha)
        MotorForce(w4,4.1,4.1,tipV,alpha)
        MotorForce(w5,4.1,4.1,tipV,alpha)
        MotorForce(w6,4.1,4.1,tipV,alpha)
    ];

    % Motor positions angles (deg) consistent with main file
    angles_deg = [30, 90, 150, 210, 270, 330];
    angles_rad = deg2rad(angles_deg);

    % --- Moments from vertical thrust with CG offset correction ---
    % Effective moment arms change due to CG offset
    M_phi = 0;
    M_theta = 0;
    
    for i = 1:6
        alpha_i = angles_rad(i);
        
        % Motor position relative to geometric center
        motor_x = l * cos(alpha_i);
        motor_y = l * sin(alpha_i);
        motor_z = 0;  % motors in same plane
        
        % Position relative to actual CG
        r_x = motor_x - delta_x_cg;
        r_y = motor_y - delta_y_cg;
        r_z = motor_z - delta_z_cg;
        
        % Thrust force in body frame (vertical, alpha=0)
        F_z = MF(i);
        
        % Moments: M = r × F
        % M_phi (roll) = r_y * F_z (thrust at offset y creates roll)
        % M_theta (pitch) = -r_x * F_z (thrust at offset x creates pitch)
        M_phi = M_phi - r_y * F_z;
        M_theta = M_theta + r_x * F_z;
    end

    % Yaw moment from rotor torque: Kd * direction * w^2 * inv_J
    rot_dir = [1, -1, 1, -1, 1, -1]; % must match main script
    M_psi = Kd*(rot_dir(1)*w1^2*inv_J + rot_dir(2)*w2^2*inv_J + rot_dir(3)*w3^2*inv_J + rot_dir(4)*w4^2*inv_J + rot_dir(5)*w5^2*inv_J + rot_dir(6)*w6^2*inv_J);

    % Store values
    if psi_counter <= length(psiarr)
        psiarr(psi_counter) = M_psi;
        ttarr(psi_counter) = sum(MF); % total vertical thrust (alpha=0 => cosd(alpha)=1)
        time_arr = [time_arr t];
        psi_counter = psi_counter + 1;
    end

    Tt = sum(MF);  % total thrust

    V = [u;v;w];

    xyz_dot = [
        cos(theta)*cos(psi) (cos(psi)*sin(phi)*sin(theta) - cos(phi)*sin(psi)) (sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta));
        cos(theta)*sin(psi) (cos(phi)*cos(psi) + sin(phi)*sin(theta)*sin(psi)) (cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi));
        (-sin(theta)) (cos(theta)*sin(phi)) (cos(theta)*cos(phi))
        ]*V;

    ptp_dot = [
        1 (sin(phi)*tan(theta)) (cos(phi)*tan(theta));
        0 cos(phi) (-sin(phi));
        0 (sin(phi)*sec(theta)) (cos(phi)*sec(theta))]*[p;q;r];
 
    uvw_dot = [
        -((0*V(1))/m) r -q;
        -r -((0*V(2))/m) p;
        q -p -(0*V(3)/m)]*[u;v;w] + [-g*sin(theta); +g*sin(phi)*cos(theta); + (g*cos(phi)*cos(theta) - (Tt)/m)];

    alpha_R = 6; % linear damping co-eff of drag
    
    % Gravity-induced moments from CG offset
    % When tilted, offset CG creates restoring/destabilizing moments
    % M_g = r_cg × F_g (in body frame)
    M_g_phi = m * g * delta_z_cg * sin(phi) * cos(theta) - m * g * delta_y_cg * cos(phi) * cos(theta);
    M_g_theta = m * g * delta_x_cg * cos(phi) * cos(theta) + m * g * delta_z_cg * sin(theta);
    M_g_psi = 0;  % CG offset doesn't create yaw moment from gravity

    p_dot = ((Iyy - Izz)/Ixx)*q*r + (M_phi + M_g_phi)/Ixx;
    q_dot = ((Izz - Ixx)/Iyy)*p*r + (M_theta + M_g_theta)/Iyy;
    r_dot = ((Ixx - Iyy)/Izz)*p*q + (M_psi + M_g_psi)/Izz - alpha_R*r;

    % Last 6 entries correspond to motor speeds (treated as states but no dynamics here)
    st_dot = [xyz_dot(1,:);xyz_dot(2,:);xyz_dot(3,:);ptp_dot(1,:);ptp_dot(2,:);ptp_dot(3,:);uvw_dot(1,:);uvw_dot(2,:);uvw_dot(3,:);p_dot;q_dot;r_dot;0;0;0;0;0;0];
end

function mf = MotorForce(w,d,pitch,V0,aoa)
    % unchanged motor force model
    mf = 4.392399e-8*(w*60/(2*pi))*((d^3.5)/sqrt(pitch))*((4.23333e-4 * (w*60/(2*pi)) * pitch) - V0*sind(aoa));
end
