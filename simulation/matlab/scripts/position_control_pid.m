% Position Control PID on top of change_tilt.m

% Initialize global variables for integration with change_tilt
global x_des y_des x_err_integral y_err_integral x_err_prev y_err_prev

% Desired world frame positions
x_des = 0; % Desired x position (m)
y_des = 0; % Desired y position (m)


% Function to compute desired world pitch and roll based on position error
function [phi_world_des, theta_world_des] = position_pid_control(x_current, y_current, dt)
    global x_des y_des
    
    % PID coefficients for position control
    Kp_pos = 1;  % Proportional gain
    Ki_pos = 0;  % Integral gain
    Kd_pos = 0;  % Derivative gain
    
    % Initialize error integrals and previous errors
    x_err_integral = 0; x_err_prev = 0;
    y_err_integral = 0; y_err_prev = 0;


    % Compute position errors
    x_error = x_current - x_des;
    y_error = y_current - y_des;
    % Compute error derivatives
    x_error_dot = (x_error - x_err_prev) / dt;
    y_error_dot = (y_error - y_err_prev) / dt;

    % Update error integrals
    x_err_integral = x_err_integral + x_error * dt;
    y_err_integral = y_err_integral + y_error * dt;

    % Compute control outputs
    theta_world_des = -(Kp_pos * x_error + Ki_pos * x_err_integral + Kd_pos * x_error_dot); % Desired pitch
    phi_world_des = Kp_pos * y_error + Ki_pos * y_err_integral + Kd_pos * y_error_dot;      % Desired roll
    
    % Update previous errors
    x_err_prev = x_error;
    y_err_prev = y_error;
end

% Copy of change_tilt.m starts here
close all;

global m g alpha time_arr psi_counter
time_arr = [];
psi_counter = 1;
m = 2;
g = 9.81;
Ct = 0.0027;
rho = 1.225;
Kd = 1.58e-4; % = Ct rho d^4 for small propellers

w_init = 3199.5;

frequency = 100; % Controller Frequency in Hertz
timePeriod = 50; % Simulation time
tspan = [0 1/frequency];
stINIT = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, w_init, w_init, w_init, w_init];
options = odeset('RelTol', 1e-4, 'AbsTol', 1e-6);

% Initialize arrays for storing states and time values
max_iterations = timePeriod * frequency;
all_states = zeros(max_iterations * 50, 16);
all_times = zeros(max_iterations * 50, 1);
states_counter = 1;

i = 1;

% Desired world frame attitudes and altitude
Z_des = -10; % Desired altitude

% PID Coefficients - Attitude and Altitude Control
Kp_Z = 500; Ki_Z = 0; Kd_Z = 5000;
Kp_att_max = 500; Ki_att = 0; Kd_att = 0;
Kp_att_min = 50; Kff_att = 0;

% Adaptive gain parameters
att_deadband = deg2rad(0.5);

% Initialize error integrals and previous errors
Z_err_integral = 0; Z_err_prev = 0;
phi_err_integral = 0; phi_err_prev = 0;
theta_err_integral = 0; theta_err_prev = 0;

% Initialize yaw tracking variables
psi_prev = 0; psi_rate = 0; psi_rate_prev = 0;

% Pre-allocate arrays for better performance
T_des_arr = zeros(1, max_iterations);
time_des_arr = zeros(1, max_iterations);

global psiarr ttarr
psiarr = zeros(1, max_iterations * 50);
ttarr = zeros(1, max_iterations * 50);
motor_speed = zeros(4, max_iterations);

while i < (timePeriod * frequency)
    current_time = tspan(1);
    alpha = 30 * (1 / (1 + exp(-4 * (current_time - 0))));
    
    if current_time > 0
        w_init = 3499.5;
        Kp_Z = 700; Ki_Z = 1; Kd_Z = 1500;
    end
    
    [t, states] = ode45(@quad_dynamics, tspan, stINIT, options);

    % Store states
    new_data_points = length(t);
    all_states(states_counter:states_counter+new_data_points-1, :) = states;
    all_times(states_counter:states_counter+new_data_points-1) = t;
    states_counter = states_counter + new_data_points;
    
    % Update initial conditions for the next iteration
    stINIT = states(end, :);
    
    % Current state
    z_current = states(end, 3);
    phi_current = states(end, 4);
    theta_current = states(end, 5);
    psi_current = states(end, 6);
    r_current = states(end, 12);  % Current yaw rate
    
    dt = 1 / frequency;
    
    % Estimate yaw rate and predict future yaw
    psi_rate = (psi_current - psi_prev) / dt;
    psi_rate_filtered = 0.7 * psi_rate + 0.3 * psi_rate_prev;

    % Predict yaw at next time step for feedforward
    psi_predicted = psi_current + psi_rate_filtered * dt / 100;
    [phi_world_des, theta_world_des] = position_pid_control(states(end, 1), states(end, 2), dt);

    % Convert world frame desired angles to body frame using predicted yaw
    phi_body_des = phi_world_des * cos(psi_predicted) + theta_world_des * sin(psi_predicted);
    theta_body_des = -phi_world_des * sin(psi_predicted) + theta_world_des * cos(psi_predicted);
    
    % Additional feedforward compensation for yaw rate coupling
    phi_feedforward = -theta_world_des * psi_rate_filtered * Kff_att;
    theta_feedforward = phi_world_des * psi_rate_filtered * Kff_att;
    
    % Z position error (no adaptive gain)
    z_error = z_current - Z_des;
    z_error_dot = (z_error - Z_err_prev) / dt;
    Z_err_integral = Z_err_integral + z_error * dt;
    
    % Body frame attitude errors
    phi_error = phi_current - phi_body_des;
    theta_error = theta_current - theta_body_des;
    
    % Calculate combined attitude error for adaptive gain
    att_error_combined = sqrt(phi_error^2 + theta_error^2);
    
    % Adaptive attitude gain
    if att_error_combined <= att_deadband
        att_gain_factor = att_error_combined / att_deadband;
        Kp_att = Kp_att_min + (Kp_att_max - Kp_att_min) * att_gain_factor;
    else
        Kp_att = Kp_att_max;
    end
    
    phi_error_dot = (phi_error - phi_err_prev) / dt;
    phi_err_integral = phi_err_integral + phi_error * dt;
    
    theta_error_dot = (theta_error - theta_err_prev) / dt;
    theta_err_integral = theta_err_integral + theta_error * dt;
    
    % Control outputs with feedforward
    z_ctrl_adj = Kp_Z * z_error + Ki_Z * Z_err_integral + Kd_Z * z_error_dot;
    phi_ctrl = Kp_att * phi_error + Ki_att * phi_err_integral + Kd_att * phi_error_dot + phi_feedforward;
    theta_ctrl = Kp_att * theta_error + Ki_att * theta_err_integral + Kd_att * theta_error_dot + theta_feedforward;
    
    upper_limit = 6000;
    lower_limit = 0;
    
    % Base motor speed from altitude control
    w_base = max(lower_limit, min(w_init + z_ctrl_adj, upper_limit));
    
    % FIXED CONTROL ALLOCATION
    l = 0.25; % arm length
    
    % Convert moments to force differentials using proper scaling
    force_scale = 10; % Scale factor to convert con      trol to force
    delta_f_phi = phi_ctrl * force_scale;
    delta_f_theta = theta_ctrl * force_scale;
    
    % CORRECTED Motor allocation based on moment equations:
    % M_phi = (f1 + f2 - f3 - f4) * l * cos(alpha)
    % M_theta = (-f1 + f2 + f3 - f4) * l * cos(alpha)
    
    % Solve for individual motor force adjustments
    delta_f1 = delta_f_phi/4 - delta_f_theta/4;
    delta_f2 = delta_f_phi/4 + delta_f_theta/4;
    delta_f3 = -delta_f_phi/4 + delta_f_theta/4;
    delta_f4 = -delta_f_phi/4 - delta_f_theta/4;
    
    % Convert force to motor speed (approximate scaling)
    speed_scale = 0.1; % Adjust this based on your motor characteristics
    delta_w1 = delta_f1 * speed_scale;
    delta_w2 = delta_f2 * speed_scale;
    delta_w3 = delta_f3 * speed_scale;
    delta_w4 = delta_f4 * speed_scale;
    
    % Apply motor speeds with limits
    w1 = max(lower_limit, min(w_base + delta_w1, upper_limit));
    w2 = max(lower_limit, min(w_base + delta_w2, upper_limit));
    w3 = max(lower_limit, min(w_base + delta_w3, upper_limit));
    w4 = max(lower_limit, min(w_base + delta_w4, upper_limit));
    fade_time = 10;
    fade_factor = (1 - 1/(1 + exp(-4*(current_time - fade_time))));  % Fixed timing like Z-only
    w1 = w1*fade_factor;

    % Store motor speeds for plotting
    if i <= max_iterations
        motor_speed(:, i) = [w1; w2; w3; w4];
        T_des_arr(i) = w_base;
        time_des_arr(i) = current_time;
    end
    
    % Apply motor speeds
    stINIT(13) = w1;
    stINIT(14) = w2;
    stINIT(15) = w3;
    stINIT(16) = w4;
    
    % Store previous errors and yaw for next iteration
    Z_err_prev = z_error;
    phi_err_prev = phi_error;
    theta_err_prev = theta_error;
    psi_prev = psi_current;
    psi_rate_prev = psi_rate_filtered;

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
T_des_arr = T_des_arr(1:actual_iterations);
time_des_arr = time_des_arr(1:actual_iterations);
motor_speed = motor_speed(:, 1:actual_iterations);

fprintf('World frame attitude control simulation complete.\n');
fprintf('Desired world frame: Roll=%.2f deg, Pitch=%.2f deg, Z=%.2f m\n', ...
    rad2deg(phi_world_des), rad2deg(theta_world_des), Z_des);

%% PLOT RESULTS

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
title('Yaw Angle: Psi');
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

% World frame attitude tracking
figure(8);
subplot(2,1,1);
if ~isempty(all_states) && size(all_states, 1) > 0
    % Convert body frame angles to world frame for plotting
    phi_world = zeros(size(all_times));
    theta_world = zeros(size(all_times));
    
    for j = 1:length(all_times)
        psi = all_states(j,6);
        phi_body = all_states(j,4);
        theta_body = all_states(j,5);
        
        % Body frame to world frame transformation
        phi_world(j) = phi_body * cos(psi) - theta_body * sin(psi);
        theta_world(j) = phi_body * sin(psi) + theta_body * cos(psi);
    end
    
    % Compute combined limits (degrees) and apply to both subplots
    phi_world_deg = rad2deg(phi_world);
    theta_world_deg = rad2deg(theta_world);
    combined_min = min([phi_world_deg; theta_world_deg]);
    combined_max = max([phi_world_deg; theta_world_deg]);
    margin = 0.5; % degrees margin
    
    % Calculate average roll angle
    avg_phi_world_deg = mean(phi_world_deg);
    
    plot(all_times, phi_world_deg, 'b-', 'LineWidth', 1.5); hold on;
    plot([0 all_times(end)], [rad2deg(phi_world_des) rad2deg(phi_world_des)], 'r--', 'LineWidth', 1.5);
    plot([0 all_times(end)], [avg_phi_world_deg avg_phi_world_deg], 'g-.', 'LineWidth', 1.5); % Average line
    hold off;
    title('World Frame Roll Angle');
    ylabel('deg');
    ylim([combined_min - margin, combined_max + margin]);
    legend('Actual', 'Desired', 'Average', 'Location', 'best');
    grid on;
else
    plot([0 1], [0 0], 'b-');
    title('World Frame Roll - No Data Available');
end

subplot(2,1,2);
if ~isempty(all_states) && size(all_states, 1) > 0
    % Calculate average pitch angle
    avg_theta_world_deg = mean(theta_world_deg);
    
    plot(all_times, theta_world_deg, 'b-', 'LineWidth', 1.5); hold on;
    plot([0 all_times(end)], [rad2deg(theta_world_des) rad2deg(theta_world_des)], 'r--', 'LineWidth', 1.5);
    plot([0 all_times(end)], [avg_theta_world_deg avg_theta_world_deg], 'g-.', 'LineWidth', 1.5); % Average line
    hold off;
    title('World Frame Pitch Angle');
    ylabel('deg');
    xlabel('Time (s)');
    ylim([combined_min - margin, combined_max + margin]);
    legend('Actual', 'Desired', 'Average', 'Location', 'best');
    grid on;
else
    plot([0 1], [0 0], 'b-');
    title('World Frame Pitch - No Data Available');
end

% Add debugging plot for motor differences - IMPROVED VISUALIZATION
figure(9);
if size(motor_speed, 2) > 0
    % Option 1: Subplots for each motor
    subplot(2,2,1);
    plot(time_des_arr, motor_speed(1,:) - mean(motor_speed,1), 'r-', 'LineWidth', 2);
    title('Motor 1 Speed Difference');
    ylabel('Speed Diff (rad/s)');
    grid on;
    
    subplot(2,2,2);
    plot(time_des_arr, motor_speed(2,:) - mean(motor_speed,1), 'g-', 'LineWidth', 2);
    title('Motor 2 Speed Difference');
    ylabel('Speed Diff (rad/s)');
    grid on;
    
    subplot(2,2,3);
    plot(time_des_arr, motor_speed(3,:) - mean(motor_speed,1), 'b-', 'LineWidth', 2);
    title('Motor 3 Speed Difference');
    ylabel('Speed Diff (rad/s)');
    xlabel('Time (s)');
    grid on;
    
    subplot(2,2,4);
    plot(time_des_arr, motor_speed(4,:) - mean(motor_speed,1), 'k-', 'LineWidth', 2);
    title('Motor 4 Speed Difference');
    ylabel('Speed Diff (rad/s)');
    xlabel('Time (s)');
    grid on;
    
    sgtitle('Motor Speed Differences from Average (Individual)');
end

% Alternative combined plot with different line styles
figure(11);
if size(motor_speed, 2) > 0
    plot(time_des_arr, motor_speed(1,:) - mean(motor_speed,1), 'r-', 'LineWidth', 2, 'DisplayName', 'w1 - avg'); hold on;
    plot(time_des_arr, motor_speed(2,:) - mean(motor_speed,1), 'g--', 'LineWidth', 2, 'DisplayName', 'w2 - avg');
    plot(time_des_arr, motor_speed(3,:) - mean(motor_speed,1), 'b:', 'LineWidth', 3, 'DisplayName', 'w3 - avg');
    plot(time_des_arr, motor_speed(4,:) - mean(motor_speed,1), 'k-.', 'LineWidth', 2, 'DisplayName', 'w4 - avg');
    hold off;
    title('Motor Speed Differences from Average');
    xlabel('Time (s)');
    ylabel('Speed Difference (rad/s)');
    legend('Location', 'best');
    grid on;
end

% Keep all your other plotting code...
figure(2);

if ~isempty(all_states) && size(all_states, 1) > 0
    startColor = [0, 0, 1];     % Start color (blue)
    endColor = [1, 0, 0];       % End color (red)
    numPoints = size(all_states, 1);
    colorMap = [linspace(startColor(1), endColor(1), numPoints)', ...
                linspace(startColor(2), endColor(2), numPoints)', ...
                linspace(startColor(3), endColor(3), numPoints)'];

    scatter3(all_states(:,1),all_states(:,2),-1.*all_states(:,3),10,colorMap); hold on;
    % Plot desired position as a red star
    scatter3(0, 0, -Z_des, 200, 'r*', 'LineWidth', 3); hold off;

    minScale = 1;
    xlim([min(all_states(:,1))-minScale, max(all_states(:,1))+minScale]);
    ylim([min(all_states(:,2))-minScale, max(all_states(:,2))+minScale]);
    zlim([min(-1.*all_states(:,3))-minScale, max(-1.*all_states(:,3))+minScale]);
    grid on;
    title('3D Trajectory (Blue to Red) - World Frame Attitude Control');
    legend('Trajectory', 'Desired Altitude', 'Location', 'best');
else
    scatter3(0, 0, 0, 10, 'b'); hold on;
    scatter3(0, 0, -Z_des, 200, 'r*', 'LineWidth', 3); hold off;
    title('3D Trajectory - No Data Available');
    legend('Start', 'Desired Altitude', 'Location', 'best');
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

% Motor speeds plot
figure(7);
if size(all_states, 1) > 0
    plot(all_times, all_states(:, 13), 'r-', 'DisplayName', 'w1'); hold on;
    plot(all_times, all_states(:, 14), 'g-', 'DisplayName', 'w2');
    plot(all_times, all_states(:, 15), 'b-', 'DisplayName', 'w3');
    plot(all_times, all_states(:, 16), 'k-', 'DisplayName', 'w4');
    hold off;
    
    xlabel('Time (s)');
    ylabel('Motor Speeds (rad/s)');
    title('Motor Speeds Over Time');
    legend('Location', 'best');
    grid on;
else
    title('Motor Speeds Over Time - No Data Available');
    xlabel('Time (s)');
    ylabel('Motor Speeds (rad/s)');
    grid on;
end

figure(1);

subplot(4,1,1);
if ~isempty(all_states) && size(all_states, 1) > 0
    plot(all_times,all_states(:,1), 'b-', 'LineWidth', 1.5);
    minScale = 1;
    ylim([min(all_states(:,1))-minScale, max(all_states(:,1))+minScale]);
    title('X Position');
else
    plot([0 1], [0 0], 'b-');
    title('X Position - No Data Available');
end
ylabel('m');

subplot(4,1,2);
if ~isempty(all_states) && size(all_states, 1) > 0
    plot(all_times,all_states(:,2), 'b-', 'LineWidth', 1.5);
    minScale = 1;
    ylim([min(all_states(:,2))-minScale, max(all_states(:,2))+minScale]);
    title('Y Position');
else
    plot([0 1], [0 0], 'b-');
    title('Y Position - No Data Available');
end
ylabel('m');

subplot(4,1,3);
if ~isempty(all_states) && size(all_states, 1) > 0
    plot(all_times,rad2deg(all_states(:,4)), 'b-', 'LineWidth', 1.5);
    % Calculate combined min/max for consistent scaling
    phi_deg = rad2deg(all_states(:,4));
    theta_deg = rad2deg(all_states(:,5));
    combined_min = min([phi_deg; theta_deg]);
    combined_max = max([phi_deg; theta_deg]);
    minScale = 5;
    ylim([combined_min-minScale, combined_max+minScale]);
    title('Body Frame Roll Angle (φ)');
else
    plot([0 1], [0 0], 'b-');
    title('Body Frame Roll Angle (φ) - No Data Available');
end
ylabel('deg');

subplot(4,1,4);
if ~isempty(all_states) && size(all_states, 1) > 0
    plot(all_times,rad2deg(all_states(:,5)), 'b-', 'LineWidth', 1.5);
    % Use same scaling as roll angle plot
    phi_deg = rad2deg(all_states(:,4));
    theta_deg = rad2deg(all_states(:,5));
    combined_min = min([phi_deg; theta_deg]);
    combined_max = max([phi_deg; theta_deg]);
    minScale = 5;
    ylim([combined_min-minScale, combined_max+minScale]);
    title('Body Frame Pitch Angle (θ)');
else
    plot([0 1], [0 0], 'b-');
    title('Body Frame Pitch Angle (θ) - No Data Available');
end
ylabel('deg');

% Add new figure for velocity rates
figure(10);

subplot(2,1,1);
if ~isempty(all_states) && size(all_states, 1) > 0
    % Convert body frame velocities to world frame
    x_rate_world = zeros(size(all_times));
    y_rate_world = zeros(size(all_times));
    
    for j = 1:length(all_times)
        psi = all_states(j,6);
        phi = all_states(j,4);
        theta = all_states(j,5);
        u = all_states(j,7);  % body frame x velocity
        v = all_states(j,8);  % body frame y velocity
        w = all_states(j,9);  % body frame z velocity
        
        % Body to world frame velocity transformation
        R_body_to_world = [
            cos(theta)*cos(psi), (cos(psi)*sin(phi)*sin(theta) - cos(phi)*sin(psi)), (sin(phi)*sin(psi) + cos(phi)*cos(psi)*sin(theta));
            cos(theta)*sin(psi), (cos(phi)*cos(psi) + sin(phi)*sin(theta)*sin(psi)), (cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi));
            -sin(theta), cos(theta)*sin(phi), cos(theta)*cos(phi)
        ];
        
        world_vel = R_body_to_world * [u; v; w];
        x_rate_world(j) = world_vel(1);
        y_rate_world(j) = world_vel(2);
    end
    
    plot(all_times, x_rate_world, 'b-', 'LineWidth', 1.5);
    minScale = 0.5;
    ylim([min(x_rate_world)-minScale, max(x_rate_world)+minScale]);
    title('X Rate (World Frame Velocity)');
    ylabel('m/s');
    grid on;
else
    plot([0 1], [0 0], 'b-');
    title('X Rate (World Frame) - No Data Available');
end

subplot(2,1,2);
if ~isempty(all_states) && size(all_states, 1) > 0
    plot(all_times, y_rate_world, 'b-', 'LineWidth', 1.5);
    minScale = 0.5;
    ylim([min(y_rate_world)-minScale, max(y_rate_world)+minScale]);
    title('Y Rate (World Frame Velocity)');
    ylabel('m/s');
    xlabel('Time (s)');
    grid on;
else
    plot([0 1], [0 0], 'b-');
    title('Y Rate (World Frame) - No Data Available');
end

%%

xarr = all_states(:,1);
yarr = all_states(:,2);
zarr = -1.*all_states(:,3);

%%
function st_dot = quad_dynamics(t,states)

    global m g alpha psiarr ttarr time_arr psi_counter;
    Ixx = 7.5e-3;
    Iyy = 7.5e-3;
    Izz = 2.3e-2;

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

    Kd = 2.047e-8;

    l = 0.25; %m arm length
    tipV = l*r; % m/s
    J = tipV/(0.25*(w1 + 0.1)); 
    inv_J = max(0,(1-J)); % fixes Kd as the velocity changes

    MF = [
        MotorForce(w1,4.1,4.1,tipV,alpha)
        MotorForce(w2,4.1,4.1,tipV,alpha)
        MotorForce(w3,4.1,4.1,tipV,alpha)
        MotorForce(w4,4.1,4.1,tipV,alpha)
    ];

    M_phi = MF(1)*l*cosd(alpha) + MF(2)*l*cosd(alpha) - MF(3)*l*cosd(alpha) - MF(4)*l*cosd(alpha);
    M_theta = -MF(1)*l*cosd(alpha) + MF(2)*l*cosd(alpha) - MF(3)*l*cosd(alpha) + MF(4)*l*cosd(alpha);
    M_psi = Kd*w1^2*cosd(alpha)*inv_J + MF(1)*l*sind(alpha) + Kd*w2^2*cosd(alpha)*inv_J + MF(2)*l*sind(alpha) + Kd*w3^2*cosd(alpha)*inv_J + MF(3)*l*sind(alpha) + Kd*w4^2*cosd(alpha)*inv_J + MF(4)*l*sind(alpha);
    
    % Store values more efficiently
    if psi_counter <= length(psiarr)
        psiarr(psi_counter) = M_psi;
        ttarr(psi_counter) = MF(1)*cosd(alpha) + MF(2)*cosd(alpha) + MF(3)*cosd(alpha) + MF(4)*cosd(alpha);
        time_arr = [time_arr t];
        psi_counter = psi_counter + 1;
    end
    
    Tt = MF(1)*cosd(alpha) + MF(2)*cosd(alpha) + MF(3)*cosd(alpha) + MF(4)*cosd(alpha);  
    
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

    p_dot = ((Iyy - Izz)/Ixx)*q*r + (M_phi)/Ixx;
    q_dot = ((Izz - Ixx)/Iyy)*p*r + (M_theta)/Iyy ;
    r_dot = ((Ixx - Iyy)/Izz)*p*q + (M_psi)/Izz - alpha_R*r;

    st_dot = [xyz_dot(1,:);xyz_dot(2,:);xyz_dot(3,:);ptp_dot(1,:);ptp_dot(2,:);ptp_dot(3,:);uvw_dot(1,:);uvw_dot(2,:);uvw_dot(3,:);p_dot;q_dot;r_dot;0;0;0;0];
end

function mf = MotorForce(w,d,pitch,V0,aoa)
    mf = 4.392399e-8*(w*60/(2*pi))*((d^3.5)/sqrt(pitch))*((4.23333e-4 * (w*60/(2*pi)) * pitch) - V0*sind(aoa));
end