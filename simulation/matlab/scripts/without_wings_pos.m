clear;
clc;
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

frequency = 100; %Controller Frequency in Hertz (increased for speed)
timePeriod = 20; % Simulation time
tspan = [0 1/frequency];
stINIT = [ 0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 , w_init, w_init, w_init, w_init];
%stIN = [x, y, z, phi, theta, psi, u, v, w, p, q, r, Mphi, MTheta, MPsi, Force_z];
options = odeset('RelTol',1e-4,'AbsTol',1e-6);

% Initialize arrays for storing states and time values
max_iterations = timePeriod * frequency;
all_states = zeros(max_iterations * 50, 16);
all_times = zeros(max_iterations * 50, 1);
states_counter = 1;

i = 1;

% Desired positions - simple trajectory
X_des = 10;  % Start at origin
Y_des = 10;  
Z_des = 0; % Start with small altitude

% PID Coefficients for position control
Kp_pos_xy = 1.4;  Ki_pos_xy = 0; Kd_pos_xy = 80;  % Fixed position gains
Kp_Z = 500;       Ki_Z = 0;          Kd_Z = 5000;      % Z position control gains

% PID Coefficients for angle control
Kp_ang = 0.002; Ki_ang = 0; Kd_ang = 0;

% Initialize error integrals and previous errors
X_err_integral = 0; X_err_prev = 0;
Y_err_integral = 0; Y_err_prev = 0;
Z_err_integral = 0; Z_err_prev = 0;

Phi_err_integral = 0; Phi_err_prev = 0;
Theta_err_integral = 0; Theta_err_prev = 0;

% Define maximum allowable angles (in radians)
phi_max = deg2rad(10); % Maximum roll angle limit
theta_max = deg2rad(10); % Maximum pitch angle limit
% 
% % Setup real-time 3D trajectory plot
% figure('Name', 'Real-time 3D Trajectory', 'Position', [100, 100, 800, 600]);
% h_traj = plot3(0, 0, 0, 'b-', 'LineWidth', 2);
% hold on;
% h_current = plot3(0, 0, 0, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
% h_target = plot3(X_des, Y_des, -Z_des, 'g*', 'MarkerSize', 12, 'LineWidth', 2);
% grid on;
% xlabel('X Position (m)');
% ylabel('Y Position (m)');
% zlabel('Z Position (m)');
% title('Real-time 3D Trajectory');
% legend('Trajectory', 'Current Position', 'Target Position', 'Location', 'best');
% axis equal;
% view(45, 30);
% drawnow;

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

global psiarr ttarr
psiarr = zeros(1, max_iterations * 50);
ttarr = zeros(1, max_iterations * 50);
motor_speed = zeros(1, max_iterations);

% Debug: Add this to check if the function is being called

while i < (timePeriod * frequency)
    current_time = tspan(1);
    alpha = 30*(1/(1 + exp(-4*(current_time - 1))));
    
    if current_time > 0
        w_init = 3499.5;
        Kp_Z = 700;    Ki_Z = 1;   Kd_Z = 1500;
        Kp_pos_xy = 1; Kd_pos_xy = 100; 
        Kp_ang = 1; Kd_ang = 0.1;
    end
    
    [t, states] = ode45(@quad_dynamics, tspan, stINIT, options);

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
    ax_des_world = Kp_pos_xy * x_error + Ki_pos_xy * X_err_integral + Kd_pos_xy * x_error_dot;
    ay_des_world = Kp_pos_xy * y_error + Ki_pos_xy * Y_err_integral + Kd_pos_xy * y_error_dot;
    
    % Transform desired accelerations from world frame to body frame considering yaw
    % Rotation matrix for yaw transformation (world to body)
    cos_psi = cos(psi_current);
    sin_psi = sin(psi_current);
    
    ax_des_body = cos_psi * ax_des_world + sin_psi * ay_des_world;
    ay_des_body = -sin_psi * ax_des_world + cos_psi * ay_des_world;
    
    % Convert desired body accelerations to desired roll and pitch angles
    % For small angles: ax ≈ g*theta, ay ≈ -g*phi
    phi_des = -ay_des_body / g;  % Desired roll angle
    theta_des = ax_des_body / g; % Desired pitch angle
    
    % Limit desired angles to maximum allowable values
    phi_des = max(-phi_max, min(phi_max, phi_des));
    theta_des = max(-theta_max, min(theta_max, theta_des));

    % Current angles
    phi_current = states(end,4);
    theta_current = states(end,5);
    
    % Angle errors
    phi_error = phi_des - phi_current;
    theta_error = theta_des - theta_current;
    
    % Angle error derivatives
    phi_error_dot = (phi_error - Phi_err_prev) / dt;
    theta_error_dot = (theta_error - Theta_err_prev) / dt;
    
    % Angle error integrals
    Phi_err_integral = Phi_err_integral + phi_error * dt;
    Theta_err_integral = Theta_err_integral + theta_error * dt;
    
    % Altitude control (Z-axis) - use Z-only method
    z_ctrl_adj = Kp_Z * z_error + Ki_Z * Z_err_integral + Kd_Z * z_error_dot;
    
    upper_limit = 6000;
    lower_limit = 0;
    
    % Base motor speed for altitude control
    w_base = max(lower_limit, min(w_init + z_ctrl_adj, upper_limit));
    
    % Attitude control adjustments - use normal angle gains
    phi_ctrl = Kp_ang * phi_error + Ki_ang * Phi_err_integral + Kd_ang * phi_error_dot;
    theta_ctrl = Kp_ang * theta_error + Ki_ang * Theta_err_integral + Kd_ang * theta_error_dot;
    
    % Motor mixing for attitude control (use smaller adjustments like Z-only)
    % Motor layout: 1-front-left, 2-front-right, 3-rear-right, 4-rear-left
    motor_phi_adjustment = phi_ctrl * 50;   % Scale factor for roll control
    motor_theta_adjustment = theta_ctrl * 50; % Scale factor for pitch control
    
    % Apply motor mixing
    w1 = max(lower_limit, min(w_base - motor_phi_adjustment - motor_theta_adjustment, upper_limit));
    w2 = max(lower_limit, min(w_base + motor_phi_adjustment - motor_theta_adjustment, upper_limit));
    w3 = max(lower_limit, min(w_base + motor_phi_adjustment + motor_theta_adjustment, upper_limit));
    w4 = max(lower_limit, min(w_base - motor_phi_adjustment + motor_theta_adjustment, upper_limit));
    
    % Store motor speed for plotting
    if i <= max_iterations
        motor_speed(i) = w_base;
    end
        
    % Apply motor speeds with time-based fade-in (like Z-only)
    fade_factor = (1 - 1/(1 + exp(-4*(current_time - 200))));  % Fixed timing like Z-only
    stINIT(13) = w1;
    stINIT(14) = w2 * fade_factor; 
    stINIT(15) = w3; 
    stINIT(16) = w4;
    
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
    end

    % % Update real-time 3D trajectory plot every few iterations
    % if mod(i, 5) == 0 || i == 1  % Update every 5 iterations to avoid too frequent updates
    %     % Add current position to trajectory
    %     traj_x = [traj_x, x_current];
    %     traj_y = [traj_y, y_current];
    %     traj_z = [traj_z, -z_current];
    % 
    %     % Update trajectory line
    %     set(h_traj, 'XData', traj_x, 'YData', traj_y, 'ZData', traj_z);
    % 
    %     % Update current position marker
    %     set(h_current, 'XData', x_current, 'YData', y_current, 'ZData', -z_current);
    % 
    %     % Update target position if it has changed
    %     set(h_target, 'XData', X_des, 'YData', Y_des, 'ZData', -Z_des);
    % 
    %     % Update axis limits to follow the trajectory
    %     if ~isempty(traj_x)
    %         x_range = [min([traj_x, X_des]) - 2, max([traj_x, X_des]) + 2];
    %         y_range = [min([traj_y, Y_des]) - 2, max([traj_y, Y_des]) + 2];
    %         z_range = [min([traj_z, -Z_des]) - 2, max([traj_z, -Z_des]) + 2];
    %         axis([x_range, y_range, z_range]);
    %     end
        
        % Force the plot to update
    %     drawnow limitrate;  % limitrate prevents too frequent updates
    % end

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

% Trim plotting arrays
actual_iterations = min(i-1, max_iterations);
Phi_des_arr = Phi_des_arr(1:actual_iterations);
Theta_des_arr = Theta_des_arr(1:actual_iterations);
Phi_actual_arr = Phi_actual_arr(1:actual_iterations);
Theta_actual_arr = Theta_actual_arr(1:actual_iterations);
X_error_arr = X_error_arr(1:actual_iterations);
Y_error_arr = Y_error_arr(1:actual_iterations);

fprintf('Position control simulation complete. Using fixed gains: Kp_ang=%.3f, Ki_ang=%.3f, Kd_ang=%.3f\n', Kp_ang, Ki_ang, Kd_ang);
%% PLOT RESULTS
% Note: Real-time 3D trajectory plot is already displayed above
% The following plots show detailed analysis of the simulation results

% Create consistent time array for control data
time_control = linspace(0, all_times(end), actual_iterations);

figure(4);

subplot(4,1,1);
if ~isempty(all_states) && size(all_states, 1) > 0
    invZ = (-1.*all_states(:,3));
    plot(all_times,invZ, 'b-', 'LineWidth', 1.5); hold on;
    plot([0 all_times(end)], [-Z_des -Z_des], 'r--', 'LineWidth', 1.5); hold off;
    minScale = 1;  % Adjust this value according to your desired minimum scale
    ylim([min([invZ; -Z_des])-minScale, max([invZ; -Z_des])+minScale]);
    legend('Actual', 'Desired', 'Location', 'best');
else
    plot([0 1], [0 0], 'b-'); % Dummy plot
    title('Position: Z - No Data Available');
end
title('Position: Z');
ylabel('m');

subplot(4,1,2);
if ~isempty(all_states) && size(all_states, 1) > 0
    plot(all_times,rad2deg(all_states(:,6)))
    minScale = 1;  % Adjust this value according to your desired minimum scale
    ylim([min(rad2deg(all_states(:,6)))-minScale, max(rad2deg(all_states(:,6)))+minScale]);
else
    plot([0 1], [0 0], 'b-'); % Dummy plot
    title('Angle: Psi - No Data Available');
end
title('Angle: Psi');
ylabel('deg');



subplot(4,1,3);
if ~isempty(all_states) && size(all_states, 1) > 0
    invW = (-1.*all_states(:,9));
    plot(all_times,invW)
    minScale = 1;  % Adjust this value according to your desired minimum scale
    ylim([min(invW)-minScale, max(invW)+minScale]);
else
    plot([0 1], [0 0], 'b-'); % Dummy plot
    title('Velocity(z): W - No Data Available');
end
title('Velocity(z): W');
ylabel('m/s');

subplot(4,1,4);
if ~isempty(all_states) && size(all_states, 1) > 0
    plot(all_times,all_states(:,12))
    minScale = 1;  % Adjust this value according to your desired minimum scale
    ylim([min(all_states(:,12))-minScale, max(all_states(:,12))+minScale]);
else
    plot([0 1], [0 0], 'b-'); % Dummy plot
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

    %plot3(states(:,1),states(:,2),states(:,3));
    scatter3(all_states(:,1),all_states(:,2),-1.*all_states(:,3),10,colorMap); hold on;
    % Plot desired position as a red star
    scatter3(X_des, Y_des, -Z_des, 200, 'r*', 'LineWidth', 3); hold off;

    minScale = 1;  % Adjust this value according to your desired minimum scale
    xlim([min(all_states(:,1))-minScale, max(all_states(:,1))+minScale]);
    ylim([min(all_states(:,2))-minScale, max(all_states(:,2))+minScale]);
    zlim([min(-1.*all_states(:,3))-minScale, max(-1.*all_states(:,3))+minScale]);
    grid on;
    title('3D Trajectory (Blue to Red) with Desired Position');
    legend('Trajectory', 'Desired Position', 'Location', 'best');
else
    % Create empty 3D plot
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

% Motor speeds plot (like Z-only version)
figure(7); % Use figure 7 for motor speeds
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


figure(1);

subplot(4,1,1);
if ~isempty(all_states) && size(all_states, 1) > 0
    plot(all_times,all_states(:,1), 'b-', 'LineWidth', 1.5); hold on;
    plot([0 all_times(end)], [X_des X_des], 'r--', 'LineWidth', 1.5); % Target position
    hold off;
    minScale = 1;
    ylim([min([all_states(:,1); X_des])-minScale, max([all_states(:,1); X_des])+minScale]);
    legend('Actual', 'Target', 'Location', 'best');
else
    plot([0 1], [X_des X_des], 'r--', 'LineWidth', 1.5);
    title('X Position - No Data Available');
    legend('Target', 'Location', 'best');
end
title('X Position (Blue=Actual, Red=Target)');
ylabel('m');

subplot(4,1,2);
if ~isempty(all_states) && size(all_states, 1) > 0
    plot(all_times,all_states(:,2), 'b-', 'LineWidth', 1.5); hold on;
    plot([0 all_times(end)], [Y_des Y_des], 'r--', 'LineWidth', 1.5); % Target position
    hold off;
    minScale = 1;
    ylim([min([all_states(:,2); Y_des])-minScale, max([all_states(:,2); Y_des])+minScale]);
    legend('Actual', 'Target', 'Location', 'best');
else
    plot([0 1], [Y_des Y_des], 'r--', 'LineWidth', 1.5);
    title('Y Position - No Data Available');
    legend('Target', 'Location', 'best');
end
title('Y Position');
ylabel('m');

subplot(4,1,3);
if ~isempty(all_states) && size(all_states, 1) > 0
    plot(all_times,rad2deg(all_states(:,4)), 'b-', 'LineWidth', 1.5); hold on;
    if length(Phi_des_arr) > 0 && length(time_control) == length(Phi_des_arr)
        plot(time_control, rad2deg(Phi_des_arr), 'r--', 'LineWidth', 1.5);
        legend('Actual', 'Desired', 'Location', 'best');
    else
        legend('Actual', 'Location', 'best');
    end
    hold off;
    minScale = 5;  % Adjust this value according to your desired minimum scale
    ylim([min(rad2deg(all_states(:,4)))-minScale, max(rad2deg(all_states(:,4)))+minScale]);
else
    plot([0 1], [0 0], 'b-');
    title('Roll Angle (φ) - No Data Available');
end
title('Roll Angle (φ)');
ylabel('deg');

subplot(4,1,4);
if ~isempty(all_states) && size(all_states, 1) > 0
    plot(all_times,rad2deg(all_states(:,5)), 'b-', 'LineWidth', 1.5); hold on;
    if length(Theta_des_arr) > 0 && length(time_control) == length(Theta_des_arr)
        plot(time_control, rad2deg(Theta_des_arr), 'r--', 'LineWidth', 1.5);
        legend('Actual', 'Desired', 'Location', 'best');
    else
        legend('Actual', 'Location', 'best');
    end
    hold off;
    minScale = 5;  % Adjust this value according to your desired minimum scale
    ylim([min(rad2deg(all_states(:,5)))-minScale, max(rad2deg(all_states(:,5)))+minScale]);
else
    plot([0 1], [0 0], 'b-');
    title('Pitch Angle (θ) - No Data Available');
end
title('Pitch Angle (θ)');
ylabel('deg');

% Add figure for autotuning progress - Simple plotting only
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



%%

xarr = states(:,1);
yarr = states(:,2);
zarr = -1.*states(:,3);

Motor_Speeds = [];

%%
function st_dot = quad_dynamics(t,states)

%     sg = @(x) 1/(1 + exp(1)^(-x));
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
    % HAA = + MF(2)*l*sind(alpha)
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