clear;
clc;
close all;

% Add paths if needed
addpath('c:\Users\anike\Desktop\RESEARCH\Matlab');

% Run 1: Fade time = 10
fade_time_1 = 10;
global fade_time;
fade_time = fade_time_1;
run('change_tilt.m');
results_1.all_states = all_states;
results_1.all_times = all_times;
results_1.motor_speed = motor_speed;

% Run 2: Fade time = 1000
fade_time_2 = 1000;
fade_time = fade_time_2;
run('change_tilt.m');
results_2.all_states = all_states;
results_2.all_times = all_times;
results_2.motor_speed = motor_speed;

% Extract results for plotting
all_states_1 = results_1.all_states;
all_times_1 = results_1.all_times;
motor_speed_1 = results_1.motor_speed;

all_states_2 = results_2.all_states;
all_times_2 = results_2.all_times;
motor_speed_2 = results_2.motor_speed;

% Plot comparison results
figure('Name', 'Comparison of Change Tilt Runs', 'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);

% 1. Trajectories (3D)
subplot(2,3,1);
plot3(all_states_1(:,1), all_states_1(:,2), -all_states_1(:,3), 'b-', 'LineWidth', 1.5); hold on;
plot3(all_states_2(:,1), all_states_2(:,2), -all_states_2(:,3), 'r--', 'LineWidth', 1.5);
scatter3(0, 0, -Z_des, 200, 'k*', 'LineWidth', 2); % Desired position
hold off;
grid on;
title('3D Trajectories');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
legend(sprintf('Fade Time = %d', fade_time_1), sprintf('Fade Time = %d', fade_time_2), 'Desired', 'Location', 'best');

% 2. World Attitudes
subplot(2,3,2);
phi_world_1 = rad2deg(all_states_1(:,4));
phi_world_2 = rad2deg(all_states_2(:,4));
theta_world_1 = rad2deg(all_states_1(:,5));
theta_world_2 = rad2deg(all_states_2(:,5));
plot(all_times_1, phi_world_1, 'b-', 'LineWidth', 1.5); hold on;
plot(all_times_2, phi_world_2, 'r--', 'LineWidth', 1.5);
plot(all_times_1, theta_world_1, 'b-.', 'LineWidth', 1.5);
plot(all_times_2, theta_world_2, 'r:', 'LineWidth', 1.5);
hold off;
grid on;
title('World Attitudes');
xlabel('Time (s)'); ylabel('Angle (deg)');
legend(sprintf('Roll (Fade = %d)', fade_time_1), sprintf('Roll (Fade = %d)', fade_time_2), ...
       sprintf('Pitch (Fade = %d)', fade_time_1), sprintf('Pitch (Fade = %d)', fade_time_2), 'Location', 'best');

% 3. X, Y, Z Positions
subplot(2,3,3);
plot(all_times_1, all_states_1(:,1), 'b-', 'LineWidth', 1.5); hold on;
plot(all_times_2, all_states_2(:,1), 'r--', 'LineWidth', 1.5);
plot(all_times_1, all_states_1(:,2), 'b-.', 'LineWidth', 1.5);
plot(all_times_2, all_states_2(:,2), 'r:', 'LineWidth', 1.5);
plot(all_times_1, -all_states_1(:,3), 'b:', 'LineWidth', 1.5);
plot(all_times_2, -all_states_2(:,3), 'r-.', 'LineWidth', 1.5);
hold off;
grid on;
title('X, Y, Z Positions');
xlabel('Time (s)'); ylabel('Position (m)');
legend(sprintf('X (Fade = %d)', fade_time_1), sprintf('X (Fade = %d)', fade_time_2), ...
       sprintf('Y (Fade = %d)', fade_time_1), sprintf('Y (Fade = %d)', fade_time_2), ...
       sprintf('Z (Fade = %d)', fade_time_1), sprintf('Z (Fade = %d)', fade_time_2), 'Location', 'best');

% 4. Motor Differences (Fade Time = 10)
figure('Name', 'Motor Differences (Fade Time = 10)', 'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);
for i = 1:4
    subplot(2,2,i);
    plot(time_des_arr, motor_speed_1(i,:) - mean(motor_speed_1, 1), 'b-', 'LineWidth', 1.5);
    title(sprintf('Motor %d Speed Difference', i));
    xlabel('Time (s)'); ylabel('Speed Diff (rad/s)');
    grid on;
end

% 5. Motor Differences (Fade Time = 1000)
figure('Name', 'Motor Differences (Fade Time = 1000)', 'Units', 'normalized', 'Position', [0.05 0.05 0.9 0.85]);
for i = 1:4
    subplot(2,2,i);
    plot(time_des_arr, motor_speed_2(i,:) - mean(motor_speed_2, 1), 'r--', 'LineWidth', 1.5);
    title(sprintf('Motor %d Speed Difference', i));
    xlabel('Time (s)'); ylabel('Speed Diff (rad/s)');
    grid on;
end
