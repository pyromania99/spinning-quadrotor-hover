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
Cp = 1.57e-4;

% w_init = sqrt((m*g)/(4*Kt));

4*MotorForce(4000,4.1,4.1,0,0)/(m*g)

MotorForce(4000,4.1,4.1,0,0)

% w_init = 656.68;
w_init = 3199.5;

frequency = 100; %Controller Frequency in Hertz (increased for speed)
tspan = [0 1/frequency];
stINIT = [ 0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 , w_init, w_init, w_init, w_init];
%stIN = [x, y, z, phi, theta, psi, u, v, w, p, q, r, Mphi, MTheta, MPsi, Force_z];
options = odeset('RelTol',1e-4,'AbsTol',1e-6); % Faster solver settings
timePeriod = 30;

% Initialize empty arrays for storing states and time values (pre-allocate for speed)
max_iterations = timePeriod * frequency;
all_states = zeros(max_iterations * 50, 16); % Pre-allocate with extra space
all_times = zeros(max_iterations * 50, 1);
states_counter = 1;
i = 1;
% Initialize position error and derivative arrays for X, Y, and Z
X_err = [0]; X_dif = [0];
Y_err = [0]; Y_dif = [0];
Z_err = [0]; Z_dif = [0];

% Desired positions (X_des, Y_des, Z_des) over time - define these based on your trajectory
X_des = 0;
Y_des = 0;
Z_des = -10;

% PID Coefficients for position control
Kp_pos = 0.15; Ki_pos = 0.00; Kd_pos = 10;
Kp_Z = 250;    Ki_Z = 0;   Kd_Z = 5000;

% PID Coefficients for angle control (as before)
Kp_ang = 0.4; Ki_ang = 0; Kd_ang = 5;


% Initialize angle error and derivative arrays for Phi and Theta
Phi_err = [0]; Phi_dif = [0];
Theta_err = [0]; Theta_dif = [0];

% Assume desired Psi (yaw angle) and initialize its error and derivative arrays
Psi_des = 0; % Example: keeping yaw constant
Psi_err = [0]; Psi_dif = [0];

% Define maximum allowable angles (in radians)
phi_max = deg2rad(60); % Maximum roll angle limit
theta_max = deg2rad(60); % Maximum pitch angle limit

i = 1;
X_traj = [];
Y_traj = [];
T_des_arr = [];
Phi_des_arr = [];
Theta_des_arr = [];
% Pre-allocate arrays for better performance
global psiarr ttarr
psiarr = zeros(1, max_iterations * 50);
ttarr = zeros(1, max_iterations * 50);
motor_speed = zeros(1, max_iterations);

fprintf('Starting Z-only control simulation...\n');

while i < (timePeriod * frequency)
    current_time = tspan(1);
    alpha = 30*(1/(1 + exp(-4*(current_time - 60))));
    
    % Remove print statements for speed - only print every 5 seconds
    if mod(i, frequency*5) == 0
        fprintf('Time: %.1f seconds\n', current_time);
    end
    
    if current_time > 59
        Kp_Z = 700;    Ki_Z = 0;   Kd_Z = 1000;
    end
    [t, states] = ode45(@quad_dynamics, tspan, stINIT, options);

    % Store states more efficiently
    new_data_points = length(t);
    all_states(states_counter:states_counter+new_data_points-1, :) = states;
    all_times(states_counter:states_counter+new_data_points-1) = t;
    states_counter = states_counter + new_data_points;
    
    % Update initial conditions for the next iteration
    stINIT = states(end,:);
    
    err_Z = (states(end,3) - Z_des);
    diff_err_Z = 0;    
    sum_err_Z = sum(all_states(max(1,states_counter-41):states_counter-1,3)); % More efficient indexing

    if i > 1
        diff_err_Z =  (err_Z - (states(end-1,3) - Z_des))/(1/frequency);
    end
    
    ctrl_adj = Kp_Z*err_Z + Ki_Z*sum_err_Z + Kd_Z*diff_err_Z;
    
    upper_limit = 6000;

    w_current = max(0, min(w_init + ctrl_adj, upper_limit));
    if i <= max_iterations
        motor_speed(i) = w_current;
    end
    stINIT(13) = w_current * (1 - 1/(1 + exp(-4*(current_time - 10))));
    stINIT(14) = w_current; 
    stINIT(15) = w_current; 
    stINIT(16) = w_current;% * (1 - 1/(1 + exp(-4*(tspan(1) - 5))));  

    tspan = tspan + 1/frequency;
    i = i + 1;
end

% Trim pre-allocated arrays to actual size
all_states = all_states(1:states_counter-1, :);
all_times = all_times(1:states_counter-1);
psiarr = psiarr(1:psi_counter-1);
ttarr = ttarr(1:psi_counter-1);

fprintf('Z-only simulation complete.\n');
%% PLOT RESULTS

t_arr = 0:1:(frequency*timePeriod)-2;

figure(4);

subplot(4,1,1);
invZ = (-1.*all_states(:,3));
plot(all_times,invZ)
minScale = 1;  % Adjust this value according to your desired minimum scale
ylim([min(invZ)-minScale, max(invZ)+minScale]);
title('Position: Z');
ylabel('m');

subplot(4,1,2);
plot(all_times,rad2deg(all_states(:,6)))
minScale = 1;  % Adjust this value according to your desired minimum scale
ylim([min(rad2deg(all_states(:,6)))-minScale, max(rad2deg(all_states(:,6)))+minScale]);
title('Angle: Psi');
ylabel('deg');



subplot(4,1,3);
invW = (-1.*all_states(:,9));
plot(all_times,invW)
minScale = 1;  % Adjust this value according to your desired minimum scale
ylim([min(invW)-minScale, max(invW)+minScale]);
title('Velocity(z): W');
ylabel('m/s');

subplot(4,1,4);
plot(all_times,all_states(:,12))
minScale = 1;  % Adjust this value according to your desired minimum scale
ylim([min(all_states(:,12))-minScale, max(all_states(:,12))+minScale]);
title('Angular Rate: R');
ylabel('rad/s');

figure(2);

startColor = [0, 0, 1];     % Start color (blue)
endColor = [1, 0, 0];       % End color (red)
numPoints = numel(all_states(:,1));
colorMap = [linspace(startColor(1), endColor(1), numPoints)', ...
            linspace(startColor(2), endColor(2), numPoints)', ...
            linspace(startColor(3), endColor(3), numPoints)'];

%plot3(states(:,1),states(:,2),states(:,3));
scatter3(all_states(:,1),all_states(:,2),-1.*all_states(:,3),10,colorMap)

minScale = 1;  % Adjust this value according to your desired minimum scale
xlim([min(all_states(:,1))-minScale, max(all_states(:,1))+minScale]);
ylim([min(all_states(:,2))-minScale, max(all_states(:,2))+minScale]);
zlim([min(-1.*all_states(:,3))-minScale, max(-1.*all_states(:,3))+minScale]);
grid on;
title('3D Trajectory (Blue to Red)');

figure(3);
plot(time_arr,ttarr);
% % figure();
% Add a subplot for motor speeds
figure(5); % Create a new figure or reuse an existing figure ID
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


figure(1);

subplot(4,1,1);
invZ = (-1.*all_states(:,1));
plot(all_times,invZ)
minScale = 1;  % Adjust this value according to your desired minimum scale
ylim([min(invZ)-minScale, max(invZ)+minScale]);
title('X');
ylabel('m');

subplot(4,1,2);
plot(all_times,rad2deg(all_states(:,2)))
minScale = 1;  % Adjust this value according to your desired minimum scale
ylim([min(rad2deg(all_states(:,6)))-minScale, max(rad2deg(all_states(:,6)))+minScale]);
title('Y');
ylabel('m');

subplot(4,1,3);
invW = (-1.*all_states(:,4));
plot(all_times,invW)
minScale = 1;  % Adjust this value according to your desired minimum scale
ylim([min(invW)-minScale, max(invW)+minScale]);
title(' phi');
ylabel('rad');

subplot(4,1,4);
plot(all_times,all_states(:,5))
minScale = 1;  % Adjust this value according to your desired minimum scale
ylim([min(all_states(:,12))-minScale, max(all_states(:,12))+minScale]);
title('theta');
ylabel('rad');



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