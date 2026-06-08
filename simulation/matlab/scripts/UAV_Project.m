clear;
clc;
close all;

global m g alpha time_arr psiarr ttarr
time_arr = [];
m = 2;
g = 9.81;
Ct = 0.0027;
Kt = Ct * 1.225 * 0.104^4; % x rho x D^4
Kd = 1.58e-4;

Cp = 1.57e-4;

% w_init = sqrt((m*g)/(4*Kt));

4*MotorForce(4000,4.1,4.1,0,0)/(m*g)

MotorForce(4000,4.1,4.1,0,0)

% w_init = 656.68;
w_init = 3199.5;

frequency = 50; %Controller Frequency in Hertz
tspan = [0 1/frequency];
stINIT = [ 0, 0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 , w_init, w_init, w_init, w_init];
%stIN = [x, y, z, phi, theta, psi, u, v, w, p, q, r, Mphi, MTheta, MPsi, Force_z];
options = odeset('RelTol',1e-3,'AbsTol',1e-5);

% Initialize empty arrays for storing states and time values
all_states = [];
all_times = [];
timePeriod = 60;
% Initialize position error and derivative arrays for X, Y, and Z
X_err = [0]; X_dif = [0];
Y_err = [0]; Y_dif = [0];
Z_err = [0]; Z_dif = [0];

% Desired positions (X_des, Y_des, Z_des) over time - define these based on your trajectory
X_des = 0;
Y_des = 0;
Z_des = 0;

% PID Coefficients for position control
Kp_pos = 0.15; Ki_pos = 0.001; Kd_pos = 10;
Kp_Z = 50;    Ki_Z = 0.25;   Kd_Z = 6000;
% Kp_Z = 500;    Ki_Z = 0;   Kd_Z = 5000;

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
psiarr = [];
ttarr = [];
% 4*MotorForce(656.68,10,4.5,0,0) - m*g
motor_speed = [];
while i < (timePeriod * frequency)
    alpha = 90*(1/(1 + exp(-4*(tspan(1) - 5))));
    % alpha = 90;
    [t, states] = ode45(@quad_dynamics, tspan, stINIT);

    % Append new states and time values for plotting
    all_states = [all_states; states];
    all_times = [all_times; t];
    
    % Update initial conditions for the next iteration
    stINIT = states(end,:);
    
    err_Z = (states(end,3) - Z_des);
    diff_err_Z = 0;    
    sum_err_Z = sum(all_states(:,3));
  

    if i > 1
        diff_err_Z =  (err_Z - (states(end-1,3) - Z_des))/(1/frequency);
    end
    
    ctrl_adj = Kp_Z*err_Z + 0.05*Ki_Z*sum_err_Z + Kd_Z*diff_err_Z;
    
    upper_limit = 6000;

    w_current = max(0, min(w_init + ctrl_adj, upper_limit));
    motor_speed = [motor_speed w_current];
    stINIT(13) = w_current * (1 - 1/(1 + exp(-4*(tspan(1) - 45))));
    stINIT(14) = w_current; 
    stINIT(15) = w_current; 
    stINIT(16) = w_current * (1 - 1/(1 + exp(-4*(tspan(1) - 25))));  

    tspan = tspan + 1/frequency;
    i = i + 1;
end
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
figure(1); % Create a new figure or reuse an existing figure ID
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
%%

xarr = states(:,1);
yarr = states(:,2);
zarr = -1.*states(:,3);

Motor_Speeds = [];

%%
function st_dot = quad_dynamics(t,states)

%     sg = @(x) 1/(1 + exp(1)^(-x));
    global m g alpha psiarr ttarr time_arr;
    Jr = 0;
    Ixx = 7.5e-3;
    Iyy = 7.5e-3;
    Izz = 2.3e-2;

    x = states(1);
    y = states(2);
    z = states(3);
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
    Cp = 1.57e-4;

    Pwr = 4*Cp*1.225*w1^3*0.104^5;

    l = 0.25; %m arm length
    tipV = l*r; % m/s
    J = tipV/(0.104*(w1 + 0.1)); 
     

    MF = [
        MotorForce(w1,4.1,4.1,tipV,alpha)
        MotorForce(w2,4.1,4.1,tipV,alpha)
        MotorForce(w3,4.1,4.1,tipV,alpha)
        MotorForce(w4,4.1,4.1,tipV,alpha)
    ];

    inv_J = max(0,(1-J));

    M_phi = MF(1)*l*cosd(alpha) + MF(2)*l*cosd(alpha) - MF(3)*l*cosd(alpha) - MF(4)*l*cosd(alpha);
    M_theta = -MF(1)*l*cosd(alpha) + MF(2)*l*cosd(alpha) - MF(3)*l*cosd(alpha) + MF(4)*l*cosd(alpha);
    % HAA = + MF(2)*l*sind(alpha)
    M_psi = -Kd*w1^2*cosd(alpha)*inv_J + MF(1)*l*sind(alpha) + Kd*w2^2*cosd(alpha)*inv_J + MF(2)*l*sind(alpha) + Kd*w3^2*cosd(alpha)*inv_J + MF(3)*l*sind(alpha) - Kd*w4^2*cosd(alpha)*inv_J + MF(4)*l*sind(alpha);
    psiarr = [psiarr M_psi];
    Tt = MF(1)*cosd(alpha) + MF(2)*cosd(alpha) + MF(3)*cosd(alpha) + MF(4)*cosd(alpha) + MotorForce(r,39,4.5,0,0);  
    ttarr = [ttarr Tt];
    time_arr = [time_arr t];
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

    alpha_R = 0.5;

    p_dot = ((Iyy - Izz)/Ixx)*q*r + (M_phi)/Ixx;
    q_dot = ((Izz - Ixx)/Iyy)*p*r + (M_theta)/Iyy ;
    r_dot = ((Ixx - Iyy)/Izz)*p*q + (M_psi)/Izz - alpha_R*r;

    st_dot = [xyz_dot(1,:);xyz_dot(2,:);xyz_dot(3,:);ptp_dot(1,:);ptp_dot(2,:);ptp_dot(3,:);uvw_dot(1,:);uvw_dot(2,:);uvw_dot(3,:);p_dot;q_dot;r_dot;0;0;0;0];
end

function mf = MotorForce(w,d,pitch,V0,aoa)
    mf = 4.392399e-8*(w*60/(2*pi))*((d^3.5)/sqrt(pitch))*((4.23333e-4 * (w*60/(2*pi)) * pitch) - V0*sind(aoa));
end