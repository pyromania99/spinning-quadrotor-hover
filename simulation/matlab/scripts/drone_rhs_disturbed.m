function Xdot = drone_rhs_disturbed(~, X, M, G1, B, A4, delta_A)
% drone_rhs_disturbed - Dynamics with disturbance in A matrix
%
% Same as drone_rhs but with additive disturbance delta_A on the A1 matrix
% This simulates model uncertainty or external disturbances affecting dynamics

% State split
w     = X(1:3);     % [p; q; r]
theta = X(4:6);     % [phi; theta; psi]

phi   = theta(1);
theta_ang = theta(2);
% psi   = theta(3);  % Not needed for w_des calculation

r = w(3);  % Current yaw rate (state variable)

% ============================================================
% 1. A1 MATRIX WITH DISTURBANCE
%    A1_disturbed = A1_nominal + delta_A
% ============================================================
A1_nominal = [-1,  r,  0;
              -r, -1,  0;
               0,  0, -2];

% Add disturbance to A1 matrix
A1 = A1_nominal + delta_A;

% ============================================================
% 2. COMPUTE w_des(phi, theta, r) - State-dependent desired angular velocity
%    w_des = [-tan(theta)/cos(phi)*r; tan(phi)*r; r]
% ============================================================
% Note: Using same w_des formula as nominal (controller doesn't know about disturbance)
eps_val = 1e-6;
cos_phi_safe = cos(phi) + eps_val * sign(cos(phi) + eps_val);

w_des = [ -phi;
          -theta_ang;
          9 ];

% ============================================================
% 3. UPDATE c4 FROM YOUR EXACT FORMULA
% ============================================================
c4 = (-(M*w_des + A4))/B;

% ============================================================
% 4. REBUILD C MATRIX
% ============================================================
C = [G1 c4];

% ============================================================
% 5. FORM FULL A + BC MATRIX (with disturbed A)
% ============================================================
A  = [A1 A4];
Acl = A + B*C;

% ============================================================
% 6. OMEGA DYNAMICS
% ============================================================
X_aug = [w; 1];
Xdot_aug = Acl * X_aug;
wdot = Xdot_aug(1:3);

% ============================================================
% 7. EULER KINEMATICS MATRIX T(phi,theta)
%    Saturate theta_ang to avoid gimbal lock singularity at ±90°
% ============================================================
theta_max = 1.5;  % ~86 degrees, stay away from 90° singularity
theta_sat = max(-theta_max, min(theta_max, theta_ang));

% Protected trig functions
cos_theta_safe = cos(theta_sat);
if abs(cos_theta_safe) < 1e-6
    cos_theta_safe = sign(cos_theta_safe + 1e-10) * 1e-6;
end

T = [1  sin(phi)*tan(theta_sat)    cos(phi)*tan(theta_sat);
     0  cos(phi)                   -sin(phi);
     0  sin(phi)/cos_theta_safe    cos(phi)/cos_theta_safe];

thetadot = T * w;

% ============================================================
% 8. ASSEMBLE STATE DERIVATIVE
% ============================================================
Xdot = [wdot; thetadot];

end
