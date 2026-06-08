function Xdot = drone_rhs_compensated(~, X, M, G1, B, A4)
% Enhanced version with full w_des_dot compensation
% This compensates for T(phi,theta)*w term to achieve exact e_dot = M*e

% State split
w     = X(1:3);     % [p; q; r]
theta = X(4:6);     % [phi; theta; psi]

phi   = theta(1);
theta_ang = theta(2);

r = w(3);  % Current yaw rate (state variable)

% ============================================================
% 1. A1 MATRIX - Depends on current yaw rate r
% ============================================================
A1 = [-0.03,  r, 0;
       -r, -0.03, 0;
       0, 0, -0.1];

% ============================================================
% 2. COMPUTE w_des(phi, theta, r) - State-dependent desired angular velocity
% ============================================================
w_des = [-phi;
         -theta_ang;
          9];

% ============================================================
% 3. COMPUTE EULER KINEMATICS MATRIX T(phi,theta)
% ============================================================
theta_max = 1.5;  % ~86 degrees
theta_sat = max(-theta_max, min(theta_max, theta_ang));

% Protected trig functions
cos_theta_safe = cos(theta_sat);
if abs(cos_theta_safe) < 1e-6
    cos_theta_safe = sign(cos_theta_safe + 1e-10) * 1e-6;
end

T = [1  sin(phi)*tan(theta_sat)    cos(phi)*tan(theta_sat);
     0  cos(phi)                   -sin(phi);
     0  sin(phi)/cos_theta_safe    cos(phi)/cos_theta_safe];

% ============================================================
% 4. COMPUTE w_des_dot using kinematics
%    w_des_dot = d/dt[-phi; -theta; r]
%              = -[phi_dot; theta_dot; r_dot]
%              = -T(phi,theta)*w + [0; 0; r_dot]
% 
%    For the compensation term, we approximate r_dot ≈ 0 (or could use
%    r_dot from the dynamics if needed for higher accuracy)
% ============================================================
% Option 1: Assume r_dot ≈ 0 for compensation (simple)
r_dot_approx = 0;

% Compute w_des_dot

w_des_dot = -T * w .* [1;1;0];

% ============================================================
% 5. ENHANCED c4 WITH FEEDFORWARD COMPENSATION
%    Original: c4 = -(M*w_des + A4)/B
%    Enhanced: c4 = -(M*w_des + A4 - w_des_dot)/B
%
%    This makes the closed-loop satisfy:
%    w_dot = w_des_dot + M*(w - w_des)
%
%    Equivalently: e_dot = w_dot - w_des_dot = M*e (exact!)
% ============================================================
c4_compensated = (-(M*w_des + A4 - w_des_dot))/B;

% ============================================================
% 6. REBUILD C MATRIX WITH COMPENSATION
% ============================================================
C = [G1, c4_compensated];

% ============================================================
% 7. FORM FULL A + BC MATRIX
% ============================================================
A  = [A1 A4];
Acl = A + B*C;

% ============================================================
% 8. OMEGA DYNAMICS
% ============================================================
X_aug = [w; 1];
Xdot_aug = Acl * X_aug;
wdot = Xdot_aug(1:3);

% ============================================================
% 9. EULER KINEMATICS
% ============================================================
thetadot = T * w;

% ============================================================
% 10. ASSEMBLE STATE DERIVATIVE
% ============================================================
Xdot = [wdot; thetadot];

end
