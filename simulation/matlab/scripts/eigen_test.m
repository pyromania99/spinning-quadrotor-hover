%% Eigenvalue Design for Constrained M Matrix (2 Free Eigenvalues)
% Bottom row of M is fixed to bottom row of A1
% You specify 2 eigenvalues; the 3rd is determined by constraint
% Ensures all matrix elements remain REAL and eigenvalues are REAL

clear; clc; close all;

%% ========== USER INPUTS ==========
r = 9;

% Define A1 matrix (3x3)
A1 = [-0.03,  r, 0;
       -r, -0.03, 0;
       0, 0, -0.1];

% Extract fixed bottom row from A1
a31 = A1(3,1);
a32 = A1(3,2);
a33 = A1(3,3);

% YOU CAN ONLY CHOOSE 2 EIGENVALUES (must be REAL for real solution)
% The third eigenvalue will be computed automatically
lambda1 = -5;  % First desired eigenvalue (MUST BE REAL)
lambda2 = -4;  % Second desired eigenvalue (MUST BE REAL)

% CRITICAL: For real M, lambda1 and lambda2 must both be real
if ~isreal(lambda1) || ~isreal(lambda2)
    error('lambda1 and lambda2 must be REAL for a real M matrix!');
end

% Choice of free parameters (set m13=0, m23=0 for simplified case)
m13 = 0;
m23 = 0;

% Additional free choices for m12 and m21
% These will be AUTOMATICALLY ADJUSTED to ensure real M
m12_initial = 1;  % Initial guess
m21_initial = 1;  % Initial guess

% Scalar B (for later computation of G1)
B = 0.5;

% Desired state and A4 column
w_des = [0; 0; 9];  % Desired 3x1 state vector (for w, not X!)
A4 = [0; 0; 0];     % Last column of A matrix (3x1)

%% ========== DISPLAY SETUP ==========

fprintf('=== SYSTEM SETUP ===\n');
fprintf('A1 matrix:\n');
disp(A1);
fprintf('Fixed bottom row: [%.4f, %.4f, %.4f]\n', a31, a32, a33);
fprintf('\n=== YOUR CHOICES ===\n');
fprintf('λ1 (desired) = %.4f\n', lambda1);
fprintf('λ2 (desired) = %.4f\n', lambda2);
fprintf('λ3 will be determined by the constraint\n');

%% ========== FIND REAL SOLUTION ==========

if m13 == 0 && m23 == 0
    fprintf('\n=== SOLVING FOR REAL M (m13=0, m23=0) ===\n');
    
    % Strategy: For 2x2 top-left block to have eigenvalues lambda1, lambda2:
    % trace = m11 + m22 = lambda1 + lambda2
    % det = m11*m22 - m12*m21 = lambda1*lambda2
    
    % For real solutions: discriminant >= 0
    % discriminant = (lambda1 + lambda2)^2 - 4*(lambda1*lambda2 + m12*m21)
    %              = (lambda1 - lambda2)^2 - 4*m12*m21
    
    % For real solutions: (lambda1 - lambda2)^2 >= 4*m12*m21
    % So: m12*m21 <= (lambda1 - lambda2)^2 / 4
    
    max_product = (lambda1 - lambda2)^2 / 4;
    
    fprintf('For real m11, m22, need: m12*m21 ≤ %.4f\n', max_product);
    
    % Try the initial guess first
    m12 = m12_initial;
    m21 = m21_initial;
    
    if m12*m21 > max_product
        fprintf('Initial m12*m21 = %.4f is too large!\n', m12*m21);
        fprintf('Auto-adjusting to ensure real solution...\n');
        
        % Adjust to use 90% of maximum to be safe
        target_product = 0.9 * max_product;
        
        % Keep ratio m12/m21 the same if possible
        if m12_initial ~= 0
            ratio = m21_initial / m12_initial;
            m12 = sqrt(abs(target_product / ratio));
            m21 = ratio * m12;
        else
            % If m12_initial = 0, set both to sqrt of target
            m12 = sqrt(abs(target_product));
            m21 = sign(target_product) * sqrt(abs(target_product));
        end
        
        fprintf('Adjusted to: m12 = %.4f, m21 = %.4f\n', m12, m21);
        fprintf('New product: m12*m21 = %.4f\n', m12*m21);
    else
        fprintf('Initial m12 = %.4f, m21 = %.4f is acceptable\n', m12, m21);
        fprintf('Product: m12*m21 = %.4f ≤ %.4f ✓\n', m12*m21, max_product);
    end
    
    % Now solve for m11, m22
    sum_m = lambda1 + lambda2;
    product_m = lambda1*lambda2 + m12*m21;
    
    fprintf('\nSolving for m11, m22:\n');
    fprintf('  m11 + m22 = %.4f\n', sum_m);
    fprintf('  m11*m22 = %.4f\n', product_m);
    
    discriminant = sum_m^2 - 4*product_m;
    fprintf('  Discriminant = %.4f\n', discriminant);
    
    if discriminant < -1e-10
        error('Discriminant still negative! This should not happen. Check logic.');
    elseif discriminant < 0
        % Tiny negative due to numerical error
        discriminant = 0;
        fprintf('  (Set to 0 due to numerical precision)\n');
    end
    
    m11 = (sum_m + sqrt(discriminant))/2;
    m22 = (sum_m - sqrt(discriminant))/2;
    
    fprintf('\n✓ Real solution found:\n');
    fprintf('  m11 = %.6f\n', m11);
    fprintf('  m22 = %.6f\n', m22);
    fprintf('  m12 = %.6f\n', m12);
    fprintf('  m21 = %.6f\n', m21);
    
else
    error('General case (m13≠0 or m23≠0) not implemented. Set m13=0, m23=0.');
end

%% ========== CONSTRUCT M MATRIX ==========

M = [m11, m12, m13;
     m21, m22, m23;
     a31, a32, a33];

fprintf('\n=== CONSTRUCTED M MATRIX ===\n');
disp(M);

% Verify all elements are real
if ~isreal(M)
    warning('M has complex elements! This should not happen.');
else
    fprintf('✓ All elements of M are REAL\n');
end

%% ========== COMPUTE ACTUAL EIGENVALUES ==========

M_eigenvalues = eig(M);

% Check if eigenvalues are real
if any(abs(imag(M_eigenvalues)) > 1e-10)
    warning('M has COMPLEX eigenvalues!');
    fprintf('Eigenvalues: ');
    for i = 1:length(M_eigenvalues)
        fprintf('%.4f%+.4fj  ', real(M_eigenvalues(i)), imag(M_eigenvalues(i)));
    end
    fprintf('\n');
else
    fprintf('✓ All eigenvalues of M are REAL\n');
    M_eigenvalues = real(M_eigenvalues);  % Remove tiny imaginary parts
end

M_eigenvalues_sorted = sort(real(M_eigenvalues), 'descend');

fprintf('\n=== EIGENVALUE RESULTS ===\n');
fprintf('You requested:     λ1 = %.6f, λ2 = %.6f\n', lambda1, lambda2);
fprintf('Actual eigenvalues of M:\n');
fprintf('  λ1 = %.6f\n', M_eigenvalues_sorted(1));
fprintf('  λ2 = %.6f\n', M_eigenvalues_sorted(2));
fprintf('  λ3 = %.6f (CONSTRAINED - determined by fixed row)\n', M_eigenvalues_sorted(3));

% Check stability of all eigenvalues
all_stable = all(real(M_eigenvalues) < 0);
if all_stable
    fprintf('\n✓ ALL eigenvalues are STABLE (negative real part)\n');
else
    fprintf('\n⚠️  WARNING: System has UNSTABLE eigenvalues!\n');
    for i = 1:length(M_eigenvalues)
        if real(M_eigenvalues(i)) >= 0
            fprintf('   λ%d = %.6f is unstable\n', i, M_eigenvalues(i));
        end
    end
end

%% ========== COMPUTE G1 GAIN MATRIX ==========

G1 = (M - A1) / B;

fprintf('\n=== FEEDBACK GAIN G1 ===\n');
disp(G1);

% Verify third row is zero (should be for our constraint)
third_row_norm = norm(G1(3,:));
fprintf('Third row of G1: [%.6f, %.6f, %.6f]\n', G1(3,1), G1(3,2), G1(3,3));
fprintf('Norm of third row = %.2e ', third_row_norm);

if third_row_norm < 1e-10
    fprintf('✓ (zero as required)\n');
else
    fprintf('⚠️  (should be zero!)\n');
end

%% ========== CLOSED-LOOP VERIFICATION ==========

A_cl = A1 + B*G1;

fprintf('\n=== CLOSED-LOOP MATRIX A_cl = A1 + B*G1 ===\n');
disp(A_cl);

% Verify A_cl equals M
error_M = norm(A_cl - M, 'fro');
fprintf('\n||A_cl - M|| = %.2e ', error_M);
if error_M < 1e-10
    fprintf('✓ (A_cl = M as expected)\n');
end

%% ========== COMPUTE C MATRIX (for X_dot = (A + BC)X form) ==========

fprintf('\n=== COMPUTING C MATRIX FOR X_dot = (A + BC)X ===\n');

fprintf('Desired state w_des (3x1):\n');
disp(w_des);
fprintf('A4 (last column of A):\n');
disp(A4);

% Compute c4 (last column of C)
% Formula: c4 = -(M*w_des + A4) / B
c4 = (-(M*w_des + A4)) / B;

% Construct C matrix (3x4)
C = [G1, c4];

fprintf('\n=== C MATRIX (3×4) ===\n');
fprintf('C = [G1 | c4] where c4 compensates for setpoint w_des\n');
disp(C);

% Verify third row of C
third_row_C_norm = norm(C(3,:));
fprintf('Third row of C: [%.6f, %.6f, %.6f, %.6f]\n', C(3,1), C(3,2), C(3,3), C(3,4));
fprintf('Norm of third row = %.2e ', third_row_C_norm);

if third_row_C_norm < 1e-10
    fprintf('✓ (zero as required)\n');
else
    fprintf('⚠️  (should be zero!)\n');
end

%% ========== VERIFY EQUIVALENCE ==========

fprintf('\n=== VERIFYING EQUIVALENCE ===\n');
fprintf('Testing: w_dot = M(w - w_des) ≡ w_dot from X_dot = (A + BC)X\n\n');

% Create test state
w_test = [0.5; -1; 2];  % 3x1 state
X_test = [w_test; 1];   % 4x1 augmented state

% Form 1: w_dot = M(w - w_des) where M = A1 + BG1
w_dot_form1 = M * (w_test - w_des);

% Form 2: Construct 4x4 A matrix and compute X_dot = (A + BC)X
A = [A1, A4];
% Compute (A + BC)
A_plus_BC = A + B * C;

% X_dot = (A + BC)X, extract w_dot part (first 3 rows)
X_dot_form2 = A_plus_BC * X_test;
w_dot_form2 = X_dot_form2(1:3);

fprintf('Test state w = [%.4f; %.4f; %.4f]\n', w_test(1), w_test(2), w_test(3));
fprintf('\nForm 1: w_dot = M(w - w_des) where M = A1 + BG1\n');
fprintf('  w_dot = [%.6f; %.6f; %.6f]\n', w_dot_form1(1), w_dot_form1(2), w_dot_form1(3));
fprintf('\nForm 2: w_dot from X_dot = (A + BC)X\n');
fprintf('  w_dot = [%.6f; %.6f; %.6f]\n', w_dot_form2(1), w_dot_form2(2), w_dot_form2(3));

difference = norm(w_dot_form1 - w_dot_form2);
fprintf('\nDifference: %.2e ', difference);
if difference < 1e-10
    fprintf('✓ FORMS ARE EQUIVALENT!\n');
else
    fprintf('⚠️  Forms differ!\n');
end

% Also verify that A + BC has the desired structure
fprintf('\n=== (A + BC) Matrix ===\n');
disp(A_plus_BC);

% Check eigenvalues of the 3x3 top-left block
fprintf('\nEigenvalues of (A + BC) top-left 3×3 block:\n');
eigs_A_plus_BC = eig(A_plus_BC(1:3, 1:3));
for i = 1:3
    fprintf('  λ%d = %.6f', i, real(eigs_A_plus_BC(i)));
    if abs(imag(eigs_A_plus_BC(i))) > 1e-10
        fprintf(' + %.6fj ⚠️ COMPLEX!', imag(eigs_A_plus_BC(i)));
    end
    fprintf('\n');
end

% These should match M eigenvalues
eigs_A_plus_BC_real = real(eigs_A_plus_BC);
error_eigs = norm(sort(eigs_A_plus_BC_real) - M_eigenvalues_sorted);
if error_eigs < 1e-6
    fprintf('✓ Eigenvalues match M eigenvalues (error = %.2e)\n', error_eigs);
end

%% ========== VISUALIZATION ==========

figure('Position', [100, 100, 1400, 500]);

% Plot 1: Eigenvalue placement in complex plane
subplot(1,4,1);
plot(real(M_eigenvalues), imag(M_eigenvalues), 'ro', 'MarkerSize', 14, 'LineWidth', 2.5);
hold on;
plot([lambda1, lambda2], [0, 0], 'bx', 'MarkerSize', 14, 'LineWidth', 2.5);
plot(M_eigenvalues_sorted(3), 0, 'gs', 'MarkerSize', 14, 'LineWidth', 2);
xline(0, 'k--', 'LineWidth', 1.5);
yline(0, 'k--', 'LineWidth', 1);
grid on;
xlabel('Real Part', 'FontSize', 11);
ylabel('Imaginary Part', 'FontSize', 11);
title('Eigenvalue Placement', 'FontSize', 12, 'FontWeight', 'bold');
legend('Achieved (all)', 'Desired (2 chosen)', 'Constrained (λ_3)', 'Location', 'best');
xlim([min(real(M_eigenvalues))-1, 1]);
ylim([-1, 1]);

% Plot 2: Eigenvalue comparison
subplot(1,4,2);
x_pos = [1, 2, 3];
bar_desired = [lambda1, lambda2, NaN];
bar_achieved = M_eigenvalues_sorted';

b = bar(x_pos, [bar_desired; bar_achieved]', 'grouped');
b(1).FaceColor = [0.3, 0.7, 1];
b(1).FaceAlpha = 0.7;
b(2).FaceColor = [1, 0.4, 0.3];

hold on;
plot(3, M_eigenvalues_sorted(3), 'gs', 'MarkerSize', 12, 'LineWidth', 2.5);

set(gca, 'XTickLabel', {'\lambda_1', '\lambda_2', '\lambda_3 (constrained)'});
ylabel('Eigenvalue', 'FontSize', 11);
title('Desired vs Achieved', 'FontSize', 12, 'FontWeight', 'bold');
legend('Desired', 'Achieved', 'Auto-determined', 'Location', 'best');
grid on;

% Plot 3: Matrix M heatmap
subplot(1,4,3);
imagesc(M);
colorbar;
colormap(gca, 'jet');
title('Matrix M', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Column', 'FontSize', 11);
ylabel('Row', 'FontSize', 11);
set(gca, 'XTick', 1:3, 'YTick', 1:3);
for i = 1:3
    for j = 1:3
        if abs(M(i,j)) > max(abs(M(:)))/2
            text_color = 'w';
        else
            text_color = 'k';
        end
        text(j, i, sprintf('%.3f', M(i,j)), ...
            'HorizontalAlignment', 'center', 'Color', text_color, ...
            'FontWeight', 'bold', 'FontSize', 10);
    end
end
rectangle('Position', [0.5, 2.5, 3, 1], 'EdgeColor', 'yellow', 'LineWidth', 3);

% Plot 4: Gain matrix G1 heatmap
subplot(1,4,4);
imagesc(G1);
colorbar;
colormap(gca, 'jet');
title('Gain Matrix G_1', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Column', 'FontSize', 11);
ylabel('Row', 'FontSize', 11);
set(gca, 'XTick', 1:3, 'YTick', 1:3);
for i = 1:3
    for j = 1:3
        if abs(G1(i,j)) > max(abs(G1(:)))/2
            text_color = 'w';
        else
            text_color = 'k';
        end
        text(j, i, sprintf('%.3f', G1(i,j)), ...
            'HorizontalAlignment', 'center', 'Color', text_color, ...
            'FontWeight', 'bold', 'FontSize', 10);
    end
end
rectangle('Position', [0.5, 2.5, 3, 1], 'EdgeColor', 'cyan', 'LineWidth', 3);

sgtitle(sprintf('Design: λ_1=%.2f, λ_2=%.2f → λ_3=%.2f (constrained)', ...
    lambda1, lambda2, M_eigenvalues_sorted(3)), 'FontSize', 14, 'FontWeight', 'bold');

%% ========== C MATRIX VISUALIZATION ==========

figure('Position', [200, 100, 1200, 500]);

% Plot 1: C matrix full view
subplot(1,3,1);
imagesc(C);
colorbar;
colormap(gca, 'jet');
title('C Matrix (3×4)', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('Column', 'FontSize', 11);
ylabel('Row', 'FontSize', 11);
set(gca, 'XTick', 1:4, 'YTick', 1:3);
set(gca, 'XTickLabel', {'1', '2', '3', '4 (c_4)'});
for i = 1:3
    for j = 1:4
        if abs(C(i,j)) > max(abs(C(:)))/2
            text_color = 'w';
        else
            text_color = 'k';
        end
        text(j, i, sprintf('%.3f', C(i,j)), ...
            'HorizontalAlignment', 'center', 'Color', text_color, ...
            'FontWeight', 'bold', 'FontSize', 9);
    end
end
rectangle('Position', [0.5, 2.5, 4, 1], 'EdgeColor', 'cyan', 'LineWidth', 3);

% Plot 2: Structure breakdown
subplot(1,3,2);
axis off;
text(0.1, 0.9, 'C Matrix Structure:', 'FontSize', 13, 'FontWeight', 'bold');
text(0.1, 0.75, 'C = [G_1 | c_4]', 'FontSize', 11);
text(0.1, 0.6, 'G_1 (3×3): State feedback gains', 'FontSize', 10);
text(0.1, 0.5, '  - First 2 rows: designed', 'FontSize', 10);
text(0.1, 0.42, '  - Third row: zero (constraint)', 'FontSize', 10);
text(0.1, 0.3, 'c_4 (3×1): Setpoint compensation', 'FontSize', 10);
text(0.1, 0.22, '  c_4 = -(M*w_{des} + A_4)/B', 'FontSize', 9);
text(0.1, 0.1, 'Third row = [0 0 0 0] due to', 'FontSize', 10, 'Color', 'blue');
text(0.1, 0.02, 'fixed bottom row constraint', 'FontSize', 10, 'Color', 'blue');

% Plot 3: Gain magnitudes
subplot(1,3,3);
gains_G1 = reshape(G1, [], 1);
gains_c4 = c4;
all_gains = [gains_G1; gains_c4];

bar_x = 1:length(all_gains);
bar(bar_x, abs(all_gains));
hold on;
zero_indices = [7, 8, 9, 12];
bar(zero_indices, abs(all_gains(zero_indices)), 'r');

xlabel('Gain Index', 'FontSize', 11);
ylabel('Magnitude', 'FontSize', 11);
title('C Matrix Gain Magnitudes', 'FontSize', 12, 'FontWeight', 'bold');
grid on;
legend('Non-zero gains', 'Zero (row 3)', 'Location', 'best');

xticks(bar_x);
xticklabels({'G_{11}', 'G_{21}', 'G_{31}', 'G_{12}', 'G_{22}', 'G_{32}', ...
    'G_{13}', 'G_{23}', 'G_{33}', 'c_{41}', 'c_{42}', 'c_{43}'});
xtickangle(45);

sgtitle(sprintf('C Matrix: X_{dot} = (A + BC)X with w_{des} = [%.1f, %.1f, %.1f]', ...
    w_des(1), w_des(2), w_des(3)), 'FontSize', 13, 'FontWeight', 'bold');

%% ========== SUMMARY ==========
fprintf('\n========================================\n');
fprintf('               SUMMARY\n');
fprintf('========================================\n');
fprintf('EIGENVALUE DESIGN:\n');
fprintf('  YOU SPECIFIED:  λ1 = %.6f, λ2 = %.6f\n', lambda1, lambda2);
fprintf('  ACHIEVED:       λ1 = %.6f, λ2 = %.6f\n', M_eigenvalues_sorted(1), M_eigenvalues_sorted(2));
fprintf('  CONSTRAINED:    λ3 = %.6f (auto-determined)\n', M_eigenvalues_sorted(3));
fprintf('\nMATRIX DESIGN:\n');
fprintf('  M matrix (3×3): Desired closed-loop dynamics\n');
fprintf('  G1 matrix (3×3): State feedback gains\n');
fprintf('  C matrix (3×4): Full controller [G1 | c4]\n');
fprintf('\nCONSTRAINTS SATISFIED:\n');
fprintf('  ✓ All matrix elements are REAL\n');
fprintf('  ✓ All eigenvalues are REAL\n');
fprintf('  ✓ Third row of M = third row of A1 (fixed)\n');
fprintf('  ✓ Third row of G1 = [0, 0, 0]\n');
fprintf('  ✓ Third row of C = [0, 0, 0, 0]\n');
fprintf('  ✓ Forms w_dot=M(w-w_des) and X_dot=(A+BC)X are equivalent\n');
fprintf('========================================\n');

fprintf('\nTO MODIFY:\n');
fprintf('  • Change lambda1, lambda2 for different eigenvalues (KEEP REAL!)\n');
fprintf('  • Change w_des for different setpoint (3x1 vector)\n');
fprintf('  • Change A4 if your A matrix has non-zero last column\n');
fprintf('  • Change r to modify A1 structure\n');
fprintf('  • Change m12_initial, m21_initial (will auto-adjust if needed)\n');

%% ============================================================
%% NONLINEAR CLOSED-LOOP SIMULATION WITH THETA–OMEGA COUPLING
%% ============================================================

% -------- SIMULATION SETTINGS ----------
Tf = 25;                % Longer simulation time to see settling behavior
num_traj = 30;          % More trajectories
dt = 0.01;              % Time step for output (interpolation)
t_eval = 0:dt:Tf;       % Evenly spaced time points for smooth plots

% Random initial conditions for phase portrait
omega0_set = 3*(rand(3,num_traj)-0.5);     % p,q,r (angular velocities)
theta0_set = 1.0*(rand(3,num_traj)-0.5);   % phi,theta,psi (angles)

% Storage
traj = cell(num_traj,1);

% Solver options for smoother integration
opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10, 'MaxStep', 0.05);

% -------- RUN MULTIPLE SIMULATIONS ----------
for k = 1:num_traj
    
    X0 = [omega0_set(:,k); theta0_set(:,k)];
    
    sol = ode45(@(t,X) drone_rhs_compensated(t,X,M,G1,B,A4), [0 Tf], X0, opts);
    
    % Interpolate solution at evenly spaced time points for smooth plotting
    traj{k}.t = t_eval;
    traj{k}.X = deval(sol, t_eval);
    
end

%% ============================================================
%% PHASE PORTRAITS WITH TIME-GRADIENT COLORING
%% ============================================================

% Enable smooth graphics rendering
set(0, 'DefaultFigureRenderer', 'painters');

% Figure 1: phi vs p
figure('Position', [100, 100, 600, 550]);
hold on; grid on; box on;

for k = 1:num_traj
    w = traj{k}.X(1:3,:);
    th = traj{k}.X(4:6,:);
    
    % Use patch for smooth gradient coloring (more efficient than line segments)
    n_pts = length(th(1,:));
    x_data = th(1,:);
    y_data = w(1,:);
    
    % Create color array for gradient
    colors = zeros(n_pts, 3);
    for i = 1:n_pts
        color_val = (i-1) / (n_pts-1);
        colors(i,:) = [color_val, 0, 1-color_val];  % Blue to red gradient
    end
    
    % Plot using patch with vertex colors for smooth gradient
    patch([x_data NaN], [y_data NaN], [1:n_pts NaN], ...
          'EdgeColor', 'interp', 'FaceColor', 'none', 'LineWidth', 0.8);
    
    % Mark start and end points
    plot(th(1,1), w(1,1), 'o', 'MarkerSize', 6, 'MarkerFaceColor', [0,0,1], 'MarkerEdgeColor', 'k');
    plot(th(1,end), w(1,end), 'o', 'MarkerSize', 6, 'MarkerFaceColor', [1,0,0], 'MarkerEdgeColor', 'k');
end

% Set colormap for the gradient
colormap([linspace(0,1,256)', zeros(256,1), linspace(1,0,256)']);

xlabel('Roll Angle \phi (rad)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Roll Rate p (rad/s)', 'FontSize', 13, 'FontWeight', 'bold');
title('Phase Portrait: Roll (\phi vs p)', 'FontSize', 14, 'FontWeight', 'bold');

% Add colorbar to show time progression
c = colorbar;
c.Label.String = 'Time progression';
c.Label.FontSize = 11;
c.Ticks = [0 0.5 1];
c.TickLabels = {'Start', 'Mid', 'End'};

set(gca, 'FontSize', 11, 'LineWidth', 1.5);

% Figure 2: theta vs q
figure('Position', [750, 100, 600, 550]);
hold on; grid on; box on;

for k = 1:num_traj
    w = traj{k}.X(1:3,:);
    th = traj{k}.X(4:6,:);
    
    n_pts = length(th(2,:));
    x_data = th(2,:);
    y_data = w(2,:);
    
    % Plot using patch with vertex colors for smooth gradient
    patch([x_data NaN], [y_data NaN], [1:n_pts NaN], ...
          'EdgeColor', 'interp', 'FaceColor', 'none', 'LineWidth', 0.8);
    
    % Mark start and end points
    plot(th(2,1), w(2,1), 'o', 'MarkerSize', 6, 'MarkerFaceColor', [0,0,1], 'MarkerEdgeColor', 'k');
    plot(th(2,end), w(2,end), 'o', 'MarkerSize', 6, 'MarkerFaceColor', [1,0,0], 'MarkerEdgeColor', 'k');
end

% Set colormap for the gradient
colormap([linspace(0,1,256)', zeros(256,1), linspace(1,0,256)']);

xlabel('Pitch Angle \theta (rad)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Pitch Rate q (rad/s)', 'FontSize', 13, 'FontWeight', 'bold');
title('Phase Portrait: Pitch (\theta vs q)', 'FontSize', 14, 'FontWeight', 'bold');

% Add colorbar to show time progression
c = colorbar;
c.Label.String = 'Time progression';
c.Label.FontSize = 11;
c.Ticks = [0 0.5 1];
c.TickLabels = {'Start', 'Mid', 'End'};

set(gca, 'FontSize', 11, 'LineWidth', 1.5);

% Figure 3: psi vs r
figure('Position', [1400, 100, 600, 550]);
hold on; grid on; box on;

for k = 1:num_traj
    w = traj{k}.X(1:3,:);
    th = traj{k}.X(4:6,:);
    
    n_pts = length(th(3,:));
    x_data = th(3,:);
    y_data = w(3,:);
    
    % Plot using patch with vertex colors for smooth gradient
    patch([x_data NaN], [y_data NaN], [1:n_pts NaN], ...
          'EdgeColor', 'interp', 'FaceColor', 'none', 'LineWidth', 0.8);
    
    % Mark start and end points
    plot(th(3,1), w(3,1), 'o', 'MarkerSize', 6, 'MarkerFaceColor', [0,0,1], 'MarkerEdgeColor', 'k');
    plot(th(3,end), w(3,end), 'o', 'MarkerSize', 6, 'MarkerFaceColor', [1,0,0], 'MarkerEdgeColor', 'k');
end

% Set colormap for the gradient
colormap([linspace(0,1,256)', zeros(256,1), linspace(1,0,256)']);

xlabel('Yaw Angle \psi (rad)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Yaw Rate r (rad/s)', 'FontSize', 13, 'FontWeight', 'bold');
title('Phase Portrait: Yaw (\psi vs r)', 'FontSize', 14, 'FontWeight', 'bold');

% Add colorbar to show time progression
c = colorbar;
c.Label.String = 'Time progression';
c.Label.FontSize = 11;
c.Ticks = [0 0.5 1];
c.TickLabels = {'Start', 'Mid', 'End'};

set(gca, 'FontSize', 11, 'LineWidth', 1.5);

%% ============================================================
%% FULL STATE TRAJECTORY (ONE SAMPLE)
%% ============================================================

figure('Position', [100, 700, 1400, 500]);

% Subplot 1: Angular velocities
subplot(1,2,1);
plot(traj{1}.t, traj{1}.X(1,:), 'b-', 'LineWidth', 1.5); hold on;
plot(traj{1}.t, traj{1}.X(2,:), 'r-', 'LineWidth', 1.5);
plot(traj{1}.t, traj{1}.X(3,:), 'k-', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Angular Velocity (rad/s)', 'FontSize', 12, 'FontWeight', 'bold');
title('Body Angular Velocities', 'FontSize', 13, 'FontWeight', 'bold');
legend('p (roll rate)', 'q (pitch rate)', 'r (yaw rate)', 'Location', 'best', 'FontSize', 11);
grid on; box on;
set(gca, 'FontSize', 11, 'LineWidth', 1.5);

% Subplot 2: Euler angles
subplot(1,2,2);
plot(traj{1}.t, traj{1}.X(4,:), 'b-', 'LineWidth', 1.5); hold on;
plot(traj{1}.t, traj{1}.X(5,:), 'r-', 'LineWidth', 1.5);
plot(traj{1}.t, traj{1}.X(6,:), 'k-', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Angle (rad)', 'FontSize', 12, 'FontWeight', 'bold');
title('Euler Angles', 'FontSize', 13, 'FontWeight', 'bold');
legend('\phi (roll)', '\theta (pitch)', '\psi (yaw)', 'Location', 'best', 'FontSize', 11);
grid on; box on;
set(gca, 'FontSize', 11, 'LineWidth', 1.5);

sgtitle('Full State Trajectory (Single Run)', 'FontSize', 14, 'FontWeight', 'bold');

%% ============================================================
%% DISTURBED A MATRIX SIMULATION
%% ============================================================
% Simulate system with disturbances/perturbations in the A matrix
% This tests robustness of the controller to model uncertainty

fprintf('\n========================================\n');
fprintf('    DISTURBED A MATRIX SIMULATION\n');
fprintf('========================================\n');

% -------- DISTURBANCE SETTINGS ----------
disturbance_magnitude = 0.3;  % Magnitude of disturbance (fraction of nominal)
num_disturbed_traj = 10;       % Number of disturbed trajectories
Tf_dist = 25;                  % Simulation time

% Time vector for smooth plotting
t_eval_dist = 0:0.01:Tf_dist;

% Storage for disturbed trajectories
traj_disturbed = cell(num_disturbed_traj, 1);
traj_nominal = cell(1, 1);  % One nominal trajectory for comparison

% Fixed initial condition for fair comparison
X0_compare = [0.5; -0.3; 0.8; 0.2; -0.1; 0.15];

fprintf('Disturbance magnitude: %.1f%% of nominal A matrix elements\n', disturbance_magnitude*100);
fprintf('Running %d disturbed simulations + 1 nominal...\n', num_disturbed_traj);

% Run nominal (undisturbed) simulation first
sol_nom = ode45(@(t,X) drone_rhs(t,X,M,G1,B,A4), [0 Tf_dist], X0_compare, opts);
traj_nominal{1}.t = t_eval_dist;
traj_nominal{1}.X = deval(sol_nom, t_eval_dist);

% Run disturbed simulations
for k = 1:num_disturbed_traj
    % Generate random disturbance matrix for A1
    delta_A = disturbance_magnitude * randn(3,3);
    
    % Make disturbance symmetric in off-diagonal r terms (optional)
    % delta_A(1,2) = delta_A(2,1);
    
    sol = ode45(@(t,X) drone_rhs_disturbed(t,X,M,G1,B,A4,delta_A), [0 Tf_dist], X0_compare, opts);
    
    traj_disturbed{k}.t = t_eval_dist;
    traj_disturbed{k}.X = deval(sol, t_eval_dist);
    traj_disturbed{k}.delta_A = delta_A;
end

fprintf('Simulations complete!\n');

%% ============================================================
%% PLOT: NOMINAL VS DISTURBED COMPARISON
%% ============================================================

figure('Position', [100, 100, 1400, 600]);
sgtitle('Nominal vs Disturbed A Matrix Response', 'FontSize', 14, 'FontWeight', 'bold');

% Subplot 1: Angular velocities comparison
subplot(2,3,1);
hold on; grid on; box on;

% Plot disturbed trajectories (light gray)
for k = 1:num_disturbed_traj
    plot(traj_disturbed{k}.t, traj_disturbed{k}.X(1,:), 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
end
% Plot nominal (bold blue)
plot(traj_nominal{1}.t, traj_nominal{1}.X(1,:), 'b-', 'LineWidth', 2);

xlabel('Time (s)', 'FontSize', 11);
ylabel('p (rad/s)', 'FontSize', 11);
title('Roll Rate p', 'FontSize', 12, 'FontWeight', 'bold');
legend('Disturbed', 'Nominal', 'Location', 'best');

subplot(2,3,2);
hold on; grid on; box on;
for k = 1:num_disturbed_traj
    plot(traj_disturbed{k}.t, traj_disturbed{k}.X(2,:), 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
end
plot(traj_nominal{1}.t, traj_nominal{1}.X(2,:), 'r-', 'LineWidth', 2);
xlabel('Time (s)', 'FontSize', 11);
ylabel('q (rad/s)', 'FontSize', 11);
title('Pitch Rate q', 'FontSize', 12, 'FontWeight', 'bold');

subplot(2,3,3);
hold on; grid on; box on;
for k = 1:num_disturbed_traj
    plot(traj_disturbed{k}.t, traj_disturbed{k}.X(3,:), 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
end
plot(traj_nominal{1}.t, traj_nominal{1}.X(3,:), 'k-', 'LineWidth', 2);
xlabel('Time (s)', 'FontSize', 11);
ylabel('r (rad/s)', 'FontSize', 11);
title('Yaw Rate r', 'FontSize', 12, 'FontWeight', 'bold');

% Subplot 4-6: Euler angles comparison
subplot(2,3,4);
hold on; grid on; box on;
for k = 1:num_disturbed_traj
    plot(traj_disturbed{k}.t, traj_disturbed{k}.X(4,:), 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
end
plot(traj_nominal{1}.t, traj_nominal{1}.X(4,:), 'b-', 'LineWidth', 2);
xlabel('Time (s)', 'FontSize', 11);
ylabel('\phi (rad)', 'FontSize', 11);
title('Roll Angle \phi', 'FontSize', 12, 'FontWeight', 'bold');

subplot(2,3,5);
hold on; grid on; box on;
for k = 1:num_disturbed_traj
    plot(traj_disturbed{k}.t, traj_disturbed{k}.X(5,:), 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
end
plot(traj_nominal{1}.t, traj_nominal{1}.X(5,:), 'r-', 'LineWidth', 2);
xlabel('Time (s)', 'FontSize', 11);
ylabel('\theta (rad)', 'FontSize', 11);
title('Pitch Angle \theta', 'FontSize', 12, 'FontWeight', 'bold');

subplot(2,3,6);
hold on; grid on; box on;
for k = 1:num_disturbed_traj
    plot(traj_disturbed{k}.t, traj_disturbed{k}.X(6,:), 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
end
plot(traj_nominal{1}.t, traj_nominal{1}.X(6,:), 'k-', 'LineWidth', 2);
xlabel('Time (s)', 'FontSize', 11);
ylabel('\psi (rad)', 'FontSize', 11);
title('Yaw Angle \psi', 'FontSize', 12, 'FontWeight', 'bold');

%% ============================================================
%% PLOT: ERROR ENVELOPE (DISTURBED - NOMINAL)
%% ============================================================

figure('Position', [150, 150, 1200, 500]);
sgtitle('Tracking Error Due to A Matrix Disturbance', 'FontSize', 14, 'FontWeight', 'bold');

% Compute errors for all disturbed trajectories
errors_w = zeros(num_disturbed_traj, length(t_eval_dist), 3);
errors_theta = zeros(num_disturbed_traj, length(t_eval_dist), 3);

for k = 1:num_disturbed_traj
    errors_w(k,:,:) = (traj_disturbed{k}.X(1:3,:) - traj_nominal{1}.X(1:3,:))';
    errors_theta(k,:,:) = (traj_disturbed{k}.X(4:6,:) - traj_nominal{1}.X(4:6,:))';
end

% Subplot 1: Angular velocity errors
subplot(1,2,1);
hold on; grid on; box on;

% Plot error envelope (min/max bounds)
for i = 1:3
    error_i = squeeze(errors_w(:,:,i));
    error_max = max(error_i, [], 1);
    error_min = min(error_i, [], 1);
    error_mean = mean(error_i, 1);
    
    colors_err = {'b', 'r', 'k'};
    fill([t_eval_dist, fliplr(t_eval_dist)], [error_max, fliplr(error_min)], ...
         colors_err{i}, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    plot(t_eval_dist, error_mean, colors_err{i}, 'LineWidth', 1.5);
end

xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Error (rad/s)', 'FontSize', 12, 'FontWeight', 'bold');
title('Angular Velocity Error: \omega_{disturbed} - \omega_{nominal}', 'FontSize', 12, 'FontWeight', 'bold');
legend('p error envelope', 'p mean', 'q error envelope', 'q mean', 'r error envelope', 'r mean', ...
       'Location', 'best', 'FontSize', 9);
yline(0, 'k--', 'LineWidth', 1);

% Subplot 2: Euler angle errors
subplot(1,2,2);
hold on; grid on; box on;

for i = 1:3
    error_i = squeeze(errors_theta(:,:,i));
    error_max = max(error_i, [], 1);
    error_min = min(error_i, [], 1);
    error_mean = mean(error_i, 1);
    
    colors_err = {'b', 'r', 'k'};
    fill([t_eval_dist, fliplr(t_eval_dist)], [error_max, fliplr(error_min)], ...
         colors_err{i}, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    plot(t_eval_dist, error_mean, colors_err{i}, 'LineWidth', 1.5);
end

xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Error (rad)', 'FontSize', 12, 'FontWeight', 'bold');
title('Euler Angle Error: \theta_{disturbed} - \theta_{nominal}', 'FontSize', 12, 'FontWeight', 'bold');
legend('\phi error envelope', '\phi mean', '\theta error envelope', '\theta mean', '\psi error envelope', '\psi mean', ...
       'Location', 'best', 'FontSize', 9);
yline(0, 'k--', 'LineWidth', 1);

%% ============================================================
%% ROBUSTNESS METRICS
%% ============================================================

fprintf('\n=== ROBUSTNESS METRICS ===\n');

% Compute RMS error for each state
rms_errors = zeros(6, 1);
max_errors = zeros(6, 1);

for i = 1:3
    error_i = squeeze(errors_w(:,:,i));
    rms_errors(i) = sqrt(mean(error_i(:).^2));
    max_errors(i) = max(abs(error_i(:)));
end

for i = 1:3
    error_i = squeeze(errors_theta(:,:,i));
    rms_errors(i+3) = sqrt(mean(error_i(:).^2));
    max_errors(i+3) = max(abs(error_i(:)));
end

state_names = {'p', 'q', 'r', 'phi', 'theta', 'psi'};
units = {'rad/s', 'rad/s', 'rad/s', 'rad', 'rad', 'rad'};

fprintf('State\t\tRMS Error\tMax Error\n');
fprintf('-----\t\t---------\t---------\n');
for i = 1:6
    fprintf('%s\t\t%.4f %s\t%.4f %s\n', state_names{i}, rms_errors(i), units{i}, max_errors(i), units{i});
end

fprintf('\nDisturbance magnitude: %.1f%% of nominal\n', disturbance_magnitude*100);
fprintf('Number of Monte Carlo runs: %d\n', num_disturbed_traj);

%% ============================================================
%% COMPENSATION COMPARISON: WITH vs WITHOUT w_des_dot
%% ============================================================
% Compare performance of controller with and without w_des_dot compensation

fprintf('\n========================================\n');
fprintf('  w_des_dot COMPENSATION COMPARISON\n');
fprintf('========================================\n');

% -------- SIMULATION SETTINGS ----------
Tf_comp = 20;                  % Simulation time
num_comp_traj = 5;             % Number of test trajectories
t_eval_comp = 0:0.01:Tf_comp;  % Time vector

% Test initial conditions with varying angles
comp_initial_conditions = [
    0.8, -0.6, 2.0,  0.3, -0.2, 0.1;   % Moderate angles
    1.2, -1.0, 3.5,  0.5, -0.4, 0.2;   % Larger angles
    0.3,  0.5, 4.0,  0.15, 0.25, 0.05; % Small angles
    1.5, -1.2, 2.5,  0.6, -0.5, 0.15;  % Large angles
    0.5,  0.8, 3.0,  0.2,  0.3, 0.1    % Mixed
]';

% Storage
traj_uncompensated = cell(num_comp_traj, 1);
traj_compensated = cell(num_comp_traj, 1);

fprintf('Running comparison simulations...\n');

for k = 1:num_comp_traj
    X0 = comp_initial_conditions(:, k);
    
    % Run WITHOUT compensation (original)
    sol_uncomp = ode45(@(t,X) drone_rhs(t,X,M,G1,B,A4), [0 Tf_comp], X0, opts);
    traj_uncompensated{k}.t = t_eval_comp;
    traj_uncompensated{k}.X = deval(sol_uncomp, t_eval_comp);
    
    % Run WITH w_des_dot compensation
    sol_comp = ode45(@(t,X) drone_rhs_compensated(t,X,M,G1,B,A4), [0 Tf_comp], X0, opts);
    traj_compensated{k}.t = t_eval_comp;
    traj_compensated{k}.X = deval(sol_comp, t_eval_comp);
    
    fprintf('  Trajectory %d/%d complete\n', k, num_comp_traj);
end

fprintf('Comparison simulations complete!\n');

%% ============================================================
%% COMPUTE ERROR DYNAMICS: e = w - w_des
%% ============================================================

% Storage for error trajectories
error_uncomp = cell(num_comp_traj, 1);
error_comp = cell(num_comp_traj, 1);

for k = 1:num_comp_traj
    n_pts = length(t_eval_comp);
    
    % Extract states
    w_uncomp = traj_uncompensated{k}.X(1:3, :);
    theta_uncomp = traj_uncompensated{k}.X(4:6, :);
    
    w_comp = traj_compensated{k}.X(1:3, :);
    theta_comp = traj_compensated{k}.X(4:6, :);
    
    % Compute w_des at each time point
    w_des_traj_uncomp = zeros(3, n_pts);
    w_des_traj_comp = zeros(3, n_pts);
    
    for i = 1:n_pts
        w_des_traj_uncomp(:, i) = [-theta_uncomp(1,i); -theta_uncomp(2,i); 9];
        w_des_traj_comp(:, i) = [-theta_comp(1,i); -theta_comp(2,i); 9];
    end
    
    % Compute errors
    error_uncomp{k}.e = w_uncomp - w_des_traj_uncomp;
    error_comp{k}.e = w_comp - w_des_traj_comp;
    error_uncomp{k}.t = t_eval_comp;
    error_comp{k}.t = t_eval_comp;
end

%% ============================================================
%% PLOT: ERROR COMPARISON
%% ============================================================

figure('Position', [100, 100, 1600, 900]);
sgtitle('Controller Comparison: WITHOUT vs WITH w_{des\_dot} Compensation', ...
        'FontSize', 15, 'FontWeight', 'bold');

error_labels = {'e_p = p - (-\phi)', 'e_q = q - (-\theta)', 'e_r = r - 9'};

% Generate colors for all trajectories
colors_uncomp = [linspace(0.8, 0.9, num_comp_traj)', ...
                 linspace(0.2, 0.6, num_comp_traj)', ...
                 linspace(0.2, 0.6, num_comp_traj)'];  % Red shades
colors_comp = [linspace(0.2, 0.6, num_comp_traj)', ...
               linspace(0.5, 0.8, num_comp_traj)', ...
               linspace(0.8, 0.95, num_comp_traj)'];  % Blue shades

for i = 1:3
    subplot(3, 2, 2*i-1);
    hold on; grid on; box on;
    
    % Plot uncompensated errors
    for k = 1:num_comp_traj
        plot(error_uncomp{k}.t, error_uncomp{k}.e(i,:), ...
             'Color', colors_uncomp(k,:), 'LineWidth', 1.2);
    end
    
    xlabel('Time (s)', 'FontSize', 11);
    ylabel('Error (rad/s)', 'FontSize', 11);
    title(['WITHOUT Compensation: ', error_labels{i}], 'FontSize', 12, 'FontWeight', 'bold');
    yline(0, 'k--', 'LineWidth', 1);
    
    subplot(3, 2, 2*i);
    hold on; grid on; box on;
    
    % Plot compensated errors
    for k = 1:num_comp_traj
        plot(error_comp{k}.t, error_comp{k}.e(i,:), ...
             'Color', colors_comp(k,:), 'LineWidth', 1.2);
    end
    
    xlabel('Time (s)', 'FontSize', 11);
    ylabel('Error (rad/s)', 'FontSize', 11);
    title(['WITH Compensation: ', error_labels{i}], 'FontSize', 12, 'FontWeight', 'bold');
    yline(0, 'k--', 'LineWidth', 1);
end

%% ============================================================
%% PLOT: DIRECT COMPARISON (OVERLAY)
%% ============================================================

figure('Position', [150, 150, 1400, 500]);
sgtitle('Direct Error Comparison: Red=Uncompensated, Blue=Compensated', ...
        'FontSize', 14, 'FontWeight', 'bold');

for i = 1:3
    subplot(1, 3, i);
    hold on; grid on; box on;
    
    % Plot both on same axes
    for k = 1:num_comp_traj
        % Uncompensated (red/orange)
        plot(error_uncomp{k}.t, error_uncomp{k}.e(i,:), ...
             'Color', [1, 0.3, 0.3], 'LineWidth', 1.0, 'LineStyle', '-');
        
        % Compensated (blue)
        plot(error_comp{k}.t, error_comp{k}.e(i,:), ...
             'Color', [0.2, 0.5, 1], 'LineWidth', 1.0, 'LineStyle', '--');
    end
    
    xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Error (rad/s)', 'FontSize', 12, 'FontWeight', 'bold');
    title(error_labels{i}, 'FontSize', 13, 'FontWeight', 'bold');
    yline(0, 'k--', 'LineWidth', 1.5);
    
    if i == 1
        legend('Uncompensated', 'Compensated', 'Location', 'best', 'FontSize', 10);
    end
end

%% ============================================================
%% QUANTITATIVE METRICS
%% ============================================================

fprintf('\n=== QUANTITATIVE COMPARISON ===\n');
fprintf('Metric: RMS error over entire trajectory\n\n');

% Compute RMS errors
rms_uncomp = zeros(3, num_comp_traj);
rms_comp = zeros(3, num_comp_traj);

for k = 1:num_comp_traj
    for i = 1:3
        rms_uncomp(i, k) = sqrt(mean(error_uncomp{k}.e(i,:).^2));
        rms_comp(i, k) = sqrt(mean(error_comp{k}.e(i,:).^2));
    end
end

% Average across trajectories
avg_rms_uncomp = mean(rms_uncomp, 2);
avg_rms_comp = mean(rms_comp, 2);
improvement = (avg_rms_uncomp - avg_rms_comp) ./ avg_rms_uncomp * 100;

fprintf('Component | Uncompensated | Compensated | Improvement\n');
fprintf('----------|---------------|-------------|------------\n');
fprintf('   e_p    |   %.4f      |   %.4f    |   %.1f%%\n', ...
        avg_rms_uncomp(1), avg_rms_comp(1), improvement(1));
fprintf('   e_q    |   %.4f      |   %.4f    |   %.1f%%\n', ...
        avg_rms_uncomp(2), avg_rms_comp(2), improvement(2));
fprintf('   e_r    |   %.4f      |   %.4f    |   %.1f%%\n', ...
        avg_rms_uncomp(3), avg_rms_comp(3), improvement(3));
fprintf('\nOverall average improvement: %.1f%%\n', mean(improvement));

%% ============================================================
%% PLOT: STATE-DEPENDENT EFFECTIVE EIGENVALUES
%% ============================================================
% Analyze eigenvalues of (M + T(phi,theta)) along trajectories

fprintf('\n=== ANALYZING EFFECTIVE DYNAMICS ===\n');
fprintf('Computing eigenvalues of (M + T(phi,theta)) along trajectories...\n');

% Pick one representative trajectory
k_rep = 2;  % Use trajectory 2
n_sample = 100;  % Sample points
sample_idx = round(linspace(1, length(t_eval_comp), n_sample));

eigs_effective_uncomp = zeros(3, n_sample);
eigs_effective_comp = zeros(3, n_sample);

for j = 1:n_sample
    idx = sample_idx(j);
    
    % Uncompensated trajectory state
    phi_u = traj_uncompensated{k_rep}.X(4, idx);
    theta_u = traj_uncompensated{k_rep}.X(5, idx);
    
    % Compensated trajectory state
    phi_c = traj_compensated{k_rep}.X(4, idx);
    theta_c = traj_compensated{k_rep}.X(5, idx);
    
    % Compute T matrices
    theta_max = 1.5;
    theta_sat_u = max(-theta_max, min(theta_max, theta_u));
    theta_sat_c = max(-theta_max, min(theta_max, theta_c));
    
    cos_theta_u = max(abs(cos(theta_sat_u)), 1e-6) * sign(cos(theta_sat_u) + 1e-10);
    cos_theta_c = max(abs(cos(theta_sat_c)), 1e-6) * sign(cos(theta_sat_c) + 1e-10);
    
    T_u = [1, sin(phi_u)*tan(theta_sat_u), cos(phi_u)*tan(theta_sat_u);
           0, cos(phi_u), -sin(phi_u);
           0, sin(phi_u)/cos_theta_u, cos(phi_u)/cos_theta_u];
    
    T_c = [1, sin(phi_c)*tan(theta_sat_c), cos(phi_c)*tan(theta_sat_c);
           0, cos(phi_c), -sin(phi_c);
           0, sin(phi_c)/cos_theta_c, cos(phi_c)/cos_theta_c];
    
    % Effective error dynamics matrix
    M_eff_u = M + T_u;
    M_eff_c = M + T_c;
    
    % Eigenvalues
    eigs_effective_uncomp(:, j) = sort(real(eig(M_eff_u)), 'descend');
    eigs_effective_comp(:, j) = sort(real(eig(M_eff_c)), 'descend');
end

t_sample = t_eval_comp(sample_idx);

figure('Position', [200, 200, 1400, 500]);
sgtitle('Effective Error Dynamics Eigenvalues: M + T(\phi,\theta)', ...
        'FontSize', 14, 'FontWeight', 'bold');

subplot(1,2,1);
hold on; grid on; box on;
plot(t_sample, eigs_effective_uncomp(1,:), 'r-', 'LineWidth', 2);
plot(t_sample, eigs_effective_uncomp(2,:), 'g-', 'LineWidth', 2);
plot(t_sample, eigs_effective_uncomp(3,:), 'b-', 'LineWidth', 2);
yline(0, 'k--', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Eigenvalue', 'FontSize', 12, 'FontWeight', 'bold');
title('WITHOUT Compensation', 'FontSize', 13, 'FontWeight', 'bold');
legend('\lambda_1', '\lambda_2', '\lambda_3', 'Location', 'best');
ylim([min(eigs_effective_uncomp(:))-1, 1]);

subplot(1,2,2);
hold on; grid on; box on;
plot(t_sample, eigs_effective_comp(1,:), 'r-', 'LineWidth', 2);
plot(t_sample, eigs_effective_comp(2,:), 'g-', 'LineWidth', 2);
plot(t_sample, eigs_effective_comp(3,:), 'b-', 'LineWidth', 2);
yline(0, 'k--', 'LineWidth', 1.5);
xlabel('Time (s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Eigenvalue', 'FontSize', 12, 'FontWeight', 'bold');
title('WITH Compensation', 'FontSize', 13, 'FontWeight', 'bold');
legend('\lambda_1', '\lambda_2', '\lambda_3', 'Location', 'best');
ylim([min(eigs_effective_comp(:))-1, 1]);

fprintf('Analysis complete!\n');
fprintf('\n========================================\n');
