%% Eigenvalue and Damping Analysis for Various Yaw Rates (r)
% This script sweeps through different yaw rate values and analyzes:
% - Eigenvalues of the closed-loop M matrix
% - Eigenvectors
% - Damping ratios and natural frequencies
% All matrices (A1, M, G1, etc.) are updated based on the current yaw rate r

clear; clc; close all;

%% ========== SWEEP PARAMETERS ==========
r_values = linspace(0.1, 15, 50);   % Range of yaw rates to test (avoid r=0 for stability)
num_r = length(r_values);

% Desired eigenvalues (user-specified)
lambda1 = -5;   % First desired eigenvalue (MUST BE REAL)
lambda2 = -4;   % Second desired eigenvalue (MUST BE REAL)

% Free parameters for M matrix construction
m13 = 0;
m23 = 0;
m12_initial = 1;   % Initial guess for m12
m21_initial = 1;   % Initial guess for m21

% Scalar B (control effectiveness)
B = 0.5;

% A4 column (last column of A matrix)
A4 = [0; 0; 0];

% Note: w_des = [0; 0; r] is computed inside the loop for each r value

% Damping coefficient (diagonal elements of A1)
damping_coeff = -0.3;
yaw_damping = -2.0;

%% ========== STORAGE FOR RESULTS ==========
% Closed-loop (with feedback)
eigenvalues_all = zeros(3, num_r);          % Closed-loop eigenvalues of M
eigenvectors_all = zeros(3, 3, num_r);      % Closed-loop eigenvectors
damping_ratios = zeros(3, num_r);           % Damping ratios
natural_frequencies = zeros(3, num_r);      % Natural frequencies
M_matrices = zeros(3, 3, num_r);            % M matrices for each r
G1_matrices = zeros(3, 3, num_r);           % Gain matrices for each r

% Open-loop (without feedback) - THIS CHANGES WITH r
A1_matrices = zeros(3, 3, num_r);           % A1 matrices for each r
A1_eigenvalues = zeros(3, num_r);           % Open-loop eigenvalues of A1(r)
A1_eigenvectors = zeros(3, 3, num_r);       % Open-loop eigenvectors

w_des_all = zeros(3, num_r);                % w_des for each r (w_des = [0; 0; r])
stability_flag = true(1, num_r);            % Stability indicator

%% ========== COMPUTE M MATRIX (CONSTANT - does not depend on r) ==========
fprintf('=== COMPUTING CLOSED-LOOP M MATRIX (CONSTANT) ===\n');

% Bottom row of M comes from A1, but it's [0, 0, yaw_damping] which is constant
a31 = 0;
a32 = 0;
a33 = yaw_damping;

% Solve for M matrix elements from desired eigenvalues
max_product = (lambda1 - lambda2)^2 / 4;
fprintf('For real m11, m22, need: m12*m21 ≤ %.4f\n', max_product);

m12 = m12_initial;
m21 = m21_initial;

% Adjust if necessary to ensure real solution
if m12*m21 > max_product
    fprintf('Initial m12*m21 = %.4f is too large! Auto-adjusting...\n', m12*m21);
    target_product = 0.9 * max_product;
    if m12_initial ~= 0
        ratio = m21_initial / m12_initial;
        m12 = sqrt(abs(target_product / ratio));
        m21 = ratio * m12;
    else
        m12 = sqrt(abs(target_product));
        m21 = sign(target_product) * sqrt(abs(target_product));
    end
    fprintf('Adjusted to: m12 = %.4f, m21 = %.4f\n', m12, m21);
else
    fprintf('m12 = %.4f, m21 = %.4f is acceptable\n', m12, m21);
end

% Solve for m11, m22
sum_m = lambda1 + lambda2;
product_m = lambda1*lambda2 + m12*m21;
discriminant = sum_m^2 - 4*product_m;

if discriminant < -1e-10
    error('Discriminant negative! Cannot achieve real eigenvalues with these parameters.');
elseif discriminant < 0
    discriminant = 0;
end

m11 = (sum_m + sqrt(discriminant))/2;
m22 = (sum_m - sqrt(discriminant))/2;

% Construct M matrix (CONSTANT for all r)
M = [m11, m12, m13;
     m21, m22, m23;
     a31, a32, a33];

fprintf('\nM matrix (CONSTANT for all r values):\n');
disp(M);

% Compute closed-loop eigenvalues/eigenvectors (CONSTANT)
[V_M, D_M] = eig(M);
eig_M = diag(D_M);
[~, sort_idx] = sort(real(eig_M), 'descend');
eig_M = eig_M(sort_idx);
V_M = V_M(:, sort_idx);

fprintf('Closed-loop eigenvalues (CONSTANT):\n');
fprintf('  λ1 = %.4f (specified: %.4f)\n', real(eig_M(1)), lambda1);
fprintf('  λ2 = %.4f (specified: %.4f)\n', real(eig_M(2)), lambda2);
fprintf('  λ3 = %.4f (constrained by yaw_damping = %.4f)\n', real(eig_M(3)), yaw_damping);

fprintf('\nClosed-loop eigenvectors (CONSTANT):\n');
disp(V_M);

%% ========== MAIN SWEEP LOOP (only A1, open-loop eigs, G1 change with r) ==========
fprintf('=== SWEEPING OVER YAW RATE r ===\n');
fprintf('Sweeping r from %.2f to %.2f (%d points)\n', r_values(1), r_values(end), num_r);
fprintf('Only A1(r), open-loop eigenvalues, and G1(r) change with r\n\n');

for idx = 1:num_r
    r = r_values(idx);
    
    %% --- 1. UPDATE A1 MATRIX BASED ON CURRENT r ---
    A1 = [damping_coeff,  r, 0;
                     -r,  damping_coeff, 0;
                      0,  0, yaw_damping];
    
    % Store A1
    A1_matrices(:,:,idx) = A1;
    
    %% --- 2. COMPUTE OPEN-LOOP EIGENVALUES OF A1(r) ---
    % These DO change with r (the yaw coupling affects open-loop dynamics)
    [V_A1, D_A1] = eig(A1);
    eig_A1 = diag(D_A1);
    [~, sort_idx_A1] = sort(real(eig_A1), 'descend');
    eig_A1 = eig_A1(sort_idx_A1);
    V_A1 = V_A1(:, sort_idx_A1);
    
    A1_eigenvalues(:, idx) = eig_A1;
    A1_eigenvectors(:,:,idx) = V_A1;
    
    %% --- 3. COMPUTE w_des FOR CURRENT r ---
    w_des = [0; 0; r];
    w_des_all(:, idx) = w_des;
    
    %% --- 4. STORE M (same for all r) ---
    M_matrices(:,:,idx) = M;
    eigenvalues_all(:, idx) = eig_M;
    eigenvectors_all(:,:,idx) = V_M;
    
    %% --- 5. CHECK STABILITY (M is always stable if designed correctly) ---
    if any(real(eig_M) >= 0)
        stability_flag(idx) = false;
    end
    
    %% --- 6. COMPUTE DAMPING RATIOS AND NATURAL FREQUENCIES (CONSTANT) ---
    for k = 1:3
        lambda_k = eig_M(k);
        
        if isreal(lambda_k) || abs(imag(lambda_k)) < 1e-10
            wn = abs(real(lambda_k));
            zeta = 1.0;
        else
            sigma = real(lambda_k);
            omega_d = imag(lambda_k);
            wn = sqrt(sigma^2 + omega_d^2);
            zeta = -sigma / wn;
        end
        
        natural_frequencies(k, idx) = wn;
        damping_ratios(k, idx) = zeta;
    end
    
    %% --- 7. COMPUTE GAIN MATRIX G1 (CHANGES with r) ---
    G1 = (M - A1) / B;
    G1_matrices(:,:,idx) = G1;
end

fprintf('Sweep complete!\n\n');

%% ========== SUMMARY STATISTICS ==========
fprintf('=== SUMMARY ===\n');
fprintf('Total configurations tested: %d\n', num_r);
fprintf('Stable configurations: %d\n', sum(stability_flag));
fprintf('Unstable configurations: %d\n', sum(~stability_flag));

%% ========== VISUALIZATION ==========

% Figure 1: Eigenvalues vs r
figure('Position', [100, 100, 1400, 500]);
sgtitle('Eigenvalue Analysis vs Yaw Rate r', 'FontSize', 14, 'FontWeight', 'bold');

% Real part of eigenvalues
subplot(1,3,1);
hold on; grid on; box on;
colors = {'b', 'r', 'k'};
for k = 1:3
    plot(r_values, real(eigenvalues_all(k,:)), [colors{k}, '-'], 'LineWidth', 2);
end
yline(0, 'g--', 'LineWidth', 2, 'DisplayName', 'Stability Boundary');
xlabel('Yaw Rate r (rad/s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Real Part of Eigenvalue', 'FontSize', 12, 'FontWeight', 'bold');
title('Eigenvalue Real Parts', 'FontSize', 13, 'FontWeight', 'bold');
legend('\lambda_1', '\lambda_2', '\lambda_3 (constrained)', 'Stability Boundary', 'Location', 'best');

% Imaginary part of eigenvalues
subplot(1,3,2);
hold on; grid on; box on;
for k = 1:3
    plot(r_values, imag(eigenvalues_all(k,:)), [colors{k}, '-'], 'LineWidth', 2);
end
yline(0, 'k--', 'LineWidth', 1);
xlabel('Yaw Rate r (rad/s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Imaginary Part of Eigenvalue', 'FontSize', 12, 'FontWeight', 'bold');
title('Eigenvalue Imaginary Parts', 'FontSize', 13, 'FontWeight', 'bold');
legend('\lambda_1', '\lambda_2', '\lambda_3', 'Location', 'best');

% Eigenvalue magnitude
subplot(1,3,3);
hold on; grid on; box on;
for k = 1:3
    plot(r_values, abs(eigenvalues_all(k,:)), [colors{k}, '-'], 'LineWidth', 2);
end
xlabel('Yaw Rate r (rad/s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Eigenvalue Magnitude |λ|', 'FontSize', 12, 'FontWeight', 'bold');
title('Eigenvalue Magnitudes', 'FontSize', 13, 'FontWeight', 'bold');
legend('\lambda_1', '\lambda_2', '\lambda_3', 'Location', 'best');

% Figure 2: Damping Analysis
figure('Position', [150, 150, 1200, 500]);
sgtitle('Damping Analysis vs Yaw Rate r', 'FontSize', 14, 'FontWeight', 'bold');

% Damping ratios
subplot(1,2,1);
hold on; grid on; box on;
for k = 1:3
    plot(r_values, damping_ratios(k,:), [colors{k}, '-'], 'LineWidth', 2);
end
yline(1, 'g--', 'LineWidth', 1.5, 'DisplayName', 'Critical Damping (ζ=1)');
yline(0.707, 'm--', 'LineWidth', 1.5, 'DisplayName', 'Optimal (ζ=0.707)');
xlabel('Yaw Rate r (rad/s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Damping Ratio ζ', 'FontSize', 12, 'FontWeight', 'bold');
title('Damping Ratios', 'FontSize', 13, 'FontWeight', 'bold');
legend('Mode 1', 'Mode 2', 'Mode 3', 'Critical Damping', 'Optimal Damping', 'Location', 'best');

% Natural frequencies
subplot(1,2,2);
hold on; grid on; box on;
for k = 1:3
    plot(r_values, natural_frequencies(k,:), [colors{k}, '-'], 'LineWidth', 2);
end
xlabel('Yaw Rate r (rad/s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Natural Frequency ω_n (rad/s)', 'FontSize', 12, 'FontWeight', 'bold');
title('Natural Frequencies', 'FontSize', 13, 'FontWeight', 'bold');
legend('Mode 1', 'Mode 2', 'Mode 3', 'Location', 'best');

% Figure 3: Eigenvalue Root Locus (Complex Plane)
figure('Position', [200, 200, 800, 700]);
hold on; grid on; box on;

% Plot eigenvalue trajectories in complex plane
for k = 1:3
    % Plot trajectory
    scatter(real(eigenvalues_all(k,:)), imag(eigenvalues_all(k,:)), ...
            30, r_values, 'filled');
    
    % Mark start and end
    plot(real(eigenvalues_all(k,1)), imag(eigenvalues_all(k,1)), ...
         'ko', 'MarkerSize', 12, 'MarkerFaceColor', 'g', 'LineWidth', 2);
    plot(real(eigenvalues_all(k,end)), imag(eigenvalues_all(k,end)), ...
         'ks', 'MarkerSize', 12, 'MarkerFaceColor', 'r', 'LineWidth', 2);
end

% Stability boundary
xline(0, 'r--', 'LineWidth', 2);

% Add labels
xlabel('Real Part', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Imaginary Part', 'FontSize', 13, 'FontWeight', 'bold');
title('Root Locus: Eigenvalues vs Yaw Rate r', 'FontSize', 14, 'FontWeight', 'bold');

c = colorbar;
c.Label.String = 'Yaw Rate r (rad/s)';
c.Label.FontSize = 11;
colormap('jet');

% Add legend for markers
plot(NaN, NaN, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'g', 'DisplayName', 'Start (r_{min})');
plot(NaN, NaN, 'ks', 'MarkerSize', 10, 'MarkerFaceColor', 'r', 'DisplayName', 'End (r_{max})');
legend('Location', 'best');

axis equal;
xlim([min(real(eigenvalues_all(:)))-1, 1]);

% Figure 4: Eigenvector Analysis
figure('Position', [250, 250, 1400, 400]);
sgtitle('Eigenvector Components vs Yaw Rate r', 'FontSize', 14, 'FontWeight', 'bold');

eigenvector_labels = {'v_1', 'v_2', 'v_3'};

for k = 1:3
    subplot(1,3,k);
    hold on; grid on; box on;
    
    % Plot each component magnitude
    for comp = 1:3
        v_component = squeeze(abs(eigenvectors_all(comp, k, :)));
        plot(r_values, v_component, [colors{comp}, '-'], 'LineWidth', 2);
    end
    
    xlabel('Yaw Rate r (rad/s)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('Component Magnitude', 'FontSize', 11, 'FontWeight', 'bold');
    title(['Eigenvector ', eigenvector_labels{k}], 'FontSize', 12, 'FontWeight', 'bold');
    legend('Component 1 (p)', 'Component 2 (q)', 'Component 3 (r)', 'Location', 'best');
end

% Figure 5: Gain Matrix G1 elements vs r
figure('Position', [300, 300, 1200, 500]);
sgtitle('Gain Matrix G_1 Elements vs Yaw Rate r', 'FontSize', 14, 'FontWeight', 'bold');

% Extract G1 elements
G11 = squeeze(G1_matrices(1,1,:));
G12 = squeeze(G1_matrices(1,2,:));
G21 = squeeze(G1_matrices(2,1,:));
G22 = squeeze(G1_matrices(2,2,:));

subplot(1,2,1);
hold on; grid on; box on;
plot(r_values, G11, 'b-', 'LineWidth', 2, 'DisplayName', 'G_{11}');
plot(r_values, G12, 'r-', 'LineWidth', 2, 'DisplayName', 'G_{12}');
plot(r_values, G21, 'g-', 'LineWidth', 2, 'DisplayName', 'G_{21}');
plot(r_values, G22, 'm-', 'LineWidth', 2, 'DisplayName', 'G_{22}');
xlabel('Yaw Rate r (rad/s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Gain Value', 'FontSize', 12, 'FontWeight', 'bold');
title('Top 2×2 Block of G_1', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best');

subplot(1,2,2);
hold on; grid on; box on;
% Cross-coupling gains
G13 = squeeze(G1_matrices(1,3,:));
G23 = squeeze(G1_matrices(2,3,:));
G31 = squeeze(G1_matrices(3,1,:));
G32 = squeeze(G1_matrices(3,2,:));
G33 = squeeze(G1_matrices(3,3,:));

plot(r_values, G13, 'b-', 'LineWidth', 2, 'DisplayName', 'G_{13}');
plot(r_values, G23, 'r-', 'LineWidth', 2, 'DisplayName', 'G_{23}');
plot(r_values, G31, 'g--', 'LineWidth', 2, 'DisplayName', 'G_{31}');
plot(r_values, G32, 'm--', 'LineWidth', 2, 'DisplayName', 'G_{32}');
plot(r_values, G33, 'k-', 'LineWidth', 2, 'DisplayName', 'G_{33}');
xlabel('Yaw Rate r (rad/s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Gain Value', 'FontSize', 12, 'FontWeight', 'bold');
title('Remaining Elements of G_1', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'best');

%% ========== OPEN-LOOP vs CLOSED-LOOP COMPARISON ==========
% KEY INSIGHT for A1 with skew-symmetric structure [d, r, 0; -r, d, 0; 0, 0, d_yaw]:
%   Eigenvalues are: λ₁,₂ = d ± jr,  λ₃ = d_yaw
%   - Real parts are CONSTANT (= damping coefficients)
%   - Imaginary parts CHANGE with r (= ±r for roll/pitch modes)

figure('Position', [100, 100, 1400, 600]);
sgtitle('OPEN-LOOP A_1(r) vs CLOSED-LOOP M Eigenvalues', 'FontSize', 14, 'FontWeight', 'bold');

% Open-loop eigenvalues (A1) - Real parts are CONSTANT
subplot(2,2,1);
hold on; grid on; box on;
for k = 1:3
    plot(r_values, real(A1_eigenvalues(k,:)), [colors{k}, '-'], 'LineWidth', 2);
end
yline(0, 'g--', 'LineWidth', 2);
xlabel('Yaw Rate r (rad/s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Real Part', 'FontSize', 12, 'FontWeight', 'bold');
title('OPEN-LOOP A_1(r): Real Parts are CONSTANT (= damping)', 'FontSize', 12, 'FontWeight', 'bold');
legend('\lambda_1^{OL}', '\lambda_2^{OL}', '\lambda_3^{OL}', 'Location', 'best');

subplot(2,2,2);
hold on; grid on; box on;
for k = 1:3
    plot(r_values, imag(A1_eigenvalues(k,:)), [colors{k}, '-'], 'LineWidth', 2);
end
yline(0, 'k--', 'LineWidth', 1);
xlabel('Yaw Rate r (rad/s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Imaginary Part', 'FontSize', 12, 'FontWeight', 'bold');
title('OPEN-LOOP A_1(r): Imaginary Parts CHANGE with r', 'FontSize', 12, 'FontWeight', 'bold');
legend('\lambda_1^{OL}', '\lambda_2^{OL}', '\lambda_3^{OL}', 'Location', 'best');

% Closed-loop eigenvalues (M) - THESE ARE CONSTANT
subplot(2,2,3);
hold on; grid on; box on;
for k = 1:3
    plot(r_values, real(eigenvalues_all(k,:)), [colors{k}, '-'], 'LineWidth', 2);
end
yline(0, 'g--', 'LineWidth', 2);
xlabel('Yaw Rate r (rad/s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Real Part', 'FontSize', 12, 'FontWeight', 'bold');
title('CLOSED-LOOP M: Real Parts CONSTANT (by design)', 'FontSize', 12, 'FontWeight', 'bold');
legend('\lambda_1^{CL}', '\lambda_2^{CL}', '\lambda_3^{CL}', 'Location', 'best');

subplot(2,2,4);
hold on; grid on; box on;
for k = 1:3
    plot(r_values, imag(eigenvalues_all(k,:)), [colors{k}, '-'], 'LineWidth', 2);
end
yline(0, 'k--', 'LineWidth', 1);
xlabel('Yaw Rate r (rad/s)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Imaginary Part', 'FontSize', 12, 'FontWeight', 'bold');
title('CLOSED-LOOP M: Imaginary Parts CONSTANT', 'FontSize', 12, 'FontWeight', 'bold');
legend('\lambda_1^{CL}', '\lambda_2^{CL}', '\lambda_3^{CL}', 'Location', 'best');

%% ========== OPEN-LOOP ROOT LOCUS IN COMPLEX PLANE ==========
% This clearly shows that A1 eigenvalues move vertically (constant real, varying imag)

figure('Position', [150, 150, 800, 700]);
hold on; grid on; box on;

% Plot open-loop eigenvalue trajectories
for k = 1:3
    scatter(real(A1_eigenvalues(k,:)), imag(A1_eigenvalues(k,:)), ...
            40, r_values, 'filled');
    
    % Mark start and end
    plot(real(A1_eigenvalues(k,1)), imag(A1_eigenvalues(k,1)), ...
         'ko', 'MarkerSize', 12, 'MarkerFaceColor', 'g', 'LineWidth', 2);
    plot(real(A1_eigenvalues(k,end)), imag(A1_eigenvalues(k,end)), ...
         'ks', 'MarkerSize', 12, 'MarkerFaceColor', 'r', 'LineWidth', 2);
end

% Stability boundary
xline(0, 'r--', 'LineWidth', 2);

xlabel('Real Part', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Imaginary Part', 'FontSize', 13, 'FontWeight', 'bold');
title({'OPEN-LOOP A_1(r) Root Locus', ...
       'Eigenvalues move VERTICALLY: Real part constant, Imag = ±r'}, ...
       'FontSize', 13, 'FontWeight', 'bold');

c = colorbar;
c.Label.String = 'Yaw Rate r (rad/s)';
c.Label.FontSize = 11;
colormap('jet');

% Add analytical annotation
text(damping_coeff + 0.5, max(r_values)/2, ...
     sprintf('\\lambda_{1,2} = %.2f \\pm jr', damping_coeff), ...
     'FontSize', 12, 'FontWeight', 'bold');
text(yaw_damping + 0.2, 0.5, sprintf('\\lambda_3 = %.2f', yaw_damping), ...
     'FontSize', 12, 'FontWeight', 'bold');

legend('Trajectory', 'Start (r_{min})', 'End (r_{max})', 'Location', 'best');

%% ========== ANALYTICAL VERIFICATION ==========
fprintf('\n=== ANALYTICAL EIGENVALUE FORMULAS ===\n');
fprintf('For A1 = [d, r, 0; -r, d, 0; 0, 0, d_yaw] with d=%.2f, d_yaw=%.2f:\n', damping_coeff, yaw_damping);
fprintf('\nOpen-loop eigenvalues (analytical):\n');
fprintf('  λ₁ = d + jr = %.2f + j*r  (imag part = +r)\n', damping_coeff);
fprintf('  λ₂ = d - jr = %.2f - j*r  (imag part = -r)\n', damping_coeff);
fprintf('  λ₃ = d_yaw = %.2f         (real, constant)\n', yaw_damping);
fprintf('\n→ Real parts are CONSTANT (stability margin unchanged)\n');
fprintf('→ Imaginary parts = ±r (oscillation frequency increases with yaw rate)\n');
fprintf('→ The skew-symmetric ±r coupling creates oscillatory modes!\n');

%% ========== VERIFICATION: M IS CONSTANT ==========
fprintf('\n=== VERIFICATION: CLOSED-LOOP M IS CONSTANT ===\n');
M_first = M_matrices(:,:,1);
M_last = M_matrices(:,:,end);
M_diff = norm(M_first - M_last, 'fro');
fprintf('||M(r_min) - M(r_max)|| = %.2e\n', M_diff);
if M_diff < 1e-10
    fprintf('✓ M is CONSTANT across all r values (as designed)\n');
else
    fprintf('⚠ M varies with r (unexpected!)\n');
end

fprintf('\nM matrix (same for all r):\n');
disp(M_first);

fprintf('Closed-loop eigenvalues (constant):\n');
fprintf('  λ1 = %.4f (specified: %.4f)\n', real(eigenvalues_all(1,1)), lambda1);
fprintf('  λ2 = %.4f (specified: %.4f)\n', real(eigenvalues_all(2,1)), lambda2);
fprintf('  λ3 = %.4f (constrained by yaw_damping)\n', real(eigenvalues_all(3,1)));

fprintf('\n=== G1 COMPENSATES FOR r-VARIATION ===\n');
fprintf('G1 = (M - A1)/B, so G1 changes to cancel the r terms in A1\n');
fprintf('G1(1,2) varies from %.4f to %.4f as r goes %.2f to %.2f\n', ...
        G1_matrices(1,2,1), G1_matrices(1,2,end), r_values(1), r_values(end));
fprintf('G1(2,1) varies from %.4f to %.4f as r goes %.2f to %.2f\n', ...
        G1_matrices(2,1,1), G1_matrices(2,1,end), r_values(1), r_values(end));

%% ========== PRINT KEY RESULTS TABLE ==========
fprintf('\n=== KEY RESULTS TABLE ===\n');
fprintf('%-8s | %-22s | %-22s | %-22s | %-8s\n', 'r', 'λ1', 'λ2', 'λ3', 'Stable');
fprintf('%s\n', repmat('-', 1, 90));

% Print at 10 evenly spaced r values
sample_idx = round(linspace(1, num_r, 10));
for idx = sample_idx
    eig1 = eigenvalues_all(1, idx);
    eig2 = eigenvalues_all(2, idx);
    eig3 = eigenvalues_all(3, idx);
    
    % Format eigenvalues
    if abs(imag(eig1)) < 1e-10
        eig1_str = sprintf('%.4f', real(eig1));
    else
        eig1_str = sprintf('%.4f%+.4fj', real(eig1), imag(eig1));
    end
    if abs(imag(eig2)) < 1e-10
        eig2_str = sprintf('%.4f', real(eig2));
    else
        eig2_str = sprintf('%.4f%+.4fj', real(eig2), imag(eig2));
    end
    if abs(imag(eig3)) < 1e-10
        eig3_str = sprintf('%.4f', real(eig3));
    else
        eig3_str = sprintf('%.4f%+.4fj', real(eig3), imag(eig3));
    end
    
    stable_str = 'Yes';
    if ~stability_flag(idx)
        stable_str = 'NO';
    end
    
    fprintf('%-8.2f | %-22s | %-22s | %-22s | %-8s\n', ...
            r_values(idx), eig1_str, eig2_str, eig3_str, stable_str);
end

%% ========== SAVE RESULTS ==========
results.r_values = r_values;
% Closed-loop
results.eigenvalues_CL = eigenvalues_all;
results.eigenvectors_CL = eigenvectors_all;
results.damping_ratios = damping_ratios;
results.natural_frequencies = natural_frequencies;
results.M_matrices = M_matrices;
results.G1_matrices = G1_matrices;
% Open-loop
results.eigenvalues_OL = A1_eigenvalues;
results.eigenvectors_OL = A1_eigenvectors;
results.A1_matrices = A1_matrices;
% Other
results.w_des_all = w_des_all;
results.stability_flag = stability_flag;
results.lambda1_desired = lambda1;
results.lambda2_desired = lambda2;
results.B = B;

save('eigen_sweep_r_results.mat', 'results');
fprintf('\nResults saved to eigen_sweep_r_results.mat\n');

%% ========== INTERACTIVE QUERY FUNCTION ==========
fprintf('\n=== HOW TO USE RESULTS ===\n');
fprintf('To query results for a specific r value:\n');
fprintf('  r_query = 5;  %% Your desired r value\n');
fprintf('  [~, idx] = min(abs(r_values - r_query));\n');
fprintf('  A1_at_r = A1_matrices(:,:,idx);      %% Open-loop (changes with r)\n');
fprintf('  eigs_OL = A1_eigenvalues(:,idx);     %% Open-loop eigenvalues (change)\n');
fprintf('  M_at_r = M_matrices(:,:,idx);        %% Closed-loop (CONSTANT)\n');
fprintf('  eigs_CL = eigenvalues_all(:,idx);    %% Closed-loop eigenvalues (CONSTANT)\n');
fprintf('  G1_at_r = G1_matrices(:,:,idx);      %% Gains (change to compensate)\n');

%% ========== ANALYSIS SUMMARY ==========
fprintf('\n========================================\n');
fprintf('         ANALYSIS SUMMARY\n');
fprintf('========================================\n');
fprintf('System: Angular rate dynamics with yaw coupling\n');
fprintf('A1(r) = [%.2f,  r,  0; -r, %.2f, 0; 0, 0, %.2f]\n', damping_coeff, damping_coeff, yaw_damping);
fprintf('\n*** KEY INSIGHT ***\n');
fprintf('OPEN-LOOP A1(r): Eigenvalues CHANGE with r\n');
fprintf('  - The ±r coupling terms create r-dependent dynamics\n');
fprintf('  - At high r, open-loop has complex eigenvalues (oscillatory)\n');
fprintf('CLOSED-LOOP M: Eigenvalues are CONSTANT (by design!)\n');
fprintf('  - M is computed from desired eigenvalues λ1, λ2\n');
fprintf('  - M does NOT depend on r\n');
fprintf('  - G1 = (M - A1)/B absorbs all r-dependence\n');
fprintf('\nDesired eigenvalues (specified):\n');
fprintf('  λ1 = %.4f (chosen)\n', lambda1);
fprintf('  λ2 = %.4f (chosen)\n', lambda2);
fprintf('  λ3 = %.4f (constrained by yaw_damping)\n', yaw_damping);
fprintf('\nYaw rate sweep: r ∈ [%.2f, %.2f] rad/s\n', r_values(1), r_values(end));
fprintf('Control effectiveness: B = %.2f\n', B);
fprintf('\nw_des(r) = [0; 0; r] (yaw rate tracking)\n');
fprintf('========================================\n');
