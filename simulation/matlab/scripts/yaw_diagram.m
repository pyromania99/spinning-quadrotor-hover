% filepath: c:\Users\anike\Desktop\RESEARCH\Matlab\yaw_diagram.m

% Generate circle at 30 degrees to horizontal with radial sampling
circle_angle = 10 * pi/180; % 30 degrees in radians

% Create radial and angular sampling for surface
n_radial = 30; % Number of radial divisions
n_angular = 100; % Number of angular divisions

% Radial coordinates (0 to 1)
r = linspace(0, 1, n_radial);
% Angular coordinates (0 to 2π)
theta = linspace(0, 2*pi, n_angular);

% Create meshgrid for surface
[R, THETA] = meshgrid(r, theta);

% Convert to Cartesian coordinates on tilted circle plane
x_circle = R .* cos(THETA) * cos(circle_angle);
y_circle = R .* sin(THETA);
z_circle = R .* cos(THETA) * sin(circle_angle);

figure;
hold on;
axis equal;

% Define circle's axis direction (perpendicular to circle plane)
axis_direction = [-sin(circle_angle), 0, cos(circle_angle)];

% Calculate all magnitudes first to determine color scale
all_magnitudes = [];
for i = 1:size(x_circle, 1)
    for j = 1:size(x_circle, 2)
        if R(i, j) < 0.05
            continue;
        end
        x_pt = x_circle(i, j);
        y_pt = y_circle(i, j);
        angle_from_horizontal = atan2(y_pt, x_pt);
        magnitude = cos(angle_from_horizontal)*R(i,j);
        all_magnitudes = [all_magnitudes, magnitude];
    end
end

% Find min and max magnitudes for color scaling
min_mag = min(all_magnitudes);
max_mag = max(all_magnitudes);

% For each point on the surface, draw projection parallel to circle's axis
for i = 1:size(x_circle, 1)
    for j = 1:size(x_circle, 2)
        x_pt = x_circle(i, j);
        y_pt = y_circle(i, j);
        z_pt = z_circle(i, j);
        
        % Skip center point to avoid division by zero
        if R(i, j) < 0.05
            continue;
        end
        
        % Calculate angle from horizontal (in xy plane)
        angle_from_horizontal = atan2(y_pt, x_pt);
        magnitude = cos(angle_from_horizontal)*R(i,j) ; % Scale projection length
        
        % End point of projection (parallel to circle's axis)
        x_end = x_pt + magnitude * axis_direction(1);
        y_end = y_pt + magnitude * axis_direction(2);
        z_end = z_pt + magnitude * axis_direction(3);
        
        % Calculate normalized magnitude for color mapping
        norm_mag = (magnitude - min_mag) / (max_mag - min_mag);
        
        % Create gradient color based on magnitude
        color = [1-norm_mag, norm_mag, 0];

        
        % Draw projection line
        plot3([x_pt, x_end], [y_pt, y_end], [z_pt, z_end], 'Color', color, 'LineWidth', 1);
        plot3(x_end, y_end, z_end, 'o', 'Color', color, 'MarkerSize', 2);
    end
end

% Plot the boundary circle
theta_boundary = linspace(0, 2*pi, 100);
x_boundary = cos(theta_boundary) * cos(circle_angle);
y_boundary = sin(theta_boundary);
z_boundary = cos(theta_boundary) * sin(circle_angle);
plot3(x_boundary, y_boundary, z_boundary, 'k-', 'LineWidth', 3);

% Plot the tilted disk surface
surf(x_circle, y_circle, z_circle, 'FaceAlpha', 0.3, 'EdgeColor', 'none', 'FaceColor', [0.7 0.7 1]);

grid on;
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Tilted Disk Surface with Radial Cosine Projections');
view(3); % Set 3D view
xlim([-1.5 1.5]);
ylim([-1.5 1.5]);
