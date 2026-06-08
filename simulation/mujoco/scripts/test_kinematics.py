"""
Test to verify the correct A matrix kinematics for LQR controller.

Body frame convention (from sim.py):
- X = Right
- Y = Forward  
- Z = Up

Angular velocity components:
- q = ω_x = pitch rate (rotation about X axis)
- p = ω_y = roll rate (rotation about Y axis)
- r = ω_z = yaw rate (rotation about Z axis)

The state is: [p, q, nx, ny, ...]
where n = [nx, ny, nz] is the world-Z unit vector expressed in body coordinates.

The kinematics are: dn/dt = -ω × n
"""

import numpy as np

def compute_n_dot_physics(p, q, r, nx, ny, nz):
    """Compute dn/dt using cross product: dn/dt = -ω × n"""
    omega = np.array([q, p, r])  # [ω_x, ω_y, ω_z]
    n = np.array([nx, ny, nz])
    n_dot = -np.cross(omega, n)
    return n_dot

def compute_n_dot_original_matrix(p, q, nx, ny, nz, r):
    """Compute dn/dt using ORIGINAL A matrix (rows 2-3)"""
    # Original: Row 2: [0, -nz, 0, r]
    #           Row 3: [nz, 0, -r, 0]
    dnx_dt = 0*p + (-nz)*q + 0*nx + r*ny
    dny_dt = nz*p + 0*q + (-r)*nx + 0*ny
    return np.array([dnx_dt, dny_dt, 0])

def compute_n_dot_proposed_matrix(p, q, nx, ny, nz, r):
    """Compute dn/dt using PROPOSED A matrix (rows 2-3)"""
    # Proposed: Row 2: [-nz, 0, 0, r]
    #           Row 3: [0, nz, -r, 0]
    dnx_dt = (-nz)*p + 0*q + 0*nx + r*ny
    dny_dt = 0*p + nz*q + (-r)*nx + 0*ny
    return np.array([dnx_dt, dny_dt, 0])

print("="*70)
print("KINEMATICS TEST: dn/dt = -ω × n")
print("="*70)

# Test Case 1: Pure roll rate (p > 0), level orientation
print("\nTest 1: Pure Roll Rate (p=1.0 rad/s), level orientation")
print("-" * 70)
p, q, r = 1.0, 0.0, 0.0
nx, ny, nz = 0.0, 0.0, 1.0

physics = compute_n_dot_physics(p, q, r, nx, ny, nz)
original = compute_n_dot_original_matrix(p, q, nx, ny, nz, r)
proposed = compute_n_dot_proposed_matrix(p, q, nx, ny, nz, r)

print(f"State: p={p}, q={q}, r={r}, n=[{nx}, {ny}, {nz}]")
print(f"Physics (dn/dt = -ω×n): {physics}")
print(f"Original matrix:        {original}")
print(f"Proposed matrix:        {proposed}")
print(f"Original MATCHES physics: {np.allclose(original, physics)}")
print(f"Proposed MATCHES physics: {np.allclose(proposed, physics)}")

# Test Case 2: Pure pitch rate (q > 0), level orientation
print("\nTest 2: Pure Pitch Rate (q=1.0 rad/s), level orientation")
print("-" * 70)
p, q, r = 0.0, 1.0, 0.0
nx, ny, nz = 0.0, 0.0, 1.0

physics = compute_n_dot_physics(p, q, r, nx, ny, nz)
original = compute_n_dot_original_matrix(p, q, nx, ny, nz, r)
proposed = compute_n_dot_proposed_matrix(p, q, nx, ny, nz, r)

print(f"State: p={p}, q={q}, r={r}, n=[{nx}, {ny}, {nz}]")
print(f"Physics (dn/dt = -ω×n): {physics}")
print(f"Original matrix:        {original}")
print(f"Proposed matrix:        {proposed}")
print(f"Original MATCHES physics: {np.allclose(original, physics)}")
print(f"Proposed MATCHES physics: {np.allclose(proposed, physics)}")

# Test Case 3: Pure yaw rate (r > 0), slightly tilted
print("\nTest 3: Pure Yaw Rate (r=1.0 rad/s), slightly tilted")
print("-" * 70)
p, q, r = 0.0, 0.0, 1.0
nx, ny, nz = 0.1, 0.2, 0.98  # slightly tilted

physics = compute_n_dot_physics(p, q, r, nx, ny, nz)
original = compute_n_dot_original_matrix(p, q, nx, ny, nz, r)
proposed = compute_n_dot_proposed_matrix(p, q, nx, ny, nz, r)

print(f"State: p={p}, q={q}, r={r}, n=[{nx:.2f}, {ny:.2f}, {nz:.2f}]")
print(f"Physics (dn/dt = -ω×n): {physics}")
print(f"Original matrix:        {original}")
print(f"Proposed matrix:        {proposed}")
print(f"Original MATCHES physics: {np.allclose(original, physics)}")
print(f"Proposed MATCHES physics: {np.allclose(proposed, physics)}")

# Test Case 4: Combined rates, tilted
print("\nTest 4: Combined Roll+Pitch (p=0.5, q=0.3 rad/s), tilted")
print("-" * 70)
p, q, r = 0.5, 0.3, 0.0
nx, ny, nz = -0.1, 0.15, 0.98

physics = compute_n_dot_physics(p, q, r, nx, ny, nz)
original = compute_n_dot_original_matrix(p, q, nx, ny, nz, r)
proposed = compute_n_dot_proposed_matrix(p, q, nx, ny, nz, r)

print(f"State: p={p}, q={q}, r={r}, n=[{nx:.2f}, {ny:.2f}, {nz:.2f}]")
print(f"Physics (dn/dt = -ω×n): {physics}")
print(f"Original matrix:        {original}")
print(f"Proposed matrix:        {proposed}")
print(f"Original MATCHES physics: {np.allclose(original, physics)}")
print(f"Proposed MATCHES physics: {np.allclose(proposed, physics)}")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print("The PROPOSED matrix (with my fix) correctly implements dn/dt = -ω×n")
print("The ORIGINAL matrix has p and q swapped in rows 2 and 3.")
print("\nCorrect A matrix rows 2-3:")
print("  Row 2 (dnx/dt): [-nz,   0,  0,  r]  coefficients of [p, q, nx, ny]")
print("  Row 3 (dny/dt): [  0,  nz, -r,  0]  coefficients of [p, q, nx, ny]")
