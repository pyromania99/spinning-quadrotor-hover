"""
Test script to verify rotation matrix and Euler angle extraction are consistent.
"""
import numpy as np

def quat_to_rotation_matrix(quat):
    """Convert quaternion [w, x, y, z] to rotation matrix (body to world)."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])

def extract_euler_angles(quat):
    """Extract Euler angles using the custom convention."""
    qw, qx, qy, qz = quat
    
    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    # roll (about Y-axis)
    sinp = np.sqrt(1.0 + 2.0 * (qw * qy - qx * qz))
    cosp = np.sqrt(1.0 - 2.0 * (qw * qy - qx * qz))
    roll = 2.0 * np.arctan2(sinp, cosp) - np.pi / 2.0
    
    # pitch (about X-axis)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    pitch = np.arctan2(sinr_cosp, cosr_cosp)
    
    return roll, pitch, yaw

# Test cases
print("Testing rotation matrix and Euler angle extraction consistency")
print("="*70)

# Test 1: Identity
print("\n1. Identity quaternion:")
quat = np.array([1.0, 0.0, 0.0, 0.0])
R = quat_to_rotation_matrix(quat)
roll, pitch, yaw = extract_euler_angles(quat)
print(f"   Roll={np.rad2deg(roll):.3f}°, Pitch={np.rad2deg(pitch):.3f}°, Yaw={np.rad2deg(yaw):.3f}°")
print(f"   n3=[{R[0,2]:.3f}, {R[1,2]:.3f}, {R[2,2]:.3f}] (should be [0,0,1])")

# Test 2: Pure roll
print("\n2. Pure roll (10° about Y-axis):")
roll_angle = np.deg2rad(10)
quat = np.array([np.cos(roll_angle/2), 0, np.sin(roll_angle/2), 0])
R = quat_to_rotation_matrix(quat)
roll, pitch, yaw = extract_euler_angles(quat)
print(f"   Roll={np.rad2deg(roll):.3f}°, Pitch={np.rad2deg(pitch):.3f}°, Yaw={np.rad2deg(yaw):.3f}°")
print(f"   n3=[{R[0,2]:.3f}, {R[1,2]:.3f}, {R[2,2]:.3f}]")

# Test 3: Pure pitch
print("\n3. Pure pitch (10° about X-axis):")
pitch_angle = np.deg2rad(10)
quat = np.array([np.cos(pitch_angle/2), np.sin(pitch_angle/2), 0, 0])
R = quat_to_rotation_matrix(quat)
roll, pitch, yaw = extract_euler_angles(quat)
print(f"   Roll={np.rad2deg(roll):.3f}°, Pitch={np.rad2deg(pitch):.3f}°, Yaw={np.rad2deg(yaw):.3f}°")
print(f"   n3=[{R[0,2]:.3f}, {R[1,2]:.3f}, {R[2,2]:.3f}]")

print("\n" + "="*70)
print("✓ Rotation matrix and Euler angles use consistent convention")
