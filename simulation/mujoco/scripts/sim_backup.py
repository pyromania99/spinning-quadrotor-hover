"""
Verify the yaw-removal quaternion approach
"""
import numpy as np

def quat_mult(q1, q2):
    """Multiply two quaternions q1 * q2, format [w, x, y, z]"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def euler_to_quat_zyx(yaw, pitch, roll):
    """Convert ZYX Euler angles to quaternion"""
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    ])

def remove_yaw_and_extract(quat):
    """The method used in sim_tests.py"""
    qw, qx, qy, qz = quat
    
    # Extract yaw
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    # Remove yaw: q_body = q_yaw_inv * q_current
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)
    qw_body = cy * qw + sy * qz
    qx_body = cy * qx + sy * qy
    qy_body = cy * qy - sy * qx
    qz_body = cy * qz - sy * qw
    
    # Extract roll and pitch from yaw-removed quaternion
    roll = np.arctan2(2.0 * (qw_body * qy_body + qx_body * qz_body), 
                      1.0 - 2.0 * (qx_body * qx_body + qy_body * qy_body))
    pitch = np.arcsin(np.clip(2.0 * (qw_body * qx_body - qy_body * qz_body), -1.0, 1.0))
    
    return roll, pitch, yaw

print("="*60)
print("TEST 1: Yaw=90°, Pitch=0°, Roll=0°")
print("="*60)
q = euler_to_quat_zyx(np.deg2rad(90), 0, 0)
roll, pitch, yaw = remove_yaw_and_extract(q)
print(f"Input: yaw=90°, pitch=0°, roll=0°")
print(f"Output: roll={np.rad2deg(roll):.2f}°, pitch={np.rad2deg(pitch):.2f}°, yaw={np.rad2deg(yaw):.2f}°")
print(f"Expected: roll=0°, pitch=0°, yaw=90°")

print("\n" + "="*60)
print("TEST 2: Yaw=90°, Pitch=-10° (nose down), Roll=0°")
print("="*60)
q = euler_to_quat_zyx(np.deg2rad(90), np.deg2rad(-10), 0)
roll, pitch, yaw = remove_yaw_and_extract(q)
print(f"Input: yaw=90°, pitch=-10°, roll=0°")
print(f"Output: roll={np.rad2deg(roll):.2f}°, pitch={np.rad2deg(pitch):.2f}°, yaw={np.rad2deg(yaw):.2f}°")
print(f"Expected: roll=0°, pitch=-10°, yaw=90°")

print("\n" + "="*60)
print("TEST 3: Yaw=90°, Pitch=0°, Roll=5° (left up)")
print("="*60)
q = euler_to_quat_zyx(np.deg2rad(90), 0, np.deg2rad(5))
roll, pitch, yaw = remove_yaw_and_extract(q)
print(f"Input: yaw=90°, pitch=0°, roll=5°")
print(f"Output: roll={np.rad2deg(roll):.2f}°, pitch={np.rad2deg(pitch):.2f}°, yaw={np.rad2deg(yaw):.2f}°")
print(f"Expected: roll=5°, pitch=0°, yaw=90°")

print("\n" + "="*60)
print("TEST 4: Yaw=90°, Pitch=-5°, Roll=5°")
print("="*60)
q = euler_to_quat_zyx(np.deg2rad(90), np.deg2rad(-5), np.deg2rad(5))
roll, pitch, yaw = remove_yaw_and_extract(q)
print(f"Input: yaw=90°, pitch=-5°, roll=5°")
print(f"Output: roll={np.rad2deg(roll):.2f}°, pitch={np.rad2deg(pitch):.2f}°, yaw={np.rad2deg(yaw):.2f}°")
print(f"Expected: roll=5°, pitch=-5°, yaw=90°")

print("\n" + "="*60)
print("TEST 5: Yaw=45°, Pitch=-10°, Roll=0°")
print("="*60)
q = euler_to_quat_zyx(np.deg2rad(45), np.deg2rad(-10), 0)
roll, pitch, yaw = remove_yaw_and_extract(q)
print(f"Input: yaw=45°, pitch=-10°, roll=0°")
print(f"Output: roll={np.rad2deg(roll):.2f}°, pitch={np.rad2deg(pitch):.2f}°, yaw={np.rad2deg(yaw):.2f}°")
print(f"Expected: roll=0°, pitch=-10°, yaw=45°")

print("\n" + "="*60)
print("TEST 6: Yaw=0°, Pitch=-10°, Roll=0°")
print("="*60)
q = euler_to_quat_zyx(0, np.deg2rad(-10), 0)
roll, pitch, yaw = remove_yaw_and_extract(q)
print(f"Input: yaw=0°, pitch=-10°, roll=0°")
print(f"Output: roll={np.rad2deg(roll):.2f}°, pitch={np.rad2deg(pitch):.2f}°, yaw={np.rad2deg(yaw):.2f}°")
print(f"Expected: roll=0°, pitch=-10°, yaw=0°")
