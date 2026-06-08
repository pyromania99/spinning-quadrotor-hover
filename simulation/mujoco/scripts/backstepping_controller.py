"""
Backstepping Controller for Drone Attitude/Altitude Control

This implements the control law from the paper:
    u_{d,1} = B_2^{-1}(-K_2 * x̃_2 - A_2 * x_2 - D_2 - (A_1^T - K̇) * x̃_1 + G_3 * x_2)

State definitions:
    x_1 = [n_{3,x}, n_{3,y}]^T  (3rd column of R_eb, x and y components)
    x_2 = [v_z, p, q]^T         (vertical velocity, roll rate, pitch rate)

The rotation matrix R_eb columns are [n_1, n_2, n_3] where n_3 points in body z-direction.
"""

import numpy as np


class BacksteppingController:
    """Backstepping controller for drone altitude and attitude control."""
    
    def __init__(self, mass, Jx, Jy, Jz, g=9.81):
        """Initialize controller with drone parameters.
        
        Args:
            mass: Drone mass (kg)
            Jx: Moment of inertia about x-axis (kg*m^2)
            Jy: Moment of inertia about y-axis (kg*m^2)
            Jz: Moment of inertia about z-axis (kg*m^2)
            g: Gravitational acceleration (m/s^2)
        """
        self.m = mass
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.g = g
        
        # Controller gains
        self.k_n3 = 2.0      # Gain for K matrix (x_1 tracking)
        self.k_vz = 8.0      # Gain for v_z in K_2
        self.k_omega = 15.0  # Gain for p, q in K_2
        
        # Build constant matrices
        self._build_constant_matrices()
        
        # Previous states for derivative estimation
        self.prev_A = 0.0
        self.prev_B = 0.0
        self.prev_C = 0.0
        self.prev_D = 0.0
        self.prev_time = None
        
    def _build_constant_matrices(self):
        """Build constant matrices K_2, D_2."""
        # K_2 = diag(k_vz, k_omega, k_omega) - gains for x_2 tracking
        self.K_2 = np.diag([self.k_vz, self.k_omega, self.k_omega])
        
        # D_2 = [g, 0, 0]^T - gravity term
        self.D_2 = np.array([self.g, 0.0, 0.0])
    
    def _build_B2(self, n3z):
        """Build B_2 matrix.
        
        B_2 = [-n_{3,z}/m,  0,       0    ]
              [0,           1/Jx,    0    ]
              [0,           0,       1/Jy ]
        """
        return np.array([
            [-n3z / self.m, 0.0,        0.0],
            [0.0,           1.0/self.Jx, 0.0],
            [0.0,           0.0,        1.0/self.Jy]
        ])
    
    def _build_A1(self, C, D, A, B):
        """Build A_1 matrix from rotation matrix elements.
        
        A_1 = [0, -C,  A]
              [0, -D,  B]
        
        Args:
            C, D, A, B: Elements of rotation matrix R_eb
        """
        return np.array([
            [0.0, -C, A],
            [0.0, -D, B]
        ])
    
    def _build_A2(self, r):
        """Build A_2 matrix.
        
        A_2 = [0,   0,                           0                      ]
              [0,   0,                          -(Jz-Jy)/Jx * r         ]
              [0,  -(Jx-Jz)/Jy * r,              0                      ]
        
        Args:
            r: Yaw rate (body z angular velocity)
        """
        return np.array([
            [0.0, 0.0,                              0.0],
            [0.0, 0.0,                              -(self.Jz - self.Jy) / self.Jx * r],
            [0.0, -(self.Jx - self.Jz) / self.Jy * r, 0.0]
        ])
    
    def _build_K(self, A, B, C, D):
        """Build K matrix for backstepping.
        
        K = k_n3 * [0,  0 ]
                   [-B, A ]
                   [-D, C ]
        """
        return self.k_n3 * np.array([
            [0.0,  0.0],
            [-B,   A],
            [-D,   C]
        ])
    
    def _build_G3(self, n3z):
        """Build G_3 matrix.
        
        G_3 = [0, 0]
              [0, G]
        
        where G = -n_{3,z} * k_n3 * I_2
        """
        G = -n3z * self.k_n3 * np.eye(2)
        G_3 = np.zeros((3, 2))
        G_3[1:3, :] = G
        return G_3
    
    def _compute_K_dot(self, A, B, C, D, p, q, r, n3x, n3y, dt):
        """Compute time derivative of K matrix.
        
        K̇ = k_n3 * [0,    0  ]
                    [-Ḃ,   Ȧ  ]
                    [-Ḋ,   Ċ  ]
        
        where:
            Ȧ = Cr - q*n_{3,x}
            Ḃ = Dr - q*n_{3,y}
            Ċ = -Ar + p*n_{3,x}
            Ḋ = -Br + p*n_{3,y}
        """
        # Compute derivatives using rotation kinematics
        A_dot = C * r - q * n3x
        B_dot = D * r - q * n3y
        C_dot = -A * r + p * n3x
        D_dot = -B * r + p * n3y
        
        K_dot = self.k_n3 * np.array([
            [0.0,     0.0],
            [-B_dot,  A_dot],
            [-D_dot,  C_dot]
        ])
        
        return K_dot
    
    def compute_control(self, R_eb, v_z, omega, x1_d, x2_d, dt):
        """Compute the control input using backstepping.
        
        Args:
            R_eb: 3x3 rotation matrix from body to earth frame
            v_z: Vertical velocity in earth frame
            omega: Body angular velocities [p, q, r]
            x1_d: Desired x_1 = [n_{3,x,d}, n_{3,y,d}]^T
            x2_d: Desired x_2 = [v_{z,d}, p_d, q_d]^T
            dt: Time step
            
        Returns:
            u: Control input [F_z, tau_x, tau_y] (thrust and torques)
        """
        # Extract rotation matrix elements
        # R_eb = [n_1, n_2, n_3] where each n_i is a column
        A = R_eb[0, 0]
        B = R_eb[1, 0]
        C = R_eb[0, 1]
        D = R_eb[1, 1]
        E = R_eb[2, 0]
        F = R_eb[2, 1]
        n3x = R_eb[0, 2]
        n3y = R_eb[1, 2]
        n3z = R_eb[2, 2]
        
        # Extract body rates
        p, q, r = omega
        
        # Build state vectors
        x_1 = np.array([n3x, n3y])
        x_2 = np.array([v_z, p, q])
        
        # Compute state errors
        x_1_tilde = x_1 - x1_d
        x_2_tilde = x_2 - x2_d
        
        # Build matrices
        A1 = self._build_A1(C, D, A, B)  # 2x3
        A2 = self._build_A2(r)           # 3x3
        B2 = self._build_B2(n3z)         # 3x3
        K = self._build_K(A, B, C, D)    # 3x2
        G3 = self._build_G3(n3z)         # 3x2
        
        # Compute K_dot
        K_dot = self._compute_K_dot(A, B, C, D, p, q, r, n3x, n3y, dt)
        
        # Store current values for next iteration
        self.prev_A = A
        self.prev_B = B
        self.prev_C = C
        self.prev_D = D
        
        # Compute control law:
        # u_{d,1} = B_2^{-1}(-K_2 * x̃_2 - A_2 * x_2 - D_2 - (A_1^T - K̇) * x̃_1 + G_3 * x_2)
        
        # Term 1: -K_2 * x̃_2
        term1 = -self.K_2 @ x_2_tilde
        
        # Term 2: -A_2 * x_2
        term2 = -A2 @ x_2
        
        # Term 3: -D_2
        term3 = -self.D_2
        
        # Term 4: -(A_1^T - K̇) * x̃_1
        # Note: A_1 is 2x3, so A_1^T is 3x2
        # K̇ is 3x2
        term4 = -(A1.T - K_dot) @ x_1_tilde
        
        # Term 5: G_3 * x_2 (note: this should be G_3 @ x_1 based on dimension)
        # Looking at the dimensions: G_3 is 3x2, x_2 is 3x1
        # This seems to be a typo in the original - likely G_3 @ x_1
        # Let me check the equation again... the paper shows G_3 * x_2
        # But G_3 is 3x2 and x_2 is 3x1, so this doesn't match
        # From the structure, G_3 should multiply x_1 (2x1)
        term5 = G3 @ x_1
        
        # Compute B_2^{-1}
        # B_2 is diagonal, so inverse is straightforward
        # But n3z could be small, need to handle singularity
        eps = 1e-6
        n3z_safe = n3z if abs(n3z) > eps else eps * np.sign(n3z + eps)
        
        B2_inv = np.array([
            [-self.m / n3z_safe, 0.0,       0.0],
            [0.0,                self.Jx,   0.0],
            [0.0,                0.0,       self.Jy]
        ])
        
        # Combine all terms
        inner = term1 + term2 + term3 + term4 + term5
        
        # Compute control
        u = B2_inv @ inner
        
        return u
    
    def compute_desired_states(self, target_pos, current_pos, current_vel, v_z_d=0.0, kp_pos=1.0, kd_pos=0.5):
        """Compute desired states for position tracking.
        
        For altitude control:
            v_{z,d} is computed from position error
        
        For attitude:
            x_{1,d} = [n_{3,x,d}, n_{3,y,d}]^T is computed to produce desired
            horizontal accelerations
            
        Args:
            target_pos: Desired position [x, y, z]
            current_pos: Current position [x, y, z]
            current_vel: Current velocity [vx, vy, vz]
            v_z_d: Desired vertical velocity (default 0)
            kp_pos: Position proportional gain
            kd_pos: Position derivative gain
            
        Returns:
            x1_d: Desired [n_{3,x,d}, n_{3,y,d}]
            x2_d: Desired [v_{z,d}, p_d, q_d]
        """
        # Position error
        pos_error = target_pos - current_pos
        
        # Desired accelerations (PD control)
        ax_d = kp_pos * pos_error[0] - kd_pos * current_vel[0]
        ay_d = kp_pos * pos_error[1] - kd_pos * current_vel[1]
        az_d = kp_pos * pos_error[2] - kd_pos * current_vel[2]
        
        # Desired vertical velocity from altitude error
        v_z_des = kp_pos * pos_error[2] - kd_pos * (current_vel[2] - v_z_d)
        
        # Desired n_3 components (assuming small angles)
        # For hover, n_3 ≈ [0, 0, 1] (pointing up)
        # To generate horizontal acceleration:
        #   a_x ≈ g * n_{3,x}
        #   a_y ≈ g * n_{3,y}
        # So:
        n3x_d = ax_d / self.g
        n3y_d = ay_d / self.g
        
        # Saturate to prevent excessive tilt
        max_tilt = 0.5  # corresponds to ~30 deg
        n3x_d = np.clip(n3x_d, -max_tilt, max_tilt)
        n3y_d = np.clip(n3y_d, -max_tilt, max_tilt)
        
        # Desired angular rates (typically zero for position hold)
        p_d = 0.0
        q_d = 0.0
        
        x1_d = np.array([n3x_d, n3y_d])
        x2_d = np.array([v_z_des, p_d, q_d])
        
        return x1_d, x2_d


def quat_to_rotation_matrix(quat):
    """Convert quaternion to rotation matrix.
    
    Args:
        quat: Quaternion [w, x, y, z] (MuJoCo convention)
        
    Returns:
        R: 3x3 rotation matrix (body to world)
    """
    w, x, y, z = quat
    
    # Rotation matrix from body to world frame
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [2*(x*z - w*y),         2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    
    return R


# Test the controller
if __name__ == "__main__":
    # Create controller with example parameters
    controller = BacksteppingController(
        mass=1.2,
        Jx=2.0,
        Jy=2.0,
        Jz=4.0,
        g=9.81
    )
    
    # Test with identity rotation (hover)
    R_eb = np.eye(3)
    v_z = 0.0
    omega = np.array([0.0, 0.0, 0.0])
    x1_d = np.array([0.0, 0.0])  # Want n_3 to point up
    x2_d = np.array([0.0, 0.0, 0.0])  # Want zero velocity
    
    u = controller.compute_control(R_eb, v_z, omega, x1_d, x2_d, dt=0.001)
    
    print("Backstepping Controller Test")
    print("=" * 40)
    print(f"At hover (identity R, zero velocity):")
    print(f"  Control: F_z = {u[0]:.3f} N, tau_x = {u[1]:.3f} Nm, tau_y = {u[2]:.3f} Nm")
    print(f"  Expected F_z ≈ m*g = {1.2 * 9.81:.3f} N")
    
    # Test with tilted attitude
    from scipy.spatial.transform import Rotation
    
    # 10 degree roll
    roll = np.deg2rad(10)
    R_tilted = Rotation.from_euler('x', roll).as_matrix()
    
    u_tilted = controller.compute_control(R_tilted, v_z, omega, x1_d, x2_d, dt=0.001)
    
    print(f"\nWith 10° roll:")
    print(f"  Control: F_z = {u_tilted[0]:.3f} N, tau_x = {u_tilted[1]:.3f} Nm, tau_y = {u_tilted[2]:.3f} Nm")
