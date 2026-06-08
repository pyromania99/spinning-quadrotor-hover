 #!/usr/bin/env python3
"""
Spinning Drone Simulation with Wing Aerodynamics

Simulates a spinning drone that uses rotating wings to generate lift.
The drone starts at rest and spins up to optimal angular velocity using
tilted motors. Wing lift/drag is computed using BEMT and applied via mj_applyFT.

Key features:
- BEMT wing aerodynamics with induced velocity iteration
- Interpolated Cl, Cd, chord from XFLR5 polar data
- Induced drag calculation
- Motor tilt for torque generation
- Spin-up from rest to optimal omega
"""

import mujoco
import mujoco.viewer
import numpy as np
import json
import re
from pathlib import Path
from collections import defaultdict

# ============================================================================
# CONSTANTS
# ============================================================================
RHO = 1.225  # Air density (kg/m³)
GRAVITY = 9.81
DRONE_MASS = 0.8  # kg
DRONE_WEIGHT = DRONE_MASS * GRAVITY  # 7.85 N
MU = 1.81e-5  # Dynamic viscosity of air (kg/m·s) at 15°C

MOTOR_RADIUS = 0.4  # Motors at 0.4m from center
NUM_MOTORS = 4
NUM_BLADES = 4  # One wing per motor arm

# Wing geometry (must match optimized_design.txt range)
WING_R_INNER = 0.40  # Wing starts at motor position
WING_R_OUTER = 0.65  # Wing extends to 0.65m (matches anlysis.py)
NUM_WING_SECTIONS = 6  # Sections per wing for force application

# Controller gains
KP_SPIN = 0.05    # Spin-rate torque gain (N·m per rad/s error)
KP_ALT = 5.0      # Altitude proportional gain
KD_ALT = 20.0      # Altitude velocity damping gain
KI_ALT = 0.0       # Altitude integral gain (eliminates steady-state error)

# ============================================================================
# WING AERODYNAMICS MODEL
# ============================================================================
class WingAeroModel:
    """
    Wing aerodynamics model using BEMT with dynamic polar lookup.
    Loads XFLR5 polar files and dynamically looks up Cl/Cd based on actual
    Reynolds number at each section during flight.
    """
    
    def __init__(self, polar_file='optimized_design.txt', polar_dir='xflr_polars'):
        """Load polar data and XFLR5 polar files."""
        self.load_polar_data(polar_file)
        self.load_xflr_polars(polar_dir)
        self.setup_sections()
        
        # Induced velocity state (warm start for iteration)
        self.v_induced = 0.5  # Initial guess (m/s)
        
        # Oswald efficiency for induced drag
        self.e_oswald = 0.85
        
    def load_polar_data(self, filepath):
        """Load wing section data from optimized_design.txt.
        Uses ALL rows exactly as anlysis.py does — no radius filtering.
        """
        self.polar_radii = []
        self.polar_chords = []
        self.polar_Cl = []
        self.polar_Cd = []
        self.polar_alpha = []
        self.polar_airfoils = []  # Store airfoil name for each section
        
        script_dir = Path(__file__).parent
        full_path = script_dir / filepath
        
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) >= 7:
                        try:
                            radius = float(parts[0].strip())
                            airfoil = parts[3].strip()
                            alpha = float(parts[4].strip().replace('~', ''))
                            chord = float(parts[5].strip())
                            cl = float(parts[6].strip())
                            cl_cd = float(parts[7].strip()) if len(parts) > 7 else 30.0
                            cd = cl / cl_cd if cl_cd > 0 else 0.03
                            
                            # Use ALL sections (same as anlysis.py)
                            self.polar_radii.append(radius)
                            self.polar_chords.append(chord)
                            self.polar_Cl.append(cl)
                            self.polar_Cd.append(cd)
                            self.polar_alpha.append(alpha)
                            self.polar_airfoils.append(airfoil)
                        except (ValueError, IndexError):
                            continue
        
        self.polar_radii = np.array(self.polar_radii)
        self.polar_chords = np.array(self.polar_chords)
        self.polar_Cl = np.array(self.polar_Cl)
        self.polar_Cd = np.array(self.polar_Cd)
        self.polar_alpha = np.array(self.polar_alpha)
        # polar_airfoils stays as list of strings
        
        print(f"Loaded {len(self.polar_radii)} polar data points")
        print(f"  Radius range: {self.polar_radii.min():.2f} - {self.polar_radii.max():.2f} m")
        print(f"  Chord range:  {self.polar_chords.min():.3f} - {self.polar_chords.max():.3f} m")
        print(f"  Cl range:     {self.polar_Cl.min():.2f} - {self.polar_Cl.max():.2f}")
    
    def load_xflr_polars(self, polar_dir):
        """Load all XFLR5 polar files for dynamic Re-based lookup."""
        script_dir = Path(__file__).parent
        polar_path = script_dir / polar_dir
        
        # Dictionary: airfoil_name -> {Re: {alpha: (Cl, Cd)}}
        self.xflr_polars = defaultdict(dict)
        
        if not polar_path.exists():
            print(f"Warning: Polar directory not found: {polar_path}")
            print("  Will use fixed values from optimized_design.txt")
            return
        
        polar_files = list(polar_path.glob('*.txt'))
        print(f"\nLoading XFLR5 polar files from {polar_dir}/...")
        
        for polar_file in polar_files:
            # Parse filename: airfoil_T1_Re0.080_M0.00_N9.0.txt
            match = re.search(r'(.+?)_T1_Re([0-9.]+)_M', polar_file.name)
            if not match:
                continue
            
            airfoil = match.group(1).strip()
            re_value = float(match.group(2))  # In millions (e.g., 0.080)
            
            # Read polar data
            try:
                with open(polar_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                # Find data start (after header)
                data_start = 0
                for i, line in enumerate(lines):
                    if 'alpha' in line and 'CL' in line and 'CD' in line:
                        data_start = i + 2  # Skip header and dashes
                        break
                
                polar_data = {}
                for line in lines[data_start:]:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            alpha = float(parts[0])
                            cl = float(parts[1])
                            cd = float(parts[2])
                            polar_data[alpha] = (cl, cd)
                        except ValueError:
                            continue
                
                if polar_data:
                    self.xflr_polars[airfoil][re_value] = polar_data
                    
            except Exception as e:
                print(f"  Warning: Could not load {polar_file.name}: {e}")
                continue
        
        # Print summary
        for airfoil, re_dict in self.xflr_polars.items():
            re_values = sorted(re_dict.keys())
            print(f"  {airfoil}: {len(re_values)} Re values from {min(re_values):.3f} to {max(re_values):.3f} ×10⁶")
        
    def setup_sections(self, n_sections=30):
        """Set up wing sections by interpolating to a fine uniform grid.
        
        Matches anlysis.py's create_interpolated_blade_sections():
        linearly interpolates chord, Cl, Cd, alpha to n_sections points
        over the span so that the BEMT integral is consistent.
        """
        r_min = self.polar_radii.min()
        r_max = self.polar_radii.max()
        
        # Fine uniform grid (same as anlysis.py)
        r_fine = np.linspace(r_min, r_max, n_sections)
        
        # Linearly interpolate all properties
        chord_fine = np.interp(r_fine, self.polar_radii, self.polar_chords)
        cl_fine = np.interp(r_fine, self.polar_radii, self.polar_Cl)
        cd_fine = np.interp(r_fine, self.polar_radii, self.polar_Cd)
        alpha_fine = np.interp(r_fine, self.polar_radii, self.polar_alpha)
        
        # Nearest-neighbour for airfoil name (categorical)
        airfoil_fine = []
        for r in r_fine:
            idx = np.argmin(np.abs(self.polar_radii - r))
            airfoil_fine.append(self.polar_airfoils[idx])
        
        # Replace raw arrays with interpolated ones
        self.polar_radii = r_fine
        self.polar_chords = chord_fine
        self.polar_Cl = cl_fine
        self.polar_Cd = cd_fine
        self.polar_alpha = alpha_fine
        self.polar_airfoils = airfoil_fine
        
        # Compute dr using midpoint rule (same as anlysis.py BEMT loop)
        n = len(r_fine)
        self.section_radii = []
        self.section_dr = []
        
        for i in range(n):
            r = r_fine[i]
            if i == 0:
                dr = (r_fine[1] - r) / 2
            elif i == n - 1:
                dr = max((r - r_fine[i - 1]) / 2, 0.001)
            else:
                dr = (r_fine[i + 1] - r_fine[i - 1]) / 2
            self.section_radii.append(r)
            self.section_dr.append(dr)
        
        self.section_radii = np.array(self.section_radii)
        self.section_dr = np.array(self.section_dr)
        
        print(f"\nWing sections ({n_sections} interpolated, matches anlysis.py):")
        print(f"  Radius: {r_min:.3f} – {r_max:.3f} m, dr ≈ {self.section_dr[1]:.4f} m")
    
    def interp_chord(self, r):
        """Interpolate chord at radius r."""
        return np.interp(r, self.polar_radii, self.polar_chords)
    
    def interp_Cl(self, r):
        """Interpolate lift coefficient at radius r (fallback for fixed values)."""
        return np.interp(r, self.polar_radii, self.polar_Cl)
    
    def interp_Cd(self, r):
        """Interpolate profile drag coefficient at radius r (fallback for fixed values)."""
        return np.interp(r, self.polar_radii, self.polar_Cd)
    
    def lookup_polar(self, airfoil, Re, alpha):
        """Look up Cl and Cd from XFLR5 polars at given Re and alpha.
        
        Parameters:
        -----------
        airfoil : str
            Airfoil name (e.g., 'SD7037', 'franky')
        Re : float
            Reynolds number in millions (e.g., 0.080 for 80,000)
        alpha : float
            Angle of attack in degrees
            
        Returns:
        --------
        (Cl, Cd) : tuple of floats
        """
        # Handle airfoil name variations
        airfoil_clean = airfoil.replace(' ', '').lower()
        
        # Find matching airfoil (case-insensitive)
        airfoil_key = None
        for key in self.xflr_polars.keys():
            if key.replace(' ', '').lower() == airfoil_clean:
                airfoil_key = key
                break
        
        if airfoil_key is None or not self.xflr_polars[airfoil_key]:
            # Fallback to fixed values if polar not found
            return None, None
        
        re_dict = self.xflr_polars[airfoil_key]
        available_Re = sorted(re_dict.keys())
        
        # Find two nearest Re values for linear interpolation
        if Re <= available_Re[0]:
            # Below range - use lowest Re
            polar_data = re_dict[available_Re[0]]
            return self._interp_alpha(polar_data, alpha)
        elif Re >= available_Re[-1]:
            # Above range - use highest Re
            polar_data = re_dict[available_Re[-1]]
            return self._interp_alpha(polar_data, alpha)
        else:
            # Within range - find the two bracketing Re values
            re_low = None
            re_high = None
            for i in range(len(available_Re) - 1):
                if available_Re[i] <= Re <= available_Re[i + 1]:
                    re_low = available_Re[i]
                    re_high = available_Re[i + 1]
                    break
            
            # Safety fallback (shouldn't happen)
            if re_low is None:
                # Find closest single Re value
                idx = min(range(len(available_Re)), key=lambda i: abs(available_Re[i] - Re))
                polar_data = re_dict[available_Re[idx]]
                return self._interp_alpha(polar_data, alpha)
            
            # Get Cl, Cd at both bracketing Re values
            cl_low, cd_low = self._interp_alpha(re_dict[re_low], alpha)
            cl_high, cd_high = self._interp_alpha(re_dict[re_high], alpha)
            
            # Linear interpolation in Re
            if cl_low is None or cl_high is None:
                return None, None
            
            # Interpolate between the two closest Re values
            re_frac = (Re - re_low) / (re_high - re_low)
            cl = cl_low + re_frac * (cl_high - cl_low)
            cd = cd_low + re_frac * (cd_high - cd_low)
            
            return cl, cd
    
    def _interp_alpha(self, polar_data, alpha):
        """Interpolate Cl and Cd at given alpha from polar data dict."""
        alphas = sorted(polar_data.keys())
        
        if not alphas:
            return None, None
        
        # Clamp to available range
        alpha = np.clip(alpha, alphas[0], alphas[-1])
        
        # Find bracketing alphas
        if alpha <= alphas[0]:
            cl, cd = polar_data[alphas[0]]
            return cl, cd
        elif alpha >= alphas[-1]:
            cl, cd = polar_data[alphas[-1]]
            return cl, cd
        else:
            # Linear interpolation
            for i in range(len(alphas) - 1):
                if alphas[i] <= alpha <= alphas[i + 1]:
                    alpha_low = alphas[i]
                    alpha_high = alphas[i + 1]
                    cl_low, cd_low = polar_data[alpha_low]
                    cl_high, cd_high = polar_data[alpha_high]
                    
                    frac = (alpha - alpha_low) / (alpha_high - alpha_low)
                    cl = cl_low + frac * (cl_high - cl_low)
                    cd = cd_low + frac * (cd_high - cd_low)
                    
                    return cl, cd
        
        return None, None
    
    def find_optimal_alpha(self, airfoil, Re):
        """Find the angle of attack that gives maximum Cl/Cd (best efficiency) at given Re.
        
        Parameters:
        -----------
        airfoil : str
            Airfoil name
        Re : float
            Reynolds number in millions
            
        Returns:
        --------
        (optimal_alpha, max_cl_cd) : tuple or (None, None) if not found
        """
        # Handle airfoil name variations
        airfoil_clean = airfoil.replace(' ', '').lower()
        
        # Find matching airfoil
        airfoil_key = None
        for key in self.xflr_polars.keys():
            if key.replace(' ', '').lower() == airfoil_clean:
                airfoil_key = key
                break
        
        if airfoil_key is None or not self.xflr_polars[airfoil_key]:
            return None, None
        
        re_dict = self.xflr_polars[airfoil_key]
        available_Re = sorted(re_dict.keys())
        
        # Get polar data at closest Re
        if Re <= available_Re[0]:
            polar_data = re_dict[available_Re[0]]
        elif Re >= available_Re[-1]:
            polar_data = re_dict[available_Re[-1]]
        else:
            # Use closest single Re for efficiency search
            idx = min(range(len(available_Re)), key=lambda i: abs(available_Re[i] - Re))
            polar_data = re_dict[available_Re[idx]]
        
        # Find alpha with maximum Cl/Cd
        max_efficiency = 0
        optimal_alpha = None
        
        for alpha, (cl, cd) in polar_data.items():
            if cd > 0 and cl > 0:  # Only consider positive lift
                efficiency = cl / cd
                if efficiency > max_efficiency:
                    max_efficiency = efficiency
                    optimal_alpha = alpha
        
        return optimal_alpha, max_efficiency
    
    def compute_induced_drag(self, Cl, r, chord):
        """Compute induced drag coefficient."""
        # Local aspect ratio
        AR_local = (2 * r) / chord
        # Induced drag: Cd_i = Cl² / (π * AR * e)
        Cd_induced = Cl**2 / (np.pi * AR_local * self.e_oswald)
        return Cd_induced
    
    def tip_loss_factor(self, r):
        """Prandtl tip loss factor (simplified)."""
        r_tip = WING_R_OUTER
        return 1.0 - 0.3 * (r / r_tip)**2
    
    def compute_forces(self, omega_z, max_iterations=20, tolerance=0.01):
        """
        Compute total wing forces using BEMT with induced velocity iteration.
        
        Parameters:
        -----------
        omega_z : float
            Body yaw rate (rad/s) - positive = CCW when viewed from above
        max_iterations : int
            Max iterations for induced velocity convergence
        tolerance : float
            Convergence tolerance for induced velocity (m/s)
            
        Returns:
        --------
        dict with:
            - total_thrust: Vertical thrust from all wings (N)
            - total_torque: Drag torque opposing rotation (N·m)
            - section_forces: List of per-section forces for mj_applyFT
            - v_induced: Converged induced velocity (m/s)
        """
        omega = abs(omega_z)  # Use absolute value for calculation
        
        if omega < 0.1:
            # No significant rotation - no wing forces
            return {
                'total_thrust': 0.0,
                'total_torque': 0.0,
                'section_forces': [(r, 0.0, 0.0) for r in self.section_radii],
                'v_induced': 0.0,
                'power': 0.0,
            }
        
        # Disk area for momentum theory (use actual tip radius, same as anlysis.py)
        r_tip = self.polar_radii.max()
        A_disk = np.pi * r_tip**2
        
        # Warm start from previous timestep (better for transient sim —
        # v_i evolves smoothly, so last converged value is a better guess
        # than a fixed constant, and converges in fewer iterations)
        v_i = self.v_induced
        
        for iteration in range(max_iterations):
            thrust_total = 0.0
            torque_total = 0.0
            section_forces = []
            
            for idx, (r, dr) in enumerate(zip(self.section_radii, self.section_dr)):
                # Get section geometry
                chord = self.polar_chords[idx]
                airfoil = self.polar_airfoils[idx]
                design_alpha = self.polar_alpha[idx]  # Design angle of attack
                
                # Velocity components
                V_tan = omega * r  # Tangential from rotation
                V_axial = v_i * self.tip_loss_factor(r)  # Induced downwash
                
                # Resultant velocity
                V_resultant = np.sqrt(V_tan**2 + V_axial**2)
                
                # Calculate actual Reynolds number at this section
                Re_actual = RHO * V_resultant * chord / MU  # Full Re value
                Re_millions = Re_actual / 1e6  # Convert to millions for lookup
                
                # Inflow angle
                phi = np.arctan2(V_axial, V_tan)
                
                # Use the DESIGN alpha (blade pitch is a fixed geometric property)
                # but look up Cl/Cd at the ACTUAL flight Re for that alpha.
                # Note: find_optimal_alpha maximises Cl/Cd (efficiency) and picks
                # a much lower alpha (~4°) than the design (~10°), which cuts Cl
                # nearly in half.  The physical blade is pitched to the design
                # angle — only the Re changes in flight.
                alpha_used = design_alpha
                
                # Look up Cl and Cd from XFLR5 polars at actual Re and design alpha
                Cl, Cd_profile = self.lookup_polar(airfoil, Re_millions, alpha_used)
                
                # Fallback to fixed values if lookup fails
                if Cl is None or Cd_profile is None:
                    Cl = self.polar_Cl[idx]
                    Cd_profile = self.polar_Cd[idx]
                
                # Dynamic pressure
                q = 0.5 * RHO * V_resultant**2
                
                # Induced drag
                Cd_induced = self.compute_induced_drag(Cl, r, chord)
                Cd_total = Cd_profile + Cd_induced
                
                # Section lift and drag
                dL = q * chord * Cl * dr
                dD = q * chord * Cd_total * dr
                
                # Resolve into thrust (axial) and tangential force
                # Thrust = L*cos(φ) - D*sin(φ)  (vertical, positive up)
                # F_tan = L*sin(φ) + D*cos(φ)   (opposes rotation)
                dT = dL * np.cos(phi) - dD * np.sin(phi)
                dF_tan = dL * np.sin(phi) + dD * np.cos(phi)
                
                # Torque from tangential force
                dQ = dF_tan * r
                
                section_forces.append((r, dT, dF_tan))
                thrust_total += dT
                torque_total += dQ
            
            # Scale by number of blades
            thrust_all = thrust_total * NUM_BLADES
            torque_all = torque_total * NUM_BLADES
            
            # Update induced velocity from momentum theory
            # v_i = sqrt(T / (2 * rho * A))
            if thrust_all > 0:
                v_i_new = np.sqrt(thrust_all / (2 * RHO * A_disk))
            else:
                v_i_new = 0.0
            
            # Check convergence
            if abs(v_i_new - v_i) < tolerance:
                v_i = v_i_new
                break
            
            # Relaxation for stability (0.5/0.5 matches anlysis.py)
            v_i = 0.5 * v_i + 0.5 * v_i_new
        
        # Store for next timestep (warm start)
        self.v_induced = v_i
        
        # Power = Torque × omega
        power = torque_all * omega
        
        return {
            'total_thrust': thrust_all,
            'total_torque': torque_all,
            'section_forces': section_forces,  # Per single blade
            'v_induced': v_i,
            'power': power,
            'iterations': iteration + 1,
        }


# ============================================================================
# SIMULATION CLASS
# ============================================================================
class SpinningDroneSim:
    """Spinning drone simulation with wing aerodynamics."""
    
    def __init__(self, model_path='quad_spinning.xml'):
        # Load MuJoCo model
        script_dir = Path(__file__).parent
        self.model = mujoco.MjModel.from_xml_path(str(script_dir / model_path))
        self.data = mujoco.MjData(self.model)
        
        # Get body and site IDs
        self.quad_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "quad")
        
        # Motor body IDs (for getting tilt orientation)
        self.motor_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"motor{i+1}")
            for i in range(4)
        ]
        
        # Thrust site IDs
        self.thrust_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"thrust{i+1}")
            for i in range(4)
        ]
        
        # Wing section site IDs (6 sections × 4 wings - matches polar data rows)
        self.wing_site_ids = []
        for wing in range(1, 5):
            wing_sites = []
            for section in range(1, NUM_WING_SECTIONS + 1):
                site_name = f"wing{wing}_s{section}"
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                wing_sites.append(site_id)
            self.wing_site_ids.append(wing_sites)
        
        # Tilt actuator IDs
        self.tilt_actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"tilt{i+1}")
            for i in range(4)
        ]
        
        # Wing aerodynamics model
        self.wing_model = WingAeroModel()
        
        # Motor thrust values (will be set by controller)
        self.motor_thrusts = np.zeros(4)
        
        # Controller state
        self.target_altitude = 1.5
        self.spin_direction = 1.0  # +1 = CCW (positive yaw rate)
        
        # Altitude controller state - PID with integral for zero steady-state error
        self.altitude_error_integral = 0.0
        
        # Data logging
        self.log_data = {
            'time': [],
            'pos_x': [], 'pos_y': [], 'pos_z': [],
            'vel_x': [], 'vel_y': [], 'vel_z': [],
            'omega_z': [],
            'wing_thrust': [], 'wing_torque': [], 'wing_power': [],
            'v_induced': [],
            'motor_thrust_total': [],
            'motor_tilt_deg': [],
            'motor_1': [], 'motor_2': [], 'motor_3': [], 'motor_4': [],
        }
        
        # Wing angle offsets (direction each wing points from center)
        # Wing 1: 45°, Wing 2: 135°, Wing 3: 225°, Wing 4: 315°
        self.wing_angles = np.radians([45, 135, 225, 315])
        
        print(f"\nSimulation initialized:")
        print(f"  Quad body ID: {self.quad_body_id}")
        print(f"  Motor body IDs: {self.motor_body_ids}")
        print(f"  Thrust site IDs: {self.thrust_site_ids}")
        print(f"  Wing site IDs: {len(self.wing_site_ids)} wings × {len(self.wing_site_ids[0])} sections")
    
    def get_state(self):
        """Get current drone state."""
        pos = self.data.qpos[:3].copy()
        vel = self.data.qvel[:3].copy()
        quat = self.data.qpos[3:7].copy()
        ang_vel = self.data.qvel[3:6].copy()  # Body angular velocity
        
        return {
            'pos': pos,
            'vel': vel,
            'quat': quat,
            'ang_vel': ang_vel,
            'omega_z': ang_vel[2],  # Yaw rate (body z-axis)
        }
    
    def apply_wing_forces(self, wing_result):
        """Apply wing aerodynamic forces via mj_applyFT.
        
        Distributes the 30 BEMT sections across the 6 physical wing sites.
        """
        
        # Get body rotation matrix (body to world)
        body_xmat = self.data.xmat[self.quad_body_id].reshape(3, 3)
        
        section_forces = wing_result['section_forces']  # 30 sections from BEMT
        n_bemt_sections = len(section_forces)
        n_sites = NUM_WING_SECTIONS  # 6 physical sites per wing
        
        omega_z = self.get_state()['omega_z']
        spin_sign = np.sign(omega_z) if abs(omega_z) > 0.1 else self.spin_direction
        
        # Bin the 30 BEMT sections into 6 site groups
        sections_per_site = n_bemt_sections // n_sites
        
        # Apply forces for each wing
        for wing_idx in range(4):
            wing_angle = self.wing_angles[wing_idx]
            
            # Tangent direction (perpendicular to radial, in rotation direction)
            # For CCW rotation: tangent = (-sin(θ), cos(θ), 0)
            tangent_body = np.array([-np.sin(wing_angle), np.cos(wing_angle), 0.0])
            
            # Group BEMT sections and apply to each site
            for site_idx in range(n_sites):
                # Get the site
                site_id = self.wing_site_ids[wing_idx][site_idx]
                site_pos = self.data.site_xpos[site_id].copy()
                
                # Sum forces from multiple BEMT sections that correspond to this site
                start_idx = site_idx * sections_per_site
                end_idx = start_idx + sections_per_site if site_idx < n_sites - 1 else n_bemt_sections
                
                dT_total = 0.0
                dF_tan_total = 0.0
                for i in range(start_idx, end_idx):
                    r, dT, dF_tan = section_forces[i]
                    dT_total += dT
                    dF_tan_total += dF_tan
                
                # Force in body frame:
                # - Thrust (dT) acts upward (body +Z)
                # - Tangential force (dF_tan) opposes rotation (negative tangent direction)
                force_body = np.array([0.0, 0.0, dT_total])  # Thrust
                force_body -= dF_tan_total * spin_sign * tangent_body  # Drag opposes motion
                
                # Transform to world frame
                force_world = body_xmat @ force_body
                
                # Apply force (no additional torque, just force at point)
                torque_world = np.zeros(3)
                
                mujoco.mj_applyFT(
                    self.model,
                    self.data,
                    force_world,
                    torque_world,
                    site_pos,
                    self.quad_body_id,
                    self.data.qfrc_applied,
                )
    
    def apply_motor_forces(self, thrusts, tilt_angles_deg):
        """Apply motor thrust forces with tilt AND propeller reaction torque via mj_applyFT.
        
        Motor tilt creates tangential thrust component for yaw control:
        - Vertical component: T * cos(tilt) 
        - Tangential component: T * sin(tilt) (in rotation direction)
        
        Additionally, each propeller creates a reaction torque (all props spin CW):
        - Reaction torque: Q = K_M * T (opposes rotation, i.e., CCW yaw torque)
        - With tilt, only the vertical component contributes to yaw: Q * cos(tilt)
        
        Motor positions (X-config at r=0.4m):
        - Motor 1: 45°  → tangent = (-sin(45°), cos(45°), 0) = (-0.707, 0.707, 0)
        - Motor 2: 135° → tangent = (-sin(135°), cos(135°), 0) = (-0.707, -0.707, 0)
        - Motor 3: 225° → tangent = (-sin(225°), cos(225°), 0) = (0.707, -0.707, 0)
        - Motor 4: 315° → tangent = (-sin(315°), cos(315°), 0) = (0.707, 0.707, 0)
        """
        
        # Motor propeller reaction torque constant (from anlysis.py)
        MOTOR_KM = -0.015  # Reaction torque-to-thrust ratio (m)
        
        # Set tilt actuator controls (for visualization)
        for i in range(4):
            self.data.ctrl[self.tilt_actuator_ids[i]] = tilt_angles_deg[i]
        
        # Get body rotation matrix (body to world)
        body_xmat = self.data.xmat[self.quad_body_id].reshape(3, 3)
        
        # Motor angles in the body XY plane (X-config)
        motor_angles = np.radians([45, 135, 225, 315])
        
        for i in range(4):
            thrust = thrusts[i]
            tilt_rad = np.radians(tilt_angles_deg[i])
            site_id = self.thrust_site_ids[i]
            
            # Tangent direction for this motor (perpendicular to radial, CCW positive)
            # tangent = (-sin(angle), cos(angle), 0) in body frame
            angle = motor_angles[i]
            tangent_body = np.array([-np.sin(angle), np.cos(angle), 0.0])
            
            # Thrust components in body frame:
            # - Vertical: T * cos(tilt) along body Z
            # - Tangential: T * sin(tilt) along tangent direction (for CCW spin)
            vertical_component = thrust * np.cos(tilt_rad)
            tangential_component = thrust * np.sin(tilt_rad)
            
            force_body = np.array([0.0, 0.0, vertical_component])
            force_body += tangential_component * tangent_body * self.spin_direction
            
            # Transform to world frame
            force_world = body_xmat @ force_body
            
            # ====================================================================
            # PROPELLER REACTION TORQUE (Critical for equilibrium!)
            # ====================================================================
            # Each propeller spinning CW creates a reaction torque on the body
            # Magnitude: Q = K_M * T
            # Direction: CCW (opposes prop rotation) = +Z in body frame
            # When tilted, only vertical component contributes to yaw:
            #   Q_yaw = K_M * T * cos(tilt) 
            reaction_torque_magnitude = MOTOR_KM * thrust * np.cos(tilt_rad)
            
            # Reaction torque in body frame (CCW = +Z)
            torque_body = np.array([0.0, 0.0, reaction_torque_magnitude * self.spin_direction])
            
            # Transform to world frame
            torque_world = body_xmat @ torque_body
            
            # Application point (motor site position in world frame)
            site_pos = self.data.site_xpos[site_id].copy()
            
            # Apply force AND torque to quad body (matches sim.py)
            mujoco.mj_applyFT(
                self.model,
                self.data,
                force_world,
                torque_world,  # NOW INCLUDES REACTION TORQUE!
                site_pos,
                self.quad_body_id,
                self.data.qfrc_applied,
            )

    
    def spin_controller(self, state, wing_result):
        """
        PID altitude controller with fixed motor tilt.
        
        Uses PID (not just PD) to eliminate steady-state altitude error.
        At true hover equilibrium, z_error = 0 and the force balance is:
            motor_thrust * cos(tilt) + wing_thrust = weight
        
        The integral term ensures the controller converges to this exact
        equilibrium rather than settling with a persistent offset.
        
        Returns motor thrusts and tilt angles.
        """
        pos = state['pos']
        vel = state['vel']
        
        wing_thrust = wing_result['total_thrust']
        
        # ====================================================================
        # PID ALTITUDE CONTROL (eliminates steady-state error)
        # ====================================================================
        z_error = self.target_altitude - pos[2]
        z_vel = vel[2]
        
        # Update integral term with anti-windup clamp
        dt = self.model.opt.timestep
        self.altitude_error_integral += z_error * dt
        self.altitude_error_integral = np.clip(self.altitude_error_integral, -2.0, 2.0)
        
        # Net vertical force needed: weight minus wing contribution plus PID correction
        # Separate P+I (position tracking) from D (velocity damping)
        # D term must always be active to prevent overshoot during ramp
        pi_correction = (KP_ALT * z_error 
                        + KI_ALT * self.altitude_error_integral)
        d_correction = -KD_ALT * z_vel
        
        # Clamp P+I correction to ±50% of weight
        max_pid_force = 0.5 * DRONE_WEIGHT
        pi_correction = np.clip(pi_correction, -max_pid_force, max_pid_force)
        
        # Smooth startup ramp - only ramp P+I, D is always fully active
        RAMP_TIME = 3.0  # seconds to reach full P+I authority
        t = self.data.time
        if t < RAMP_TIME:
            ramp = (t / RAMP_TIME)**2 * (3 - 2 * t / RAMP_TIME)  # smoothstep
        else:
            ramp = 1.0
        
        vertical_needed = (DRONE_WEIGHT - wing_thrust) + pi_correction * ramp + d_correction
        
        # Ensure non-negative (can't push down)
        vertical_needed = max(vertical_needed, 0.0)
        
        # ====================================================================
        # FIXED TILT - COMPUTE THRUST
        # ====================================================================
        tilt_deg = -44.0
        tilt_rad = np.radians(tilt_deg)
        cos_tilt = np.cos(tilt_rad)
        
        # Total thrust needed from all motors
        # Vertical component: T_total * cos(tilt) = vertical_needed
        # Therefore: T_total = vertical_needed / cos(tilt)
        thrust_total = vertical_needed / cos_tilt if cos_tilt > 0.01 else 0.0
        
        # Distribute equally among motors
        thrust_per_motor = thrust_total / NUM_MOTORS
        
        # Safety clamp: max 4 N per motor (total 16 N ≈ 2× weight)
        thrust_per_motor = np.clip(thrust_per_motor, 0.0, 4.0)
        
        # All motors same thrust and tilt (symmetric spinning drone)
        thrusts = np.array([thrust_per_motor] * 4)
        tilts = np.array([tilt_deg] * 4)
        
        return thrusts, tilts
    
    def step(self):
        """Run one simulation step with wing and motor forces."""
        # Clear applied forces
        self.data.qfrc_applied[:] = 0.0
        
        # Get state
        state = self.get_state()
        omega_z = state['omega_z']
        
        # Compute wing forces
        wing_result = self.wing_model.compute_forces(omega_z)
        
        # Apply wing forces
        self.apply_wing_forces(wing_result)
        
        # Compute motor commands
        thrusts, tilts = self.spin_controller(state, wing_result)
        
        # Apply motor forces
        self.apply_motor_forces(thrusts, tilts)
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Log data
        self.log_data['time'].append(self.data.time)
        self.log_data['pos_x'].append(state['pos'][0])
        self.log_data['pos_y'].append(state['pos'][1])
        self.log_data['pos_z'].append(state['pos'][2])
        self.log_data['vel_x'].append(state['vel'][0])
        self.log_data['vel_y'].append(state['vel'][1])
        self.log_data['vel_z'].append(state['vel'][2])
        self.log_data['omega_z'].append(omega_z)
        self.log_data['wing_thrust'].append(wing_result['total_thrust'])
        self.log_data['wing_torque'].append(wing_result['total_torque'])
        self.log_data['wing_power'].append(wing_result['power'])
        self.log_data['v_induced'].append(wing_result['v_induced'])
        self.log_data['motor_thrust_total'].append(sum(thrusts))
        self.log_data['motor_tilt_deg'].append(tilts[0])
        self.log_data['motor_1'].append(thrusts[0])
        self.log_data['motor_2'].append(thrusts[1])
        self.log_data['motor_3'].append(thrusts[2])
        self.log_data['motor_4'].append(thrusts[3])
        
        return state, wing_result, thrusts, tilts
    
    def save_log(self, filename='sim_log_spinning.json'):
        """Save logged data to JSON file."""
        script_dir = Path(__file__).parent
        filepath = script_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.log_data, f)
        print(f"\nSimulation data saved to {filepath}")
    
    def run(self, duration=30.0):
        """Run simulation with viewer."""
        print("\n" + "="*60)
        print("  SPINNING DRONE SIMULATION")
        print("="*60)
        print(f"  Target altitude: {self.target_altitude} m")
        print(f"  Duration:        {duration} s")
        print("\nControls:")
        print("  - Mouse: Rotate view")
        print("  - Scroll: Zoom")
        print("  - 'Top' camera: View from above to see spin")
        print()
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Set initial camera
            viewer.cam.lookat[:] = [0, 0, 1.5]
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -30
            
            step_count = 0
            print_interval = 500  # Print status every N steps
            
            while viewer.is_running() and self.data.time < duration:
                # Run simulation step
                state, wing_result, thrusts, tilts = self.step()
                
                # Print status periodically
                if step_count % print_interval == 0:
                    omega_z = state['omega_z']
                    omega_rpm = omega_z * 60 / (2 * np.pi)
                    pos = state['pos']
                    print(f"t={self.data.time:5.1f}s | "
                          f"z={pos[2]:5.2f}m | "
                          f"ω={omega_z:5.1f} rad/s ({omega_rpm:5.0f} RPM) | "
                          f"Wing: {wing_result['total_thrust']:5.2f}N | "
                          f"Motor: {sum(thrusts):5.2f}N | "
                          f"Tilt: {tilts[0]:5.1f}°")
                
                # Sync viewer at ~60 Hz
                if step_count % 16 == 0:
                    viewer.sync()
                
                step_count += 1
        
        # Save log
        self.save_log()
        
        # Print final summary
        print("\n" + "="*60)
        print("  SIMULATION COMPLETE")
        print("="*60)
        final_omega = self.log_data['omega_z'][-1]
        final_wing = self.log_data['wing_thrust'][-1]
        final_motor = self.log_data['motor_thrust_total'][-1]
        print(f"  Final omega:      {final_omega:.1f} rad/s")
        print(f"  Final wing thrust: {final_wing:.2f} N ({100*final_wing/DRONE_WEIGHT:.0f}% of weight)")
        print(f"  Final motor thrust: {final_motor:.2f} N")
        print(f"  Final motor tilt:  {self.log_data['motor_tilt_deg'][-1]:.1f}°")


# ============================================================================
# MAIN
# ============================================================================
def main():
    sim = SpinningDroneSim()
    sim.run(duration=30.0)


if __name__ == "__main__":
    main()
