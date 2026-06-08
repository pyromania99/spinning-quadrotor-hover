#!/usr/bin/env python3
"""
Spinning Drone Motor Optimization Analysis

Analyzes a spinning drone to find optimal blade configuration that minimizes
motor thrust requirements.

System Parameters:
- Drone weight: 800g (7.85 N)
- Motors mounted at r = 0.4m from center
- All motors spin clockwise
- Baseline spin rate: 8 rad/s (at 0 motor tilt)
- Blade span: 0.20-0.60m from center

Physics Model:
1. Wing lift provides vertical force component
2. Wing drag creates aerodynamic torque (resists/assists spin)
3. Motors provide:
   - Vertical thrust component = T_motor * cos(tilt)
   - Tangential component = T_motor * sin(tilt) for torque
4. At equilibrium:
   - Lift + Motor_vertical = Weight
   - Motor_tangential * r_motor = Wing_drag_torque (steady spin)

Goal: Find blade config that maximizes lift/drag ratio to minimize motor thrust
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import re
from collections import defaultdict

# ============================================================================
# SYSTEM CONSTANTS
# ============================================================================
DRONE_WEIGHT_KG = 0.800  # 800 grams
GRAVITY = 9.81
DRONE_WEIGHT_N = DRONE_WEIGHT_KG * GRAVITY  # 7.85 N

MOTOR_RADIUS = 0.40  # Motors at 0.4m from center
NUM_MOTORS = 4  # Quad configuration
RHO = 1.225  # Air density kg/m³
MU = 1.81e-5  # Dynamic viscosity of air (kg/m·s) at 15°C

# Motor properties for power estimation
MOTOR_PROP_DIAMETER = 0.10  # 10cm propeller diameter (m)
MOTOR_EFFICIENCY = 0.65  # Typical small quadcopter motor+prop efficiency

# Motor propeller reaction torque coefficient
# Q = K_M * T  where Q is reaction torque (N·m) and T is thrust (N)
# For a small 10cm prop, typical k_M ≈ 0.01–0.02 m
# All 4 props spin the same direction (CW), so reaction torques add.
# When a motor is tilted by angle τ, its reaction torque vector tilts too:
#   yaw component = Q * cos(τ), tangent component = Q * sin(τ)
MOTOR_KM = 0.015  # Reaction torque-to-thrust ratio (m)

# Blade geometry from optimized_design.txt
BLADE_SECTIONS = None  # Will load from file

# ============================================================================
# LOAD BLADE DATA
# ============================================================================
def load_blade_data(filepath='optimized_design.txt'):
    """Load blade sections from optimized_design.txt"""
    data_rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 7:
                    try:
                        radius = float(parts[0].strip())
                        velocity = float(parts[1].strip())
                        airfoil = parts[3].strip()
                        alpha = float(parts[4].strip().replace('~', ''))
                        chord = float(parts[5].strip())
                        cl_str = parts[6].strip()
                        cl = float(cl_str) if cl_str and cl_str != '—' else 1.0
                        
                        # Get Cl/Cd ratio from column 7
                        cl_cd = float(parts[7].strip()) if len(parts) > 7 else 30.0
                        cd = cl / cl_cd if cl_cd > 0 else 0.03
                        
                        data_rows.append({
                            'radius': radius,
                            'airfoil': airfoil,
                            'alpha': alpha,
                            'chord': chord,
                            'Cl': cl,
                            'Cd': cd,
                            'Cl_Cd': cl_cd
                        })
                    except (ValueError, IndexError):
                        continue
    return pd.DataFrame(data_rows)


def load_xflr_polars(polar_dir='xflr_polars'):
    """Load all XFLR5 polar files for dynamic Re-based lookup."""
    polar_path = Path(polar_dir)
    
    # Dictionary: airfoil_name -> {Re: {alpha: (Cl, Cd)}}
    xflr_polars = defaultdict(dict)
    
    if not polar_path.exists():
        print(f"Warning: Polar directory not found: {polar_path}")
        print("  Will use fixed values from optimized_design.txt")
        return xflr_polars
    
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
                xflr_polars[airfoil][re_value] = polar_data
                
        except Exception as e:
            print(f"  Warning: Could not load {polar_file.name}: {e}")
            continue
    
    # Print summary
    for airfoil, re_dict in xflr_polars.items():
        re_values = sorted(re_dict.keys())
        print(f"  {airfoil}: {len(re_values)} Re values from {min(re_values):.3f} to {max(re_values):.3f} x10^6")
    
    return xflr_polars


def lookup_polar(xflr_polars, airfoil, Re, alpha):
    """Look up Cl and Cd from XFLR5 polars at given Re and alpha.
    
    Parameters:
    -----------
    xflr_polars : dict
        Dictionary of loaded polar data
    airfoil : str
        Airfoil name (e.g., 'SD7037', 'franky')
    Re : float
        Reynolds number in millions (e.g., 0.080 for 80,000)
    alpha : float
        Angle of attack in degrees
        
    Returns:
    --------
    (Cl, Cd) : tuple of floats or (None, None) if not found
    """
    # Handle airfoil name variations
    airfoil_clean = airfoil.replace(' ', '').lower()
    
    # Find matching airfoil (case-insensitive)
    airfoil_key = None
    for key in xflr_polars.keys():
        if key.replace(' ', '').lower() == airfoil_clean:
            airfoil_key = key
            break
    
    if airfoil_key is None or not xflr_polars[airfoil_key]:
        # Fallback to fixed values if polar not found
        return None, None
    
    re_dict = xflr_polars[airfoil_key]
    available_Re = sorted(re_dict.keys())
    
    # Find two nearest Re values for linear interpolation
    if Re <= available_Re[0]:
        # Below range - use lowest Re
        polar_data = re_dict[available_Re[0]]
        return _interp_alpha(polar_data, alpha)
    elif Re >= available_Re[-1]:
        # Above range - use highest Re
        polar_data = re_dict[available_Re[-1]]
        return _interp_alpha(polar_data, alpha)
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
            return _interp_alpha(polar_data, alpha)
        
        # Get Cl, Cd at both bracketing Re values
        cl_low, cd_low = _interp_alpha(re_dict[re_low], alpha)
        cl_high, cd_high = _interp_alpha(re_dict[re_high], alpha)
        
        # Linear interpolation in Re
        if cl_low is None or cl_high is None:
            return None, None
        
        # Interpolate between the two closest Re values
        re_frac = (Re - re_low) / (re_high - re_low)
        cl = cl_low + re_frac * (cl_high - cl_low)
        cd = cd_low + re_frac * (cd_high - cd_low)
        
        return cl, cd


def _interp_alpha(polar_data, alpha):
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


def find_optimal_alpha(xflr_polars, airfoil, Re):
    """Find the angle of attack that gives maximum Cl/Cd (best efficiency) at given Re.
    
    Parameters:
    -----------
    xflr_polars : dict
        Dictionary of loaded polar data
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
    for key in xflr_polars.keys():
        if key.replace(' ', '').lower() == airfoil_clean:
            airfoil_key = key
            break
    
    if airfoil_key is None or not xflr_polars[airfoil_key]:
        return None, None
    
    re_dict = xflr_polars[airfoil_key]
    available_Re = sorted(re_dict.keys())
    
    # Get polar data at closest Re
    if Re <= available_Re[0]:
        polar_data = re_dict[available_Re[0]]
    elif Re >= available_Re[-1]:
        polar_data = re_dict[available_Re[-1]]
    else:
        # Use closest single Re for efficiency search (interpolating efficiency is complex)
        idx = min(range(len(available_Re)), key=lambda i: abs(available_Re[i] - Re))
        polar_data = re_dict[available_Re[idx]]
    
    # Find alpha with maximum Cl/Cd
    max_efficiency = 0
    optimal_alpha = None
    
    for alpha, (cl, cd) in polar_data.items():
        if cd > 0 and cl > 0:  # Only consider positive lift, avoid division by zero
            efficiency = cl / cd
            if efficiency > max_efficiency:
                max_efficiency = efficiency
                optimal_alpha = alpha
    
    return optimal_alpha, max_efficiency


def create_interpolated_blade_sections(blade_df, n_sections=30):
    """
    Create a fine mesh of blade sections with linearly interpolated properties.
    
    This provides smooth variation of Cl and Cd across the span instead of
    discrete jumps, leading to more accurate thrust calculations.
    
    Parameters:
    -----------
    blade_df : DataFrame
        Original discrete blade sections from optimized_design.txt
    n_sections : int
        Number of sections in fine mesh (default: 30)
        
    Returns:
    --------
    DataFrame with interpolated blade sections
    """
    # Get radius range
    r_min = blade_df['radius'].min()
    r_max = blade_df['radius'].max()
    
    # Create fine mesh
    r_fine = np.linspace(r_min, r_max, n_sections)
    
    # Linearly interpolate all properties
    chord_interp = np.interp(r_fine, blade_df['radius'], blade_df['chord'])
    cl_interp = np.interp(r_fine, blade_df['radius'], blade_df['Cl'])
    cd_interp = np.interp(r_fine, blade_df['radius'], blade_df['Cd'])
    alpha_interp = np.interp(r_fine, blade_df['radius'], blade_df['alpha'])
    
    # Assign nearest airfoil name to each interpolated radius
    # (airfoil is categorical — cannot interpolate, use nearest neighbour)
    if 'airfoil' in blade_df.columns:
        airfoil_names = []
        for r in r_fine:
            idx = (blade_df['radius'] - r).abs().idxmin()
            airfoil_names.append(blade_df.loc[idx, 'airfoil'])
    else:
        airfoil_names = ['unknown'] * n_sections
    
    # Create new dataframe with interpolated sections
    blade_fine = pd.DataFrame({
        'radius': r_fine,
        'chord': chord_interp,
        'Cl': cl_interp,
        'Cd': cd_interp,
        'alpha': alpha_interp,
        'Cl_Cd': cl_interp / cd_interp,  # Recompute ratio
        'airfoil': airfoil_names,
    })
    
    return blade_fine


# ============================================================================
# BLADE AERODYNAMICS WITH BEMT
# ============================================================================
def calculate_blade_forces_bemt(omega, blade_df, n_blades=4, max_iterations=20, xflr_polars=None):
    """
    Calculate blade forces using Blade Element Momentum Theory.
    
    Includes:
    - Induced velocity from momentum theory
    - Induced drag effects
    - Proper tangential/axial force resolution
    
    Parameters:
    -----------
    omega : float
        Spin rate in rad/s
    blade_df : DataFrame
        Blade section data
    n_blades : int
        Number of blades
    max_iterations : int
        Max iterations for induced velocity convergence
        
    Returns:
    --------
    dict with detailed force breakdown
    """
    results = {
        'sections': [],
        'total_thrust': 0.0,      # Axial force (vertical)
        'total_torque': 0.0,      # Moment about axis
        'total_power': 0.0,       # Power = Torque × omega
        'induced_velocity_avg': 0.0,
    }
    
    # Initial guess for average induced velocity (from momentum theory)
    # For hovering: v_i ≈ sqrt(T / (2 * rho * A))
    # Start with rough estimate
    r_tip = blade_df['radius'].max()
    A_disk = np.pi * r_tip**2
    v_induced_avg = 0.5  # Initial guess (m/s)
    
    # Iterate to converge on induced velocity
    for iteration in range(max_iterations):
        thrust_total = 0.0
        torque_total = 0.0
        
        section_results = []
        
        for i, row in blade_df.iterrows():
            r = row['radius']
            chord = row['chord']
            airfoil = row.get('airfoil', 'unknown')
            design_alpha = row.get('alpha', 10.0)
            Cl_2d = row['Cl']  # 2D lift coefficient from airfoil (fallback)
            Cd_profile = row['Cd']  # Profile drag coefficient (fallback)
            
            # Section width (dr) — midpoint rule bounded by wing root/tip
            if i == 0:
                dr = (blade_df.iloc[1]['radius'] - r) / 2  # no extension beyond wing root
            elif i == len(blade_df) - 1:
                dr = max((r - blade_df.iloc[i-1]['radius']) / 2, 0.001)  # no extension beyond tip
            else:
                dr = (blade_df.iloc[i+1]['radius'] - blade_df.iloc[i-1]['radius']) / 2
            
            # ================================================================
            # VELOCITY COMPONENTS
            # ================================================================
            # Tangential velocity (rotation)
            V_tangential = omega * r
            
            # Induced velocity (from momentum theory)
            # Varies along span - use Prandtl tip loss approximation
            tip_loss_factor = 1.0 - 0.3 * (r / r_tip)**2  # Simplified
            v_induced = v_induced_avg * tip_loss_factor
            
            # Total velocity components
            V_axial = v_induced  # Axial (downwash)
            V_tan = V_tangential  # Tangential
            
            # Resultant velocity
            V_resultant = np.sqrt(V_axial**2 + V_tan**2)
            
            # Calculate actual Reynolds number at this section
            Re_actual = RHO * V_resultant * chord / MU  # Full Re value
            Re_millions = Re_actual / 1e6  # Convert to millions for lookup
            
            # Inflow angle (angle between V_resultant and rotor plane)
            phi = np.arctan2(V_axial, V_tan)  # radians
            
            # ================================================================
            # ANGLES AND COEFFICIENTS
            # ================================================================
            # Use the DESIGN alpha (blade pitch is a fixed geometric property)
            # but look up Cl/Cd at the ACTUAL flight Re for that alpha.
            # Note: find_optimal_alpha maximises Cl/Cd (efficiency) and picks
            # a much lower alpha (~4°) than the design (~10°), which cuts Cl
            # nearly in half.  The physical blade is pitched to the design
            # angle — only the Re changes in flight.
            if xflr_polars:
                alpha_used = design_alpha
                    
                # Look up Cl and Cd at design alpha and actual Re
                Cl, Cd_profile_lookup = lookup_polar(xflr_polars, airfoil, Re_millions, alpha_used)
                # Fallback to fixed values if lookup fails
                if Cl is None or Cd_profile_lookup is None:
                    Cl = Cl_2d
                    Cd_profile = Cd_profile
                    alpha_used = design_alpha
                else:
                    Cd_profile = Cd_profile_lookup
            else:
                # Use fixed values from blade_df
                Cl = Cl_2d
                alpha_used = design_alpha
            
            Cd_total = Cd_profile
            
            # Induced drag (due to 3D effects and induced flow)
            # Simplified: Cd_induced ≈ Cl² / (π * AR * e)
            # For rotor blade, use local aspect ratio
            AR_local = (2 * r) / chord  # Simplified local AR
            e_oswald = 0.85  # Oswald efficiency
            Cd_induced = Cl**2 / (np.pi * AR_local * e_oswald)
            Cd_total += Cd_induced
            
            # ================================================================
            # FORCES
            # ================================================================
            # Dynamic pressure based on resultant velocity
            q = 0.5 * RHO * V_resultant**2
            
            # Lift and drag per unit span (perpendicular and parallel to V_resultant)
            dL_per_dr = q * chord * Cl
            dD_per_dr = q * chord * Cd_total
            
            # Total forces for this section
            dL = dL_per_dr * dr
            dD = dD_per_dr * dr
            
            # ================================================================
            # RESOLVE INTO THRUST AND TORQUE
            # ================================================================
            # Thrust (axial force, positive upward)
            # T = L*cos(phi) - D*sin(phi)
            dT = dL * np.cos(phi) - dD * np.sin(phi)
            
            # Tangential force (creates torque)
            # F_tan = L*sin(phi) + D*cos(phi)
            dF_tan = dL * np.sin(phi) + dD * np.cos(phi)
            
            # Torque = F_tan × radius
            dQ = dF_tan * r
            
            # Store section results
            section_results.append({
                'radius': r,
                'V_tan': V_tan,
                'V_induced': v_induced,
                'V_resultant': V_resultant,
                'phi_deg': np.degrees(phi),
                'alpha_used': alpha_used,
                'Re_millions': Re_millions,
                'Cl': Cl,
                'Cd_profile': Cd_profile,
                'Cd_induced': Cd_induced,
                'Cd_total': Cd_total,
                'dL': dL,
                'dD': dD,
                'dT': dT,
                'dQ': dQ,
                'dF_tan': dF_tan,
            })
            
            thrust_total += dT
            torque_total += dQ
        
        # ================================================================
        # UPDATE INDUCED VELOCITY (Momentum Theory)
        # ================================================================
        # For hovering: v_i = sqrt(T / (2 * rho * A))
        thrust_single_blade = thrust_total
        thrust_all_blades = thrust_single_blade * n_blades
        
        # Update induced velocity estimate
        if thrust_all_blades > 0:
            v_induced_new = np.sqrt(thrust_all_blades / (2 * RHO * A_disk))
        else:
            v_induced_new = 0.0
        
        # Check convergence
        if abs(v_induced_new - v_induced_avg) < 0.01:  # 1 cm/s tolerance
            v_induced_avg = v_induced_new
            break
        
        # Relaxation for stability
        v_induced_avg = 0.5 * v_induced_avg + 0.5 * v_induced_new
    
    # Scale by number of blades
    results['total_thrust'] = thrust_total * n_blades
    results['total_torque'] = torque_total * n_blades
    results['total_power'] = results['total_torque'] * omega
    results['induced_velocity_avg'] = v_induced_avg
    results['sections'] = section_results
    results['iterations'] = iteration + 1
    
    return results


# Backward compatibility wrapper
def calculate_blade_forces(omega, blade_df, n_blades=4, xflr_polars=None):
    """Wrapper for BEMT calculation"""
    bemt_results = calculate_blade_forces_bemt(omega, blade_df, n_blades, xflr_polars=xflr_polars)
    
    # Convert to old format for compatibility
    return {
        'sections': bemt_results['sections'],
        'total_lift': bemt_results['total_thrust'],  # Thrust is vertical lift
        'total_drag': sum(s['dD'] for s in bemt_results['sections']) * n_blades,
        'total_torque': bemt_results['total_torque'],
        'induced_velocity': bemt_results['induced_velocity_avg'],
        'power': bemt_results['total_power'],
    }


# ============================================================================
# MOTOR REQUIREMENTS
# ============================================================================
def calculate_motor_requirements(blade_lift, blade_drag_torque, omega, 
                                  target_omega=None, motor_radius=MOTOR_RADIUS,
                                  num_motors=NUM_MOTORS):
    """
    Calculate motor thrust and tilt needed for equilibrium.
    
    At equilibrium:
    1. Vertical: blade_lift + motor_thrust_vertical = weight
    2. Rotational: motor_tangential_thrust * r = blade_drag_torque
       (for maintaining spin rate)
    
    If target_omega > current omega, need extra torque to accelerate.
    
    Parameters:
    -----------
    blade_lift : float
        Total lift from blades (N)
    blade_drag_torque : float
        Total drag torque from blades (N·m) - opposes rotation
    omega : float
        Current spin rate (rad/s)
    target_omega : float
        Target spin rate (rad/s), if None = maintain current
    motor_radius : float
        Motor distance from center (m)
    num_motors : int
        Number of motors
        
    Returns:
    --------
    dict with motor thrust, tilt angle, etc.
    """
    # Vertical force needed from motors
    vertical_needed = DRONE_WEIGHT_N - blade_lift
    
    # Tangential force needed to overcome blade drag torque
    # Torque = Force * radius
    # For steady spin: motors must counteract blade drag torque
    # Motor reaction torque (all props CW) also contributes to yaw:
    #   reaction_yaw_torque = N * K_M * T_per_motor * cos(tilt)
    # This is coupled with T and tilt, so we solve iteratively below.
    
    # If accelerating to higher omega, need additional torque
    if target_omega is not None and target_omega > omega:
        # Simplified: assume angular acceleration contributes
        # Just use target omega for blade force calculation instead
        pass
    
    # Each motor provides both vertical and tangential components
    # T_motor * cos(tilt) = vertical component
    # T_motor * sin(tilt) = tangential component (for torque)
    # Additionally, each motor's prop reaction torque has a yaw component:
    #   Q_yaw = K_M * T_per_motor * cos(tilt)
    # Total motor yaw torque:
    #   τ_motor = T_total * sin(tilt) * r_motor + N * K_M * (T_total/N) * cos(tilt)
    #           = T_total * (sin(tilt) * r_motor + K_M * cos(tilt))
    #
    # Required: τ_motor = blade_drag_torque
    # Vertical: T_total * cos(tilt) = vertical_needed
    # So T_total = vertical_needed / cos(tilt)
    # Substitute: (vertical_needed / cos(tilt)) * (sin(tilt)*r + K_M*cos(tilt)) = torque_needed
    #  => vertical_needed * (tan(tilt)*r + K_M) = torque_needed
    #  => tan(tilt) = (torque_needed/vertical_needed - K_M) / r
    
    torque_needed = blade_drag_torque  # Wing drag torque to overcome
    
    if vertical_needed > 0:
        # tan(tilt) = (torque_needed / vertical_needed - K_M) / r_motor
        tan_tilt = (torque_needed / vertical_needed - MOTOR_KM) / motor_radius
        tan_tilt = max(tan_tilt, 0.0)  # tilt can't be negative
        tilt_rad = np.arctan(tan_tilt)
    else:
        tilt_rad = np.pi / 2  # Pure horizontal (all lift from wings)
    
    tilt_deg = np.degrees(tilt_rad)
    
    # Total motor thrust magnitude: T = vertical_needed / cos(tilt)
    if np.cos(tilt_rad) > 0.01:
        total_thrust_needed = vertical_needed / np.cos(tilt_rad)
    else:
        total_thrust_needed = vertical_needed  # Near 90°, fallback
    
    tangential_force = total_thrust_needed * np.sin(tilt_rad)
    
    # Per-motor thrust
    thrust_per_motor = total_thrust_needed / num_motors
    
    # Reaction torque contribution to yaw
    reaction_torque_yaw = MOTOR_KM * vertical_needed  # K_M * T_total * cos(tilt) = K_M * vertical
    
    # Efficiency metric: how much of motor thrust is "vertical" vs "wasted" on spin
    vertical_efficiency = np.cos(tilt_rad) if tilt_rad < np.pi/2 else 0
    
    return {
        'blade_lift': blade_lift,
        'blade_drag_torque': blade_drag_torque,
        'vertical_needed': vertical_needed,
        'tangential_needed': tangential_force,
        'reaction_torque_yaw': reaction_torque_yaw,
        'motor_tilt_deg': tilt_deg,
        'total_motor_thrust': total_thrust_needed,
        'thrust_per_motor': thrust_per_motor,
        'vertical_efficiency': vertical_efficiency,
        'lift_fraction': blade_lift / DRONE_WEIGHT_N,  # % of weight from wings
    }


# ============================================================================
# OPTIMIZATION SWEEP
# ============================================================================
def sweep_spin_rates(blade_df, omega_range, n_blades=4, xflr_polars=None):
    """
    Sweep through spin rates and calculate motor requirements with BEMT.
    """
    results = []
    
    print(f"\n{'Omega':<8} {'Induced_v':<10} {'Thrust(N)':<10} {'Power(W)':<10} {'Motor(N)':<10} {'Tilt(°)':<10}")
    print("-"*75)
    
    for omega in omega_range:
        # Calculate blade forces at this spin rate (using BEMT)
        blade_forces = calculate_blade_forces(omega, blade_df, n_blades, xflr_polars=xflr_polars)
        
        # Calculate motor requirements
        motor_req = calculate_motor_requirements(
            blade_forces['total_lift'],
            blade_forces['total_torque'],
            omega
        )
        
        result = {
            'omega': omega,
            'rpm': omega * 60 / (2 * np.pi),
            'blade_lift': blade_forces['total_lift'],
            'blade_drag': blade_forces['total_drag'],
            'blade_torque': blade_forces['total_torque'],
            'induced_velocity': blade_forces.get('induced_velocity', 0.0),
            'power': blade_forces.get('power', 0.0),
            **motor_req
        }
        
        # Add efficiency metrics
        if result['power'] > 0:
            # Aerodynamic efficiency: Wing thrust per watt (N/W)
            result['aero_efficiency'] = result['blade_lift'] / result['power']
        else:
            result['aero_efficiency'] = 0.0
        
        # System efficiency: How much do wings reduce motor workload?
        # Wing assist ratio = wing_thrust / motor_thrust
        if result['total_motor_thrust'] > 0:
            result['wing_assist_ratio'] = result['blade_lift'] / result['total_motor_thrust']
        else:
            result['wing_assist_ratio'] = 0.0
        
        # Motor reduction: How much less motor thrust vs no wings (100% = no motors needed)
        result['motor_reduction'] = (DRONE_WEIGHT_N - result['total_motor_thrust']) / DRONE_WEIGHT_N * 100
        
        # ================================================================
        # DUAL MOTOR POWER CALCULATIONS
        # ================================================================
        # 1. Motor Spin Power: Power to maintain spin (tangential work = torque × omega)
        #    This matches sweep_tilt.py's calculation
        #    Includes both tangential thrust torque and prop reaction torque
        motor_torque = result['tangential_needed'] * MOTOR_RADIUS + result.get('reaction_torque_yaw', 0.0)
        result['motor_spin_power'] = motor_torque * omega
        
        # 2. Motor Hover Power: Power consumed by motor propellers (momentum theory)
        #    P_motor = T * v_induced / efficiency
        motor_disk_area = np.pi * (MOTOR_PROP_DIAMETER / 2)**2 * NUM_MOTORS  # Total rotor disk area
        motor_induced_v = np.sqrt(result['total_motor_thrust'] / (2 * RHO * motor_disk_area)) if result['total_motor_thrust'] > 0 else 0
        result['motor_hover_power'] = result['total_motor_thrust'] * motor_induced_v / MOTOR_EFFICIENCY
        
        # Keep backward-compatible alias
        result['motor_power'] = result['motor_hover_power']
        
        # Total system power (wing + motor hover power)
        result['total_system_power'] = result['power'] + result['motor_hover_power']
        
        results.append(result)
        
        # Print progress
        print(f"{omega:<8.1f} {result['induced_velocity']:<10.2f} {result['blade_lift']:<10.2f} "
              f"{result['power']:<10.2f} {result['total_motor_thrust']:<10.2f} {result['motor_tilt_deg']:<10.1f}")
    
    return pd.DataFrame(results)


# ============================================================================
# TILT-SWEEP MODE (Matches sweep_tilt.py simulation approach)
# ============================================================================

# Controller gains matching sim_spinning_wing.py
KP_ALT = 15.0  # Altitude proportional gain (must match sim_spinning_wing.py)
KD_ALT = 4.0   # Altitude velocity damping gain (must match sim_spinning_wing.py)
KI_ALT = 3.0   # Altitude integral gain (not used in equilibrium calc - settles to ~0)
TARGET_ALTITUDE = 1.5  # Target altitude in meters
SIMULATED_ALTITUDE = 1.5  # At true hover equilibrium, z_error = 0 (no net vertical acceleration)
DEFAULT_Z_ERROR = TARGET_ALTITUDE - SIMULATED_ALTITUDE  # 0.0m - correct for hover

# Number of wing blades (one per motor arm)
NUM_BLADES = 4


def find_equilibrium_omega_with_controller(tilt_deg, blade_df, n_blades=4, xflr_polars=None,
                                            omega_range=(4.0, 20.0), tolerance=0.01, max_iter=100,
                                            z_error=DEFAULT_Z_ERROR):
    """
    Find equilibrium omega INCLUDING the altitude controller's effect.
    
    The simulation reaches an equilibrium where altitude settles BELOW target,
    creating a steady-state altitude error. This error, multiplied by KP_ALT,
    provides extra vertical thrust. The tangential component of this extra
    thrust creates the torque needed to maintain spin.
    
    This function finds omega where:
        motor_torque = wing_torque
    
    Given that:
        vertical_needed = (DRONE_WEIGHT - wing_thrust) + KP_ALT * z_error
        motor_thrust = vertical_needed / cos(tilt)
        motor_torque = motor_thrust * sin(tilt) * r_motor
    
    Parameters:
    -----------
    tilt_deg : float
        Fixed motor tilt angle (degrees)
    blade_df : DataFrame
        Blade section data
    n_blades : int
        Number of blades
    xflr_polars : dict
        XFLR5 polar data for dynamic lookup
    omega_range : tuple
        (min_omega, max_omega) search bounds in rad/s
    tolerance : float
        Convergence tolerance (Nm) for torque residual
    max_iter : int
        Maximum iterations
    z_error : float
        Altitude error (m) used to match simulation. Default is observed from sim.
        
    Returns:
    --------
    dict with equilibrium values, or None if no equilibrium found
    """
    tilt_rad = np.radians(tilt_deg)
    cos_tilt = np.cos(tilt_rad)
    sin_tilt = np.sin(tilt_rad)
    
    omega_low, omega_high = omega_range
    
    # Bisection search for equilibrium omega
    for iteration in range(max_iter):
        omega_mid = (omega_low + omega_high) / 2
        
        # Calculate wing forces at this omega
        blade_forces = calculate_blade_forces(omega_mid, blade_df, n_blades, xflr_polars=xflr_polars)
        wing_thrust = blade_forces['total_lift']
        wing_torque = blade_forces['total_torque']
        
        # Controller calculation with fixed z_error (matching simulation)
        vertical_needed = (DRONE_WEIGHT_N - wing_thrust) + KP_ALT * z_error
        if vertical_needed < 0:
            vertical_needed = 0
        
        motor_thrust = vertical_needed / cos_tilt if cos_tilt > 0.01 else 0.0
        motor_tangential = motor_thrust * sin_tilt
        
        # Yaw torque from tilted thrust force acting through lever arm
        thrust_torque = motor_tangential * MOTOR_RADIUS
        
        # Yaw torque from tilted motor prop reaction torque
        # Each prop: Q = K_M * T_per_motor, yaw component = Q * cos(tilt)
        # Total: N_motors * K_M * T_per_motor * cos(tilt) = K_M * T_total * cos(tilt)
        # But T_total * cos(tilt) = vertical_needed, so:
        reaction_torque = MOTOR_KM * vertical_needed
        
        motor_torque = thrust_torque + reaction_torque
        
        # Residual: motor_torque - wing_torque
        # Positive = excess motor torque (omega would accelerate)
        # Negative = insufficient motor torque (omega would decelerate)
        torque_residual = motor_torque - wing_torque
        
        # Check convergence
        if abs(torque_residual) < tolerance:
            # Found equilibrium
            motor_spin_power = motor_torque * omega_mid
            
            motor_disk_area = np.pi * (MOTOR_PROP_DIAMETER / 2)**2 * NUM_MOTORS
            motor_induced_v = np.sqrt(motor_thrust / (2 * RHO * motor_disk_area)) if motor_thrust > 0 else 0
            motor_hover_power = motor_thrust * motor_induced_v / MOTOR_EFFICIENCY
            
            return {
                'tilt_deg': tilt_deg,
                'omega': omega_mid,
                'rpm': omega_mid * 60 / (2 * np.pi),
                'wing_thrust': wing_thrust,
                'wing_torque': wing_torque,
                'wing_power': blade_forces['power'],
                'motor_thrust': motor_thrust,
                'motor_per': motor_thrust / NUM_MOTORS,
                'motor_tangential': motor_tangential,
                'motor_torque': motor_torque,
                'motor_spin_power': motor_spin_power,
                'motor_hover_power': motor_hover_power,
                'induced_velocity': blade_forces.get('induced_velocity', 0.0),
                'wing_fraction': wing_thrust / DRONE_WEIGHT_N * 100,
                'z_error': z_error,
                'predicted_altitude': TARGET_ALTITUDE - z_error,
                'torque_residual': torque_residual,
                'iterations': iteration + 1,
            }
        
        # Bisection step
        # If motor_torque > wing_torque, omega should increase (more drag to absorb torque)
        if torque_residual > 0:
            omega_low = omega_mid
        else:
            omega_high = omega_mid
    
    # Failed to converge
    return None


def sweep_tilt_with_controller(blade_df, tilt_range, n_blades=4, xflr_polars=None):
    """
    Sweep through tilt angles and find equilibrium INCLUDING controller dynamics.
    
    This matches how sweep_tilt.py's simulation reaches equilibrium,
    where the altitude controller offset provides extra thrust for spin torque.
    """
    results = []
    
    print(f"\n{'Tilt':<6} {'omega':<8} {'RPM':<8} {'WingT':<8} {'MotorT':<8} {'Alt':<8} {'SpinP':<8}")
    print("-" * 70)
    
    for tilt_deg in tilt_range:
        result = find_equilibrium_omega_with_controller(tilt_deg, blade_df, n_blades, xflr_polars=xflr_polars)
        
        if result is not None:
            results.append(result)
            print(f"{tilt_deg:<6.0f} {result['omega']:<8.2f} {result['rpm']:<8.1f} "
                  f"{result['wing_thrust']:<8.2f} {result['motor_thrust']:<8.2f} "
                  f"{result['predicted_altitude']:<8.2f} "
                  f"{result['motor_spin_power']:<8.2f}")
        else:
            print(f"{tilt_deg:<6.0f} {'NO EQUILIBRIUM':<50}")
    
    return pd.DataFrame(results)



def find_equilibrium_omega(tilt_deg, blade_df, n_blades=4, xflr_polars=None,
                           omega_range=(4.0, 20.0), tolerance=0.01, max_iter=50):
    """
    Find the equilibrium omega for a given fixed tilt angle.
    
    At equilibrium:
    - motor_tangential × r_motor = wing_torque(omega)
    - motor_thrust × cos(tilt) + wing_thrust(omega) = weight
    
    This matches how sweep_tilt.py's simulation naturally finds equilibrium.
    
    Parameters:
    -----------
    tilt_deg : float
        Fixed motor tilt angle (degrees)
    blade_df : DataFrame
        Blade section data
    n_blades : int
        Number of blades
    xflr_polars : dict
        XFLR5 polar data for dynamic lookup
    omega_range : tuple
        (min_omega, max_omega) search bounds in rad/s
    tolerance : float
        Convergence tolerance for omega (rad/s)
    max_iter : int
        Maximum iterations
        
    Returns:
    --------
    dict with equilibrium values, or None if no equilibrium found
    """
    tilt_rad = np.radians(tilt_deg)
    cos_tilt = np.cos(tilt_rad)
    sin_tilt = np.sin(tilt_rad)
    
    omega_low, omega_high = omega_range
    
    # Bisection search for equilibrium omega
    for iteration in range(max_iter):
        omega_mid = (omega_low + omega_high) / 2
        
        # Calculate wing forces at this omega
        blade_forces = calculate_blade_forces(omega_mid, blade_df, n_blades, xflr_polars=xflr_polars)
        wing_thrust = blade_forces['total_lift']
        wing_torque = blade_forces['total_torque']
        
        # Calculate what motor thrust would be needed for vertical equilibrium
        vertical_needed = DRONE_WEIGHT_N - wing_thrust
        if vertical_needed < 0:
            vertical_needed = 0  # Wing provides all lift
        
        # motor_thrust × cos(tilt) = vertical_needed
        motor_thrust = vertical_needed / cos_tilt if cos_tilt > 0.01 else 0.0
        
        # Tangential force the motors would provide at this thrust and tilt
        motor_tangential = motor_thrust * sin_tilt
        
        # Yaw torque = thrust tangential component × lever arm + prop reaction torque
        thrust_torque = motor_tangential * MOTOR_RADIUS
        # Reaction torque: K_M * T_total * cos(tilt) = K_M * vertical_needed
        reaction_torque = MOTOR_KM * vertical_needed
        motor_torque = thrust_torque + reaction_torque
        
        # Residual: how much the motor torque exceeds/falls short of wing torque
        # Positive = excess motor torque (omega would accelerate)
        # Negative = insufficient motor torque (omega would decelerate)
        torque_residual = motor_torque - wing_torque
        
        # Check convergence
        if abs(torque_residual) < tolerance * MOTOR_RADIUS:  # Scale tolerance by r
            # Found equilibrium
            # Calculate power metrics
            motor_spin_power = motor_torque * omega_mid
            
            motor_disk_area = np.pi * (MOTOR_PROP_DIAMETER / 2)**2 * NUM_MOTORS
            motor_induced_v = np.sqrt(motor_thrust / (2 * RHO * motor_disk_area)) if motor_thrust > 0 else 0
            motor_hover_power = motor_thrust * motor_induced_v / MOTOR_EFFICIENCY
            
            return {
                'tilt_deg': tilt_deg,
                'omega': omega_mid,
                'rpm': omega_mid * 60 / (2 * np.pi),
                'wing_thrust': wing_thrust,
                'wing_torque': wing_torque,
                'wing_power': blade_forces['power'],
                'motor_thrust': motor_thrust,
                'motor_per': motor_thrust / NUM_MOTORS,
                'motor_tangential': motor_tangential,
                'motor_torque': motor_torque,
                'motor_spin_power': motor_spin_power,
                'motor_hover_power': motor_hover_power,
                'induced_velocity': blade_forces.get('induced_velocity', 0.0),
                'wing_fraction': wing_thrust / DRONE_WEIGHT_N * 100,
                'iterations': iteration + 1,
            }
        
        # Bisection step
        if torque_residual > 0:
            # Excess motor torque → omega would accelerate → need higher omega (more drag)
            omega_low = omega_mid
        else:
            # Insufficient motor torque → omega would decelerate → need lower omega
            omega_high = omega_mid
    
    # Failed to converge
    return None


def sweep_tilt_analytical(blade_df, tilt_range, n_blades=4, xflr_polars=None):
    """
    Sweep through tilt angles and find equilibrium omega for each.
    
    This matches sweep_tilt.py's approach where tilt is fixed and omega
    stabilizes to where torques balance.
    
    Parameters:
    -----------
    blade_df : DataFrame
        Blade section data
    tilt_range : array-like
        Tilt angles to sweep (degrees)
    n_blades : int
        Number of blades
    xflr_polars : dict
        XFLR5 polar data for dynamic lookup
        
    Returns:
    --------
    DataFrame with equilibrium values for each tilt
    """
    results = []
    
    print(f"\n{'Tilt°':<8} {'ω(rad/s)':<10} {'ω(RPM)':<10} {'WingT(N)':<10} {'MotorT(N)':<10} {'SpinP(W)':<10} {'HoverP(W)':<10}")
    print("-" * 78)
    
    for tilt_deg in tilt_range:
        result = find_equilibrium_omega(tilt_deg, blade_df, n_blades, xflr_polars=xflr_polars)
        
        if result is not None:
            results.append(result)
            print(f"{tilt_deg:<8.1f} {result['omega']:<10.2f} {result['rpm']:<10.1f} "
                  f"{result['wing_thrust']:<10.2f} {result['motor_thrust']:<10.2f} "
                  f"{result['motor_spin_power']:<10.2f} {result['motor_hover_power']:<10.2f}")
        else:
            print(f"{tilt_deg:<8.1f} {'NO EQUILIBRIUM':<60}")
    
    return pd.DataFrame(results)


# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_results(sweep_df, n_blades):
    """Create comprehensive visualization of BEMT results."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Blade Thrust vs Omega (with induced velocity)
    ax = axes[0, 0]
    ax.plot(sweep_df['omega'], sweep_df['blade_lift'], 'b-o', linewidth=2, label='Blade Thrust')
    ax.axhline(y=DRONE_WEIGHT_N, color='r', linestyle='--', label=f'Drone weight ({DRONE_WEIGHT_N:.2f} N)')
    ax.set_xlabel('Spin Rate ω (rad/s)', fontsize=11)
    ax.set_ylabel('Blade Thrust (N)', fontsize=11)
    ax.set_title(f'Blade Thrust vs Spin Rate ({n_blades} blades)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Motor Thrust & Power Required vs Omega
    ax = axes[0, 1]
    ax2 = ax.twinx()
    
    line1 = ax.plot(sweep_df['omega'], sweep_df['total_motor_thrust'], 'r-o', linewidth=2, label='Total Motor Thrust')
    line2 = ax2.plot(sweep_df['omega'], sweep_df.get('power', [0]*len(sweep_df)), 'purple', 
                     linestyle='--', linewidth=2, marker='s', label='Rotor Power')
    
    ax.set_xlabel('Spin Rate ω (rad/s)', fontsize=11)
    ax.set_ylabel('Motor Thrust (N)', fontsize=11, color='r')
    ax2.set_ylabel('Rotor Power (W)', fontsize=11, color='purple')
    ax.tick_params(axis='y', labelcolor='r')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax.set_title('Motor Thrust & Rotor Power', fontweight='bold')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 3. Motor Tilt & Induced Velocity
    ax = axes[0, 2]
    ax2 = ax.twinx()
    
    line1 = ax.plot(sweep_df['omega'], sweep_df['motor_tilt_deg'], 'm-o', linewidth=2, label='Motor Tilt')
    line2 = ax2.plot(sweep_df['omega'], sweep_df.get('induced_velocity', [0]*len(sweep_df)), 'cyan',
                     linestyle='--', linewidth=2, marker='s', label='Induced Velocity')
    
    ax.set_xlabel('Spin Rate ω (rad/s)', fontsize=11)
    ax.set_ylabel('Motor Tilt Angle (°)', fontsize=11, color='m')
    ax2.set_ylabel('Induced Velocity (m/s)', fontsize=11, color='cyan')
    ax.tick_params(axis='y', labelcolor='m')
    ax2.tick_params(axis='y', labelcolor='cyan')
    ax.set_title('Motor Tilt & Induced Downwash', fontweight='bold')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # 4. Wing Lift Fraction
    ax = axes[1, 0]
    ax.plot(sweep_df['omega'], sweep_df['lift_fraction'] * 100, 'g-o', linewidth=2.5)
    ax.axhline(y=100, color='r', linestyle='--', linewidth=2, label='100% (no motor lift needed)')
    ax.fill_between(sweep_df['omega'], 0, sweep_df['lift_fraction'] * 100, alpha=0.2, color='green')
    ax.set_xlabel('Spin Rate ω (rad/s)', fontsize=11)
    ax.set_ylabel('Wing Lift / Weight (%)', fontsize=11)
    ax.set_title('Wing Contribution to Weight Support', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. System Efficiency: Wing Assist & Motor Reduction
    ax = axes[1, 1]
    if 'wing_assist_ratio' in sweep_df.columns:
        # Primary axis: Wing/Motor ratio
        color = 'tab:blue'
        ax.plot(sweep_df['omega'], sweep_df['wing_assist_ratio'], 'o-', color=color, linewidth=2.5, markersize=6, label='Wing/Motor Ratio')
        ax.set_xlabel('Spin Rate ω (rad/s)', fontsize=11)
        ax.set_ylabel('Wing/Motor Thrust Ratio', fontsize=11, color=color)
        ax.tick_params(axis='y', labelcolor=color)
        
        # Mark maximum
        max_idx = sweep_df['wing_assist_ratio'].idxmax()
        max_row = sweep_df.iloc[max_idx]
        ax.plot(max_row['omega'], max_row['wing_assist_ratio'], 'b*', markersize=20, 
                label=f'Max: {max_row["wing_assist_ratio"]:.1f}x @ {max_row["omega"]:.1f} rad/s')
        ax.fill_between(sweep_df['omega'], 0, sweep_df['wing_assist_ratio'], alpha=0.2, color='blue')
        
        # Secondary axis: Motor reduction %
        ax2 = ax.twinx()
        color = 'tab:green'
        ax2.plot(sweep_df['omega'], sweep_df['motor_reduction'], 's--', color=color, linewidth=2, markersize=4, alpha=0.7, label='Motor Reduction %')
        ax2.set_ylabel('Motor Thrust Saved (%)', fontsize=11, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.axhline(y=100, color='red', linestyle=':', linewidth=1, alpha=0.5)
        
        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
        ax.set_title('🎯 System Efficiency (Wing Assists Motor)', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # 6. Summary Table
    ax = axes[1, 2]
    ax.axis('off')
    
    # Find key points
    baseline_idx = (sweep_df['omega'] - 8.0).abs().idxmin()
    min_motor_idx = sweep_df['total_motor_thrust'].idxmin()
    max_assist_idx = sweep_df['wing_assist_ratio'].idxmax()
    
    baseline = sweep_df.iloc[baseline_idx]
    optimal = sweep_df.iloc[min_motor_idx]
    best_assist = sweep_df.iloc[max_assist_idx]
    
    # Get induced velocity if available
    v_i_baseline = baseline.get('induced_velocity', 0)
    v_i_optimal = optimal.get('induced_velocity', 0)
    
    summary = f"""
    SPINNING DRONE BEMT ANALYSIS
    {'═'*40}
    
    System: {DRONE_WEIGHT_N:.2f} N ({DRONE_WEIGHT_KG*1000:.0f}g)
    Motors: {NUM_MOTORS} @ r={MOTOR_RADIUS:.2f}m | Blades: {n_blades}
    
    {'─'*40}
    BASELINE (ω = {baseline['omega']:.1f} rad/s)
    {'─'*40}
    • Wing thrust:     {baseline['blade_lift']:.2f} N ({baseline['lift_fraction']*100:.0f}%)
    • Motor thrust:    {baseline['total_motor_thrust']:.2f} N
    • Power:           {baseline['power']:.1f} W
    • Aero eff:        {baseline['aero_efficiency']:.2f} N/W
    • Wing/Motor:      {baseline['wing_assist_ratio']:.2f}x
    • Motor saved:     {baseline['motor_reduction']:.0f}%
    
    {'─'*40}
    MIN MOTOR (ω = {optimal['omega']:.1f} rad/s)
    {'─'*40}
    • Motor thrust:    {optimal['total_motor_thrust']:.2f} N ⭐
    • Wing thrust:     {optimal['blade_lift']:.2f} N ({optimal['lift_fraction']*100:.0f}%)
    • Power:           {optimal['power']:.1f} W
    • Wing/Motor:      {optimal['wing_assist_ratio']:.2f}x
    • Motor saved:     {optimal['motor_reduction']:.0f}%
    
    {'─'*40}
    🎯 BEST ASSIST (ω = {best_assist['omega']:.1f} rad/s)
    {'─'*40}
    • Wing/Motor:      {best_assist['wing_assist_ratio']:.2f}x 🎯
    • Motor saved:     {best_assist['motor_reduction']:.0f}%
    • Motor thrust:    {best_assist['total_motor_thrust']:.2f} N
    • Wing thrust:     {best_assist['blade_lift']:.2f} N
    • Power:           {best_assist['power']:.1f} W
    
    Motor Reduction: {optimal['motor_reduction']:.0f}%
    Best Wing Assist: {best_assist['wing_assist_ratio']:.1f}x at {best_assist['omega']:.0f} rad/s
    """
    
    ax.text(0.05, 0.97, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, edgecolor='navy', linewidth=2))
    
    plt.suptitle('Spinning Drone Motor Optimization (BEMT)', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/spinning_drone_bemt_analysis.png', dpi=200, bbox_inches='tight')
    print("\n✓ Saved: results/spinning_drone_bemt_analysis.png")
    
    return fig


# ============================================================================
# MAIN ANALYSIS
# ============================================================================
def main():
    print("="*80)
    print("  SPINNING DRONE MOTOR OPTIMIZATION ANALYSIS (BEMT)")
    print("="*80)
    print(f"\n  Drone weight: {DRONE_WEIGHT_KG*1000:.0f} g ({DRONE_WEIGHT_N:.2f} N)")
    print(f"  Motors: {NUM_MOTORS} @ r = {MOTOR_RADIUS:.2f}m")
    print(f"  Baseline spin: 8 rad/s")
    print(f"  Test range: 6-15 rad/s")
    print(f"\n  FOCUS: 4-blade configuration")
    
    # Load blade data
    print("\n" + "-"*40)
    print("Loading blade data from optimized_design.txt...")
    blade_df_discrete = load_blade_data()
    print(f"Loaded {len(blade_df_discrete)} discrete sections")
    print(blade_df_discrete[['radius', 'chord', 'Cl', 'Cl_Cd']].to_string())
    
    # Create interpolated blade sections for accurate thrust calculation
    print("\n" + "-"*40)
    print("Creating interpolated blade sections (linear interpolation)...")
    blade_df = create_interpolated_blade_sections(blade_df_discrete, n_sections=30)
    print(f"Generated {len(blade_df)} interpolated sections")
    print(f"Radius range: {blade_df['radius'].min():.3f} - {blade_df['radius'].max():.3f} m")
    print(f"Cl range: {blade_df['Cl'].min():.3f} - {blade_df['Cl'].max():.3f}")
    print(f"Cd range: {blade_df['Cd'].min():.4f} - {blade_df['Cd'].max():.4f}")
    
    # Load XFLR5 polar files for dynamic Re-based lookup
    xflr_polars = load_xflr_polars()
    
    # =========================================================================
    # SPINNING RATE ANALYSIS (4 BLADES, r_motor = 0.4m)
    # =========================================================================
    n_blades = 4
    print(f"\n{'='*80}")
    print(f"  ANALYSIS WITH {n_blades} BLADES @ r_motor = {MOTOR_RADIUS:.2f}m")
    print(f"{'='*80}")
    
    # Sweep spin rates
    omega_range = np.linspace(6, 15, 19)
    sweep_df = sweep_spin_rates(blade_df, omega_range, n_blades, xflr_polars=xflr_polars)
    
    # Print summary
    # Calculate motor-only baseline (no spinning)
    motor_only_disk_area = np.pi * (MOTOR_PROP_DIAMETER / 2)**2 * NUM_MOTORS
    motor_only_v_induced = np.sqrt(DRONE_WEIGHT_N / (2 * RHO * motor_only_disk_area))
    motor_only_power = DRONE_WEIGHT_N * motor_only_v_induced / MOTOR_EFFICIENCY
    
    print(f"\n{'Omega':<8} {'RPM':<8} {'WingT(N)':<10} {'WingP(W)':<10} {'MotorT(N)':<10} {'MotorP(W)':<10} {'TotalP(W)':<10} {'Save%':<10}")
    print("-"*95)
    for _, row in sweep_df.iterrows():
        power_savings = (motor_only_power - row['total_system_power']) / motor_only_power * 100
        print(f"{row['omega']:<8.1f} {row['rpm']:<8.1f} {row['blade_lift']:<10.2f} {row['power']:<10.2f} "
              f"{row['total_motor_thrust']:<10.2f} {row['motor_power']:<10.2f} {row['total_system_power']:<10.2f} {power_savings:<10.1f}")
    
    # Find optimal points
    optimal_min_motor_idx = sweep_df['total_motor_thrust'].idxmin()
    optimal_min_motor = sweep_df.iloc[optimal_min_motor_idx]
    
    optimal_aero_eff_idx = sweep_df['aero_efficiency'].idxmax()
    optimal_aero_eff = sweep_df.iloc[optimal_aero_eff_idx]
    
    optimal_assist_idx = sweep_df['wing_assist_ratio'].idxmax()
    optimal_assist = sweep_df.iloc[optimal_assist_idx]
    
    baseline_idx = (sweep_df['omega'] - 8.0).abs().idxmin()
    baseline = sweep_df.iloc[baseline_idx]
    
    # Find minimum total system power (most energy efficient)
    min_power_idx = sweep_df['total_system_power'].idxmin()
    min_power = sweep_df.iloc[min_power_idx]
    
    print(f"\n{'='*80}")
    print(f"  ENERGY COMPARISON")
    print(f"{'='*80}")
    print(f"\n🔋 MOTOR-ONLY BASELINE (no spinning wings):")
    print(f"  → Hover thrust:     {DRONE_WEIGHT_N:.2f} N")
    print(f"  → Induced velocity: {motor_only_v_induced:.2f} m/s")
    print(f"  → Motor power:      {motor_only_power:.1f} W ⚡ (100% baseline)")
    
    print(f"\n{'='*80}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n📊 BASELINE (ω = {baseline['omega']:.1f} rad/s):")
    print(f"  → Motor thrust:     {baseline['total_motor_thrust']:.2f} N (total)")
    print(f"  → Wing thrust:      {baseline['blade_lift']:.2f} N ({baseline['lift_fraction']*100:.0f}% of weight)")
    print(f"  → Wing power:       {baseline['power']:.1f} W")
    print(f"  → Motor power:      {baseline['motor_power']:.1f} W")
    print(f"  → Total power:      {baseline['total_system_power']:.1f} W")
    print(f"  → Energy savings:   {(motor_only_power - baseline['total_system_power'])/motor_only_power*100:.1f}% vs motor-only")
    print(f"  → Wing/Motor ratio: {baseline['wing_assist_ratio']:.2f}x")
    
    print(f"\n⭐ BEST AERODYNAMIC EFFICIENCY (max thrust/watt): {optimal_aero_eff['omega']:.1f} rad/s ({optimal_aero_eff['rpm']:.0f} RPM)")
    print(f"  → Aero efficiency:  {optimal_aero_eff['aero_efficiency']:.2f} N/W ⭐ (best output per input power)")
    print(f"  → Wing thrust:      {optimal_aero_eff['blade_lift']:.2f} N ({optimal_aero_eff['lift_fraction']*100:.0f}% of weight)")
    print(f"  → Wing power:       {optimal_aero_eff['power']:.1f} W")
    print(f"  → Motor thrust:     {optimal_aero_eff['total_motor_thrust']:.2f} N")
    print(f"  → Trade-off: Low power but needs more motor assist")
    
    print(f"\n🎯 BEST SYSTEM EFFICIENCY (max motor reduction): {optimal_assist['omega']:.1f} rad/s ({optimal_assist['rpm']:.0f} RPM)")
    print(f"  → Wing/Motor ratio: {optimal_assist['wing_assist_ratio']:.2f}x 🎯 (wings do {optimal_assist['wing_assist_ratio']:.1f}× more work)")
    print(f"  → Motor reduction:  {optimal_assist['motor_reduction']:.1f}% (saves {optimal_assist['motor_reduction']:.0f}% motor thrust)")
    print(f"  → Motor thrust:     {optimal_assist['total_motor_thrust']:.2f} N (vs {DRONE_WEIGHT_N:.2f} N without wings)")
    print(f"  → Wing thrust:      {optimal_assist['blade_lift']:.2f} N ({optimal_assist['lift_fraction']*100:.0f}% of weight)")
    print(f"  → Wing power:       {optimal_assist['power']:.1f} W")
    print(f"  → Aero efficiency:  {optimal_assist['aero_efficiency']:.2f} N/W")
    
    print(f"\n★ MINIMUM MOTOR THRUST: {optimal_min_motor['omega']:.1f} rad/s ({optimal_min_motor['rpm']:.0f} RPM)")
    print(f"  → Motor thrust:     {optimal_min_motor['total_motor_thrust']:.2f} N ★ (lowest motor load)")
    print(f"  → Motor reduction:  {optimal_min_motor['motor_reduction']:.1f}%")
    print(f"  → Wing thrust:      {optimal_min_motor['blade_lift']:.2f} N ({optimal_min_motor['lift_fraction']*100:.0f}% of weight)")
    print(f"  → Wing power:       {optimal_min_motor['power']:.1f} W")
    print(f"  → Motor power:      {optimal_min_motor['motor_power']:.1f} W")
    print(f"  → Total power:      {optimal_min_motor['total_system_power']:.1f} W")
    print(f"  → Energy savings:   {(motor_only_power - optimal_min_motor['total_system_power'])/motor_only_power*100:.1f}% vs motor-only")
    
    print(f"\n💡 MINIMUM TOTAL POWER: {min_power['omega']:.1f} rad/s ({min_power['rpm']:.0f} RPM)")
    print(f"  → Total power:      {min_power['total_system_power']:.1f} W 💡 (MOST ENERGY EFFICIENT)")
    print(f"  → Wing power:       {min_power['power']:.1f} W")
    print(f"  → Motor power:      {min_power['motor_power']:.1f} W")
    print(f"  → Energy savings:   {(motor_only_power - min_power['total_system_power'])/motor_only_power*100:.1f}% vs motor-only ⚡")
    print(f"  → Wing thrust:      {min_power['blade_lift']:.2f} N ({min_power['lift_fraction']*100:.0f}% of weight)")
    print(f"  → Motor thrust:     {min_power['total_motor_thrust']:.2f} N")
    
    print(f"\n{'='*80}")
    print(f"  KEY INSIGHTS")
    print(f"{'='*80}")
    print(f"  ⚡ ENERGY: Spinning saves {(motor_only_power - min_power['total_system_power'])/motor_only_power*100:.0f}% power at {min_power['omega']:.0f} rad/s")
    print(f"     • Motor-only: {motor_only_power:.1f} W")
    print(f"     • With spinning: {min_power['total_system_power']:.1f} W (wing {min_power['power']:.0f}W + motor {min_power['motor_power']:.0f}W)")
    print(f"  ")
    print(f"  📊 TRADE-OFFS:")
    print(f"     • Low ω ({optimal_aero_eff['omega']:.0f} rad/s): Best aero efficiency, saves {(motor_only_power - optimal_aero_eff['total_system_power'])/motor_only_power*100:.0f}% energy")
    print(f"     • Optimal ω ({min_power['omega']:.0f} rad/s): MAXIMUM energy savings ({(motor_only_power - min_power['total_system_power'])/motor_only_power*100:.0f}%)")
    print(f"     • High ω ({optimal_assist['omega']:.0f} rad/s): Best motor assist, saves {(motor_only_power - optimal_assist['total_system_power'])/motor_only_power*100:.0f}% energy")
    print(f"  ")
    if min_power['total_system_power'] < motor_only_power:
        print(f"  ✅ VERDICT: Spinning IS worth it! Saves up to {(motor_only_power - min_power['total_system_power'])/motor_only_power*100:.0f}% energy")
    else:
        print(f"  ❌ VERDICT: Spinning NOT worth it. Uses {(min_power['total_system_power'] - motor_only_power)/motor_only_power*100:.0f}% MORE energy than motor-only")
    
    # =========================================================================
    # COMPARISON WITH sim_spinning_wing.py (fixed 44° tilt, PID controller)
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"  COMPARISON: ANALYSIS vs sim_spinning_wing.py")
    print(f"{'='*80}")
    print(f"  sim_spinning_wing.py uses fixed tilt = 44°, PID controller")
    print(f"  (KP_ALT={KP_ALT}, KD_ALT={KD_ALT}, KI_ALT={KI_ALT})")
    print(f"  At steady-state: z_vel ≈ 0, integral compensates, z_error → 0")
    print(f"  So equilibrium is: motor_vert + wing_thrust = weight (pure force balance)")
    
    # Run analytical equilibrium at sim's fixed tilt = 44°
    SIM_TILT_DEG = 44.0
    result_44 = find_equilibrium_omega(SIM_TILT_DEG, blade_df, n_blades, xflr_polars=xflr_polars)
    
    # Sim steady-state values from sim_log_spinning.json (30s run, last 1000 steps avg)
    sim_omega = 9.389
    sim_alt = 1.499
    sim_wing_thrust = 5.413
    sim_motor_total = 3.388
    sim_wing_torque = 1.070
    sim_wing_power = 10.04
    sim_tilt = 44.0
    
    if result_44 is not None:
        r = result_44
        print(f"\n  At {SIM_TILT_DEG}° tilt (sim_spinning_wing.py fixed tilt):")
        print(f"  {'Metric':<20} {'Analysis':>12} {'Simulation':>12} {'Delta':>12}")
        print(f"  {'-'*56}")
        print(f"  {'omega (rad/s)':<20} {r['omega']:>12.2f} {sim_omega:>12.2f} {r['omega'] - sim_omega:>+12.2f}")
        print(f"  {'Wing thrust (N)':<20} {r['wing_thrust']:>12.2f} {sim_wing_thrust:>12.2f} {r['wing_thrust'] - sim_wing_thrust:>+12.2f}")
        print(f"  {'Motor total (N)':<20} {r['motor_thrust']:>12.2f} {sim_motor_total:>12.2f} {r['motor_thrust'] - sim_motor_total:>+12.2f}")
        print(f"  {'Motor/motor (N)':<20} {r['motor_per']:>12.2f} {sim_motor_total/4:>12.2f} {r['motor_per'] - sim_motor_total/4:>+12.2f}")
        print(f"  {'Wing torque (Nm)':<20} {r['wing_torque']:>12.3f} {sim_wing_torque:>12.3f} {r['wing_torque'] - sim_wing_torque:>+12.3f}")
        print(f"  {'Wing power (W)':<20} {r['wing_power']:>12.2f} {sim_wing_power:>12.2f} {r['wing_power'] - sim_wing_power:>+12.2f}")
        print(f"  {'Wing fraction (%)':<20} {r['wing_fraction']:>12.1f} {sim_wing_thrust/DRONE_WEIGHT_N*100:>12.1f} {r['wing_fraction'] - sim_wing_thrust/DRONE_WEIGHT_N*100:>+12.1f}")
    else:
        print(f"\n  ⚠ No equilibrium found at {SIM_TILT_DEG}° tilt")
    
    # Plot results
    plot_results(sweep_df, n_blades)
    
    print(f"\n{'='*80}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
