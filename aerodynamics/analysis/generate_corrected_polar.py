"""
Generate corrected optimized_design.txt with accurate Reynolds numbers
and aerodynamic coefficients from XFLR5 polars.
"""
import numpy as np
import os

# Constants
RHO = 1.225  # kg/m³
MU = 1.81e-5  # kg/(m·s)

def calculate_reynolds(velocity, chord):
    return RHO * velocity * chord / MU

def read_xflr_polar(filepath):
    """Read XFLR5 polar and return data"""
    data = []
    in_data_section = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            if 'alpha' in line.lower() and 'CL' in line:
                in_data_section = True
                continue
            
            if in_data_section:
                try:
                    parts = line.split()
                    if len(parts) >= 4:
                        alpha = float(parts[0])
                        cl = float(parts[1])
                        cd = float(parts[2])
                        data.append({'alpha': alpha, 'Cl': cl, 'Cd': cd})
                except (ValueError, IndexError):
                    continue
    
    return data

def find_polar_and_coeffs(airfoil, re_actual, alpha_target):
    """Find polar file and extract Cl, Cd at target alpha"""
    polar_dir = 'xflr_polars'
    
    # Map airfoil name
    if 'sd7037' in airfoil.lower():
        prefix = 'SD7037-092-88'
    else:
        prefix = 'franky'
    
    # Find available polars
    available = []
    for filename in os.listdir(polar_dir):
        if filename.startswith(prefix) and filename.endswith('.txt'):
            parts = filename.split('_Re')
            if len(parts) == 2:
                try:
                    re_file = float(parts[1].split('_')[0]) * 1e6
                    available.append((re_file, filename))
                except:
                    pass
    
    if not available:
        return None, None, None, None
    
    # Find closest Re
    available.sort(key=lambda x: abs(x[0] - re_actual))
    re_closest, filename = available[0]
    
    # Read polar
    polar_data = read_xflr_polar(os.path.join(polar_dir, filename))
    
    if not polar_data:
        return re_closest, None, None, None
    
    # Extract at target alpha
    alphas = np.array([d['alpha'] for d in polar_data])
    cls = np.array([d['Cl'] for d in polar_data])
    cds = np.array([d['Cd'] for d in polar_data])
    
    # Interpolate
    if alpha_target < alphas.min() or alpha_target > alphas.max():
        idx = np.argmin(np.abs(alphas - alpha_target))
        cl = cls[idx]
        cd = cds[idx]
    else:
        cl = np.interp(alpha_target, alphas, cls)
        cd = np.interp(alpha_target, alphas, cds)
    
    cl_cd = cl / cd if cd > 0 else 0
    
    return re_closest, cl, cd, cl_cd

# Define sections (assuming omega = 8 rad/s for velocity calculation)
# V = omega * r
omega = 8.0  # rad/s

sections = [
    {'r': 0.40, 'chord': 0.10, 'airfoil': 'SD7037', 'alpha': 8.6},
    {'r': 0.45, 'chord': 0.19, 'airfoil': 'franky', 'alpha': 10.5},
    {'r': 0.55, 'chord': 0.285, 'airfoil': 'franky', 'alpha': 10.0},
    {'r': 0.60, 'chord': 0.27, 'airfoil': 'franky', 'alpha': 10.0},
    {'r': 0.65, 'chord': 0.255, 'airfoil': 'franky', 'alpha': 10.0},
    {'r': 0.65, 'chord': 0.24, 'airfoil': 'franky', 'alpha': 10.0},
]

print("Generating corrected optimized_design.txt...")
print("\nCalculated values:\n")

corrected_lines = []
corrected_lines.append("Radius (m)\tVelocity (m/s)\tRe (×10⁶)\tAirfoil\t\tα (°)\tChord (m)\tCl\tCl/Cd\tCd\n")

for sec in sections:
    v = omega * sec['r']
    re = calculate_reynolds(v, sec['chord'])
    re_millions = re / 1e6
    
    # Find polar and get coefficients
    re_polar, cl, cd, cl_cd = find_polar_and_coeffs(sec['airfoil'], re, sec['alpha'])
    
    if cl is not None:
        # Format line
        line = f"{sec['r']:.2f}\t\t{v:.2f}\t\t{re_millions:.3f}\t\t{sec['airfoil']}\t\t{sec['alpha']:.1f}\t{sec['chord']:.3f}\t\t{cl:.3f}\t{cl_cd:.1f}\t{cd:.4f}\n"
        corrected_lines.append(line)
        
        print(f"r={sec['r']}m: V={v:.2f}m/s, Re={re_millions:.3f}×10⁶, α={sec['alpha']}° → Cl={cl:.3f}, Cl/Cd={cl_cd:.1f}, Cd={cd:.4f}")
        print(f"  (using polar at Re={re_polar/1e6:.3f}×10⁶)")
    else:
        print(f"r={sec['r']}m: ERROR - Could not find polar data!")

# Write corrected file
with open('optimized_design_corrected.txt', 'w', encoding='utf-8') as f:
    f.writelines(corrected_lines)

print(f"\n✓ Corrected file written to: optimized_design_corrected.txt")
print("\nNOTE: This uses omega = 8 rad/s for velocity calculation (V = ω × r)")
