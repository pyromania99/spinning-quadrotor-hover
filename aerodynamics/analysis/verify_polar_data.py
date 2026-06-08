"""
Verify Reynolds numbers and aerodynamic coefficients in optimized_design.txt
against actual XFLR5 polar data.
"""
import numpy as np
import os

# Constants
RHO = 1.225  # kg/m³ (air density at sea level)
MU = 1.81e-5  # kg/(m·s) (dynamic viscosity of air at 15°C)

def calculate_reynolds(velocity, chord):
    """Calculate Reynolds number: Re = ρ * V * c / μ"""
    return RHO * velocity * chord / MU

def read_xflr_polar(filepath):
    """Read XFLR5 polar file and extract alpha, Cl, Cd, Cl/Cd"""
    data = []
    in_data_section = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Look for data section start
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
                        cl_cd = cl / cd if cd > 0 else 0
                        data.append({
                            'alpha': alpha,
                            'Cl': cl,
                            'Cd': cd,
                            'Cl/Cd': cl_cd
                        })
                except (ValueError, IndexError):
                    continue
    
    return data

def find_closest_polar(airfoil, re_actual):
    """Find the closest Reynolds number polar file"""
    polar_dir = 'xflr_polars'
    
    # Map airfoil names
    if airfoil.lower() == 'sd7037':
        prefix = 'SD7037-092-88'
    else:
        prefix = 'franky'
    
    # List available polars
    available_polars = []
    for filename in os.listdir(polar_dir):
        if filename.startswith(prefix) and filename.endswith('.txt'):
            # Extract Re from filename like "franky_T1_Re0.060_M0.00_N9.0.txt"
            parts = filename.split('_Re')
            if len(parts) == 2:
                re_str = parts[1].split('_')[0]
                try:
                    re_file = float(re_str) * 1e6  # Convert to actual Re
                    available_polars.append((re_file, filename))
                except:
                    pass
    
    # Find closest Re
    if not available_polars:
        return None, None
    
    available_polars.sort(key=lambda x: abs(x[0] - re_actual))
    re_closest, filename = available_polars[0]
    
    return re_closest, os.path.join(polar_dir, filename)

def find_alpha_data(polar_data, alpha_target):
    """Find Cl, Cd, Cl/Cd at target alpha (with interpolation)"""
    if not polar_data:
        return None, None, None
    
    alphas = np.array([d['alpha'] for d in polar_data])
    cls = np.array([d['Cl'] for d in polar_data])
    cds = np.array([d['Cd'] for d in polar_data])
    cl_cds = np.array([d['Cl/Cd'] for d in polar_data])
    
    # Check if alpha is in range
    if alpha_target < alphas.min() or alpha_target > alphas.max():
        # Find closest
        idx = np.argmin(np.abs(alphas - alpha_target))
        return cls[idx], cds[idx], cl_cds[idx]
    
    # Interpolate
    cl = np.interp(alpha_target, alphas, cls)
    cd = np.interp(alpha_target, alphas, cds)
    cl_cd = cl / cd if cd > 0 else 0
    
    return cl, cd, cl_cd

# Read optimized_design.txt
sections = []
print("Reading optimized_design.txt...")
with open('optimized_design.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines[2:]:  # Skip header
        parts = line.split()
        if len(parts) >= 8:
            sections.append({
                'radius': float(parts[0]),
                'velocity': float(parts[1]),
                're_stated': parts[2],  # Keep as string (may have * or issues)
                'airfoil': parts[3],
                'alpha': float(parts[4].replace('~', '')),
                'chord': float(parts[5]),
                'cl_stated': float(parts[6]),
                'cl_cd_stated': float(parts[7])
            })

print("\n" + "="*100)
print("POLAR DATA VERIFICATION")
print("="*100)

for i, sec in enumerate(sections):
    print(f"\nSection {i+1}: r = {sec['radius']}m, c = {sec['chord']}m, α = {sec['alpha']}°")
    print("-" * 100)
    
    # Calculate actual Reynolds number
    re_actual = calculate_reynolds(sec['velocity'], sec['chord'])
    re_actual_millions = re_actual / 1e6
    
    print(f"  Stated Re:     {sec['re_stated']} × 10⁶")
    print(f"  Calculated Re: {re_actual_millions:.3f} × 10⁶  (V={sec['velocity']} m/s, c={sec['chord']} m)")
    
    re_error = "MATCH" if abs(float(sec['re_stated'].replace('*', '').replace('~', '')) - re_actual_millions) < 0.01 else "MISMATCH"
    print(f"  Reynolds Check: {re_error}")
    
    # Find closest polar file
    re_closest, polar_file = find_closest_polar(sec['airfoil'], re_actual)
    
    if polar_file:
        print(f"\n  Closest polar: {os.path.basename(polar_file)} (Re = {re_closest/1e6:.3f} × 10⁶)")
        
        # Read polar data
        polar_data = read_xflr_polar(polar_file)
        
        if polar_data:
            # Find Cl and Cl/Cd at stated alpha
            cl_polar, cd_polar, cl_cd_polar = find_alpha_data(polar_data, sec['alpha'])
            
            if cl_polar is not None:
                print(f"\n  At α = {sec['alpha']}°:")
                print(f"    Stated:   Cl = {sec['cl_stated']:.3f},  Cl/Cd = {sec['cl_cd_stated']:.1f}")
                print(f"    From polar: Cl = {cl_polar:.3f},  Cl/Cd = {cl_cd_polar:.1f},  Cd = {cd_polar:.4f}")
                
                cl_error = abs(sec['cl_stated'] - cl_polar) / cl_polar * 100
                cl_cd_error = abs(sec['cl_cd_stated'] - cl_cd_polar) / cl_cd_polar * 100
                
                print(f"    Cl error:    {cl_error:.1f}%")
                print(f"    Cl/Cd error: {cl_cd_error:.1f}%")
                
                if cl_error > 5 or cl_cd_error > 5:
                    print(f"    ⚠ WARNING: Significant mismatch detected!")
            else:
                print(f"  ⚠ Alpha {sec['alpha']}° not found in polar data")
        else:
            print(f"  ⚠ Could not read polar data from file")
    else:
        print(f"  ⚠ No polar file found for {sec['airfoil']}")

print("\n" + "="*100)
print("SUMMARY")
print("="*100)
print("""
Key Issues to Check:
1. Reynolds numbers should match: Re = (1.225 × V × c) / 1.81e-5
2. Cl values should match XFLR5 polars at stated alpha and Re
3. Cl/Cd ratios should match polar data (not estimated)
""")
