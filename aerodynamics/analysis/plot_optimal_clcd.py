import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Directory containing polar files
polar_dir = 'xflr_polars'

# Parse all polar files
airfoil_data = defaultdict(lambda: defaultdict(list))

for filename in os.listdir(polar_dir):
    if not filename.endswith('.txt'):
        continue
    
    # Extract airfoil name and Reynolds number from filename
    # Format: {airfoil}_T1_Re{re}_M0.00_N9.0.txt
    match = re.match(r'(.+)_T1_Re([\d.]+)_M0\.00_N9\.0\.txt', filename)
    if not match:
        continue
    
    airfoil_name = match.group(1)
    re_value = float(match.group(2))
    
    # Only consider Reynolds numbers in 0-0.080 range
    if re_value > 0.080:
        continue
    
    # Read the polar file
    filepath = os.path.join(polar_dir, filename)
    try:
        data = np.loadtxt(filepath, skiprows=12)
        if data.size == 0:
            continue
        
        # Extract CL and CD columns (columns 1 and 2)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        cl = data[:, 1]
        cd = data[:, 2]
        
        # Calculate CL/CD ratio, avoiding division by zero
        cl_cd = np.where(cd > 1e-6, cl / cd, 0)
        
        # Find optimal (maximum) CL/CD
        max_idx = np.argmax(cl_cd)
        max_cl_cd = cl_cd[max_idx]
        optimal_cl = cl[max_idx]
        
        airfoil_data[airfoil_name][re_value] = (max_cl_cd, optimal_cl)
        
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        continue

# Filter airfoils with more than 5 data points in the range
filtered_airfoils = {}
for airfoil, re_dict in airfoil_data.items():
    if len(re_dict) > 5:
        filtered_airfoils[airfoil] = re_dict

print(f"Found {len(filtered_airfoils)} airfoils with more than 5 Reynolds numbers in 0-0.080 range:")
for airfoil in sorted(filtered_airfoils.keys()):
    print(f"  {airfoil}: {len(filtered_airfoils[airfoil])} data points")

# Create the CL/CD plot
fig1 = plt.figure(figsize=(14, 8))

# Plot each airfoil
for airfoil in sorted(filtered_airfoils.keys()):
    re_values = sorted(filtered_airfoils[airfoil].keys())
    cl_cd_values = [filtered_airfoils[airfoil][re][0] for re in re_values]
    
    plt.plot(re_values, cl_cd_values, marker='o', linewidth=2, markersize=6, label=airfoil)

plt.xlabel('Reynolds Number (×10⁶)', fontsize=12, fontweight='bold')
plt.ylabel('Optimal CL/CD', fontsize=12, fontweight='bold')
plt.title('Optimal CL/CD vs Reynolds Number Comparison\n(Re range: 0.001-0.080 × 10⁶)', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.tight_layout()

# Save the plot
output_file1 = 'optimal_clcd_comparison.png'
plt.savefig(output_file1, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as: {output_file1}")

# Create the optimal CL plot
fig2 = plt.figure(figsize=(14, 8))

# Plot each airfoil
for airfoil in sorted(filtered_airfoils.keys()):
    re_values = sorted(filtered_airfoils[airfoil].keys())
    cl_values = [filtered_airfoils[airfoil][re][1] for re in re_values]
    
    plt.plot(re_values, cl_values, marker='s', linewidth=2, markersize=6, label=airfoil)

plt.xlabel('Reynolds Number (×10⁶)', fontsize=12, fontweight='bold')
plt.ylabel('CL at Optimal CL/CD', fontsize=12, fontweight='bold')
plt.title('Optimal Lift Coefficient vs Reynolds Number Comparison\n(CL at maximum CL/CD point, Re range: 0.001-0.080 × 10⁶)', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.tight_layout()

# Save the plot
output_file2 = 'optimal_cl_comparison.png'
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"Plot saved as: {output_file2}")

# Optionally show plots (comment out if running in batch mode)
# plt.show()

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

for airfoil in sorted(filtered_airfoils.keys()):
    re_values = sorted(filtered_airfoils[airfoil].keys())
    cl_cd_values = [filtered_airfoils[airfoil][re][0] for re in re_values]
    cl_values = [filtered_airfoils[airfoil][re][1] for re in re_values]
    
    print(f"\n{airfoil}:")
    print(f"  Reynolds range: {min(re_values):.3f} - {max(re_values):.3f}")
    print(f"  CL/CD range: {min(cl_cd_values):.2f} - {max(cl_cd_values):.2f}")
    print(f"  CL range: {min(cl_values):.3f} - {max(cl_values):.3f}")
    best_idx = cl_cd_values.index(max(cl_cd_values))
    print(f"  Best performance: CL/CD = {max(cl_cd_values):.2f} at Re = {re_values[best_idx]:.3f} (CL = {cl_values[best_idx]:.3f})")
