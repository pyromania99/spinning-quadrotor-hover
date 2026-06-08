import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Directory containing polar files
polar_dir = '03_simulation\mujoco\polars'

# Weighting factors
WEIGHT_EFFICIENCY = 0.50   # 50% for Cl/Cd
WEIGHT_LIFT = 0.30         # 30% for Cl
WEIGHT_CP = 0.20           # 20% for Cp (min pressure coefficient)

# Parse all polar files
airfoil_data = defaultdict(lambda: defaultdict(list))

for filename in os.listdir(polar_dir):
    if not filename.endswith('.txt'):
        continue
    
    # Extract airfoil name and Reynolds number from filename
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
        
        # Extract CL and CD columns
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        cl = data[:, 1]
        cd = data[:, 2]
        cpmin = data[:, 7]  # Minimum pressure coefficient
        
        # Calculate CL/CD ratio, avoiding division by zero
        cl_cd = np.where(cd > 1e-6, cl / cd, 0)
        
        # Find optimal (maximum) CL/CD
        max_idx = np.argmax(cl_cd)
        max_cl_cd = cl_cd[max_idx]
        optimal_cl = cl[max_idx]
        optimal_cp = cpmin[max_idx]
        
        airfoil_data[airfoil_name][re_value] = (max_cl_cd, optimal_cl, optimal_cp)
        
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

# Collect all Cl/Cd, Cl, and Cp values for normalization
all_cl_cd = []
all_cl = []
all_cp = []

for airfoil, re_dict in filtered_airfoils.items():
    for re_val, (cl_cd, cl, cp) in re_dict.items():
        all_cl_cd.append(cl_cd)
        all_cl.append(cl)
        all_cp.append(cp)

# Find min/max for normalization
min_cl_cd = min(all_cl_cd)
max_cl_cd = max(all_cl_cd)
min_cl = min(all_cl)
max_cl = max(all_cl)
min_cp = min(all_cp)
max_cp = max(all_cp)

print(f"\nNormalization ranges:")
print(f"  Cl/Cd: {min_cl_cd:.2f} - {max_cl_cd:.2f}")
print(f"  Cl: {min_cl:.3f} - {max_cl:.3f}")
print(f"  Cp: {min_cp:.3f} - {max_cp:.3f}")
print(f"\nWeighting: {WEIGHT_EFFICIENCY*100:.1f}% Efficiency + {WEIGHT_LIFT*100:.1f}% Lift + {WEIGHT_CP*100:.1f}% Cp")

# Calculate weighted scores
weighted_scores = {}
for airfoil, re_dict in filtered_airfoils.items():
    weighted_scores[airfoil] = {}
    for re_val, (cl_cd, cl, cp) in re_dict.items():
        # Normalize to 0-1 range
        norm_cl_cd = (cl_cd - min_cl_cd) / (max_cl_cd - min_cl_cd) if max_cl_cd > min_cl_cd else 0
        norm_cl = (cl - min_cl) / (max_cl - min_cl) if max_cl > min_cl else 0
        # For Cp, higher (less negative) is better, so normalize normally
        norm_cp = (cp - min_cp) / (max_cp - min_cp) if max_cp > min_cp else 0
        
        # Calculate weighted score
        weighted_score = WEIGHT_EFFICIENCY * norm_cl_cd + WEIGHT_LIFT * norm_cl + WEIGHT_CP * norm_cp
        weighted_scores[airfoil][re_val] = weighted_score

# Create the weighted performance plot
fig = plt.figure(figsize=(14, 8))

# Plot each airfoil
for airfoil in sorted(weighted_scores.keys()):
    re_values = sorted(weighted_scores[airfoil].keys())
    score_values = [weighted_scores[airfoil][re] for re in re_values]
    
    plt.plot(re_values, score_values, marker='o', linewidth=2, markersize=6, label=airfoil)

plt.xlabel('Reynolds Number (×10⁶)', fontsize=12, fontweight='bold')
plt.ylabel('Weighted Performance Score', fontsize=12, fontweight='bold')
plt.title(f'Weighted Performance vs Reynolds Number\n({WEIGHT_EFFICIENCY*100:.0f}% Efficiency [Cl/Cd] + {WEIGHT_LIFT*100:.0f}% Lift [Cl] + {WEIGHT_CP*100:.0f}% Cp, Normalized)\nRe range: 0.001-0.080 × 10⁶', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.ylim([0, 1.05])  # Score is normalized 0-1
plt.tight_layout()

# Save the plot
output_file = 'weighted_performance_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as: {output_file}")

# Print summary statistics
print("\n" + "="*80)
print("WEIGHTED PERFORMANCE RANKINGS")
print("="*80)

# Find best performers at each Re
re_all = sorted(set(re_val for scores in weighted_scores.values() for re_val in scores.keys()))

for re_val in re_all:
    airfoil_scores = []
    for airfoil, scores in weighted_scores.items():
        if re_val in scores:
            airfoil_scores.append((airfoil, scores[re_val]))
    
    airfoil_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nRe = {re_val:.3f} × 10⁶:")
    for i, (airfoil, score) in enumerate(airfoil_scores[:3]):
        cl_cd_val, cl_val, cp_val = filtered_airfoils[airfoil][re_val]
        print(f"  {i+1}. {airfoil}: Score={score:.3f} (Cl/Cd={cl_cd_val:.1f}, Cl={cl_val:.3f}, Cp={cp_val:.3f})")

# Overall best performers (average score across all Re)
print("\n" + "="*80)
print("OVERALL AVERAGE PERFORMANCE")
print("="*80)

avg_scores = []
for airfoil, scores in weighted_scores.items():
    avg_score = np.mean(list(scores.values()))
    re_values = sorted(scores.keys())
    max_score = max(scores.values())
    max_re = [re for re, s in scores.items() if s == max_score][0]
    
    avg_scores.append((airfoil, avg_score, max_score, max_re))

avg_scores.sort(key=lambda x: x[1], reverse=True)

print("\nRanked by average weighted score:")
for i, (airfoil, avg, peak, peak_re) in enumerate(avg_scores):
    cl_cd_peak, cl_peak, cp_peak = filtered_airfoils[airfoil][peak_re]
    print(f"{i+1}. {airfoil}:")
    print(f"   Average Score: {avg:.3f}")
    print(f"   Peak Score: {peak:.3f} at Re={peak_re:.3f} (Cl/Cd={cl_cd_peak:.1f}, Cl={cl_peak:.3f}, Cp={cp_peak:.3f})")
