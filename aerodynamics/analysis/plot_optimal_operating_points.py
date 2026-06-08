"""
Visualize optimal operating points from analyze_polars.py methodology.
This approach uses per-Reynolds normalization and evaluates all alpha angles
to find the best practical operating point for each airfoil.
"""
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Constants
DATA_DIR = "03_simulation\mujoco\polars"
FILENAME_REGEX = re.compile(r"^(?P<airfoil>.*)_T\d+_Re(?P<re>\d+\.\d+)_M.*\.txt$")

# Weights for optimization
WEIGHT_CL = 0.30
WEIGHT_CLCD = 0.50
WEIGHT_CP = 0.20

def parse_file(filepath):
    """Parse XFLR5 polar file"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    header_idx = -1
    for i, line in enumerate(lines):
        if "alpha" in line and "CL" in line:
            header_idx = i
            break
    
    if header_idx == -1:
        return None
    
    try:
        df = pd.read_csv(filepath, sep=r'\s+', skiprows=header_idx+2, header=None, engine='python')
        
        if df.shape[1] < 8:
            return None
        
        column_mapping = {0: 'alpha', 1: 'CL', 2: 'CD', 7: 'Cpmin'}
        df.rename(columns=column_mapping, inplace=True)
        df = df[['alpha', 'CL', 'CD', 'Cpmin']]
        
        for col in ['alpha', 'CL', 'CD', 'Cpmin']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        return None

def find_optimal_operating_point(re_group_all_airfoils):
    """
    Find optimal operating point for EACH airfoil within a Reynolds group.
    Uses per-Reynolds normalization across ALL airfoils at this Re.
    This matches analyze_polars.py methodology exactly.
    """
    if re_group_all_airfoils.empty:
        return {}
    
    # Calculate CL/CD for all data
    re_group_all_airfoils = re_group_all_airfoils.copy()
    re_group_all_airfoils['CL_CD'] = re_group_all_airfoils['CL'] / re_group_all_airfoils['CD']
    
    # Filter valid CL/CD
    valid_data = re_group_all_airfoils[np.isfinite(re_group_all_airfoils['CL_CD'])].copy()
    if valid_data.empty:
        return {}
    
    # Per-Reynolds normalization using max/min across ALL airfoils at this Re
    max_cl_group = valid_data['CL'].max()
    max_clcd_group = valid_data['CL_CD'].max()
    min_cp_group = valid_data['Cpmin'].min()  # Most negative value
    
    def calc_score(row):
        # CL Score: Higher is better (30%)
        s_cl = (row['CL'] / max_cl_group) if max_cl_group != 0 else 0
        
        # CL/CD Score: Higher is better (50%)
        s_clcd = (row['CL_CD'] / max_clcd_group) if max_clcd_group != 0 else 0
        
        # Cpmin Score: More negative is better (20%)
        # Both row['Cpmin'] and min_cp_group are negative
        # min_cp_group is most negative (e.g., -10)
        # row with -10 gets score 1.0, row with -5 gets score 0.5
        s_cp = (row['Cpmin'] / min_cp_group) if min_cp_group != 0 else 0
        
        return (WEIGHT_CL * s_cl) + (WEIGHT_CLCD * s_clcd) + (WEIGHT_CP * s_cp)
    
    valid_data['Score'] = valid_data.apply(calc_score, axis=1)
    
    # Find best operating point for each airfoil
    results = {}
    for airfoil, airfoil_group in valid_data.groupby('Airfoil'):
        best_idx = airfoil_group['Score'].idxmax()
        results[airfoil] = airfoil_group.loc[best_idx]
    
    return results

# Parse all files
print("Analyzing polars with per-Reynolds optimization...")
all_data = []

files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]

for filename in files:
    match = FILENAME_REGEX.match(filename)
    if match:
        airfoil = match.group("airfoil")
        re_val_str = match.group("re")
        
        try:
            re_val = float(re_val_str)
        except ValueError:
            continue
        
        # Focus on operational Reynolds range 0.040-0.080
        if re_val < 0.040 or re_val > 0.080:
            continue
        
        filepath = os.path.join(DATA_DIR, filename)
        df = parse_file(filepath)
        
        if df is not None:
            df['Airfoil'] = airfoil
            df['Re'] = re_val
            all_data.append(df)

if not all_data:
    print("No valid data found.")
    exit()

full_df = pd.concat(all_data, ignore_index=True)

# Find optimal operating points for each airfoil at each Re
# Key: Group by Re first, then find best point for each airfoil within that Re group
optimal_results = defaultdict(dict)
airfoil_re_details = defaultdict(dict)

for re, re_group in full_df.groupby('Re'):
    # Find optimal operating point for each airfoil at this Re
    re_optima = find_optimal_operating_point(re_group)
    
    for airfoil, optimal in re_optima.items():
        optimal_results[airfoil][re] = optimal['Score']
        airfoil_re_details[airfoil][re] = {
            'score': optimal['Score'],
            'alpha': optimal['alpha'],
            'CL': optimal['CL'],
            'CL_CD': optimal['CL_CD'],
            'Cpmin': optimal['Cpmin']
        }

# Filter airfoils with more than 5 data points
filtered_airfoils = {airfoil: data for airfoil, data in optimal_results.items() if len(data) > 2}

print(f"\nFound {len(filtered_airfoils)} airfoils with >2 Reynolds numbers in operational range:")
for airfoil in sorted(filtered_airfoils.keys()):
    print(f"  {airfoil}: {len(filtered_airfoils[airfoil])} data points")

# Create the plot
fig = plt.figure(figsize=(14, 8))

for airfoil in sorted(filtered_airfoils.keys()):
    re_values = sorted(filtered_airfoils[airfoil].keys())
    score_values = [filtered_airfoils[airfoil][re] for re in re_values]
    
    plt.plot(re_values, score_values, marker='o', linewidth=2, markersize=6, label=airfoil)

plt.xlabel('Reynolds Number (×10⁶)', fontsize=22, fontweight='bold')
plt.ylabel('Optimal Operating Point Score', fontsize=24, fontweight='bold')
plt.title('Optimal Operating Point Performance vs Reynolds Number', 
          fontsize=26, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='lower right', fontsize=15)
plt.tight_layout()

output_file = 'optimal_operating_points_comparison.pdf'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as: {output_file}")

# Print rankings at key Reynolds numbers
print("\n" + "="*80)
print("OPTIMAL OPERATING POINTS BY REYNOLDS NUMBER")
print("="*80)

re_all = sorted(set(re_val for scores in filtered_airfoils.values() for re_val in scores.keys()))

for re_val in re_all:
    airfoil_scores = []
    for airfoil, scores in filtered_airfoils.items():
        if re_val in scores:
            details = airfoil_re_details[airfoil][re_val]
            airfoil_scores.append((airfoil, details))
    
    airfoil_scores.sort(key=lambda x: x[1]['score'], reverse=True)
    
    print(f"\nRe = {re_val:.3f} × 10⁶:")
    print(f"  Reference values: Max CL across all airfoils, Max L/D across all, Min Cp across all")
    for i, (airfoil, details) in enumerate(airfoil_scores[:5]):  # Show top 5
        print(f"  {i+1}. {airfoil:15s}: Score={details['score']:.3f} at α={details['alpha']:5.1f}°")
        print(f"      CL={details['CL']:.3f}, L/D={details['CL_CD']:5.1f}, Cp={details['Cpmin']:6.2f}")

# Overall rankings
print("\n" + "="*80)
print("OVERALL AVERAGE PERFORMANCE")
print("="*80)

avg_scores = []
for airfoil, scores in filtered_airfoils.items():
    avg_score = np.mean(list(scores.values()))
    max_score = max(scores.values())
    max_re = [re for re, s in scores.items() if s == max_score][0]
    avg_scores.append((airfoil, avg_score, max_score, max_re))

avg_scores.sort(key=lambda x: x[1], reverse=True)

print("\nRanked by average optimal operating point score:")
for i, (airfoil, avg, peak, peak_re) in enumerate(avg_scores):
    details = airfoil_re_details[airfoil][peak_re]
    print(f"{i+1}. {airfoil}:")
    print(f"   Average Score: {avg:.3f}")
    print(f"   Peak Score: {peak:.3f} at Re={peak_re:.3f} (α={details['alpha']:.1f}°, CL={details['CL']:.2f}, L/D={details['CL_CD']:.1f}, Cp={details['Cpmin']:.2f})")

