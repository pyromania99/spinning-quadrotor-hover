
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
DATA_DIR = "xflr_polars"
RESULTS_DIR = "results"
SUMMARY_FILE = os.path.join(RESULTS_DIR, "analysis_summary.csv")

# Regex for filename parsing
# Format: AG14_T1_Re0.021_M0.00_N9.0.txt
# We want to capture the Airfoil name (start) and Re value
FILENAME_REGEX = re.compile(r"^(?P<airfoil>.*)_T\d+_Re(?P<re>\d+\.\d+)_M.*\.txt$")

def parse_file(filepath):
    """
    Parses a single XFLR5 polar file.
    Returns a DataFrame with columns: alpha, CL, CD, Cpmin, etc.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the header line index
    header_idx = -1
    for i, line in enumerate(lines):
        if "alpha" in line and "CL" in line:
            header_idx = i
            break
            
    if header_idx == -1:
        return None
        
    # Read data starting from header
    # The file uses fixed width or whitespace separation
    try:
        # Skip to the data (header_idx is "alpha" line, header_idx+1 is "----", so start at header_idx+2)
        # using header=None to get integer columns
        df = pd.read_csv(filepath, sep=r'\s+', skiprows=header_idx+2, header=None, engine='python')
        
        # Mapping based on observation of file format:
        # Col 0: alpha
        # Col 1: CL
        # Col 2: CD
        # Col 7: Cpmin (Verified from file content inspection where -0.4147 is at idx 7)
        
        # Check if we have enough columns
        if df.shape[1] < 8:
            print(f"Skipping {filepath}: Not enough columns ({df.shape[1]})")
            return None
            
        column_mapping = {0: 'alpha', 1: 'CL', 2: 'CD', 7: 'Cpmin'}
        df.rename(columns=column_mapping, inplace=True)
        
        # Keep only relevant columns to avoid confusion
        relevant_cols = ['alpha', 'CL', 'CD', 'Cpmin']
        # If there are other columns e.g. from previous bad parsing, drop them or just select these
        df = df[relevant_cols]
            
        # Ensure numeric columns
        for col in relevant_cols:
             df[col] = pd.to_numeric(df[col], errors='coerce')
                
        return df
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def main():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        
    all_data = []
    
    # scan directory
    print(f"Scanning {DATA_DIR}...")
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    
    for filename in files:
        match = FILENAME_REGEX.match(filename)
        if match:
            airfoil = match.group("airfoil")
            re_val_str = match.group("re")
            
            # Calculate actual Re number (the filename often has 'Re0.021' which implies millions usually, 
            # but let's stick to the string representation for grouping to match exactly, 
            # or convert to float for sorting)
            try:
                re_val = float(re_val_str)
            except ValueError:
                continue
                
            filepath = os.path.join(DATA_DIR, filename)
            df = parse_file(filepath)
            
            if df is not None:
                # Add metadata columns
                df['Airfoil'] = airfoil
                df['Re_str'] = re_val_str
                df['Re'] = re_val
                all_data.append(df)
    
    if not all_data:
        print("No valid data found.")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    
    # Calculate CL/CD
    full_df['CL_CD'] = full_df['CL'] / full_df['CD']
    
    # Analysis by Reynolds Number
    re_groups = full_df.groupby('Re')
    
    summary_rows = []
    
    for re_val, group in re_groups:
        print(f"\nAnalyzing Re = {re_val:.3f} million (approx)...")
        # Ensure we don't divide by zero or have empty groups
        if group.empty:
            continue
            
        # 1. Best CL
        max_cl_idx = group['CL'].idxmax()
        best_cl = group.loc[max_cl_idx]
        
        # 2. Best CL/CD
        # Filter out NaN or infinite CL/CD
        valid_clcd = group[np.isfinite(group['CL_CD'])]
        if valid_clcd.empty:
            best_clcd = None
        else:
            max_clcd_idx = valid_clcd['CL_CD'].idxmax()
            best_clcd = valid_clcd.loc[max_clcd_idx]
            
        # 3. Lowest (Most Negative) Cpmin
        min_cp_idx = group['Cpmin'].idxmin()
        best_cp = group.loc[min_cp_idx]
        
        # --- Optimization Cost Function ---
        # Normalize and Score
        # We need the global max/min for this Re group to normalize
        max_cl_group = group['CL'].max()
        max_clcd_group = valid_clcd['CL_CD'].max() if not valid_clcd.empty else 1.0
        min_cp_group = group['Cpmin'].min() # This is a negative number
        
        # Calculate 'Score' for every row in the group
        # Helper for handling potential zeros or NaNs
        def calc_score(row):
            # CL Score: Higher is better (30%)
            s_cl = (row['CL'] / max_cl_group) if max_cl_group != 0 else 0
            
            # CL/CD Score: Higher is better (50%)
            s_clcd = 0
            if np.isfinite(row['CL_CD']) and max_clcd_group != 0:
                 s_clcd = row['CL_CD'] / max_clcd_group
            
            # Cpmin Score: More negative is better (target is min_cp_group) (20%)
            # Both row['Cpmin'] and min_cp_group should be negative.
            # a value of -10 is better than -5.
            # min_cp_group is say -10. row is -5. ratio = 0.5.
            # min_cp_group is -10. row is -10. ratio = 1.0.
            s_cp = 0
            if min_cp_group != 0:
                s_cp = row['Cpmin'] / min_cp_group
            
            # Weighted Sum: 30% CL, 50% L/D, 20% Cpmin
            return (0.3 * s_cl) + (0.5 * s_clcd) + (0.2 * s_cp)

        group = group.copy()
        group['Score'] = group.apply(calc_score, axis=1)
        
        # Top 10 by Score
        top_n = 10
        top_performers = group.sort_values(by='Score', ascending=False).head(top_n)
        
        # Append to summary
        summary_rows.append({
            'Re': re_val,
            'Criteria': 'Max CL',
            'Airfoil': best_cl['Airfoil'],
            'Alpha': best_cl['alpha'],
            'Value': best_cl['CL'],
            'Details': f"CD={best_cl['CD']:.4f}"
        })
        
        if best_clcd is not None:
             summary_rows.append({
                'Re': re_val,
                'Criteria': 'Max CL/CD',
                'Airfoil': best_clcd['Airfoil'],
                'Alpha': best_clcd['alpha'],
                'Value': best_clcd['CL_CD'],
                'Details': f"CL={best_clcd['CL']:.4f}"
            })
            
        summary_rows.append({
            'Re': re_val,
            'Criteria': 'Min Cpmin',
            'Airfoil': best_cp['Airfoil'],
            'Alpha': best_cp['alpha'],
            'Value': best_cp['Cpmin'],
            'Details': '-'
        })
        
        # Append Top Optimized
        for i, (idx, row) in enumerate(top_performers.iterrows()):
            summary_rows.append({
                'Re': re_val,
                'Criteria': f'Optimized #{i+1}',
                'Airfoil': row['Airfoil'],
                'Alpha': row['alpha'],
                'Value': row['Score'],
                'Details': f"CL={row['CL']:.2f}, L/D={row['CL_CD']:.1f}, Cp={row['Cpmin']:.2f}"
            })

        # --- Plotting ---
        # We plot all airfoils in the group
        # To avoid overcrowding, we can set alpha value or linewidth, or just plot lines.
        # But user asked for "graph of all xflr polars".
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Polars at Re ~ {re_val} M', fontsize=16)
        
        unique_airfoils = group['Airfoil'].unique()
        
        for airfoil in unique_airfoils:
            subset = group[group['Airfoil'] == airfoil].sort_values(by='alpha')
            
            # CL vs Alpha
            axes[0].plot(subset['alpha'], subset['CL'], label=airfoil)
            
            # CL/CD vs Alpha
            axes[1].plot(subset['alpha'], subset['CL_CD'], label=airfoil)
            
            # Cpmin vs Alpha
            axes[2].plot(subset['alpha'], subset['Cpmin'], label=airfoil)
            
        axes[0].set_title('CL vs Alpha')
        axes[0].set_xlabel('Alpha (deg)')
        axes[0].set_ylabel('CL')
        axes[0].grid(True)
        # axes[0].legend(fontsize='small', loc='best') # Legend might be too big if many airfoils

        axes[1].set_title('CL/CD vs Alpha')
        axes[1].set_xlabel('Alpha (deg)')
        axes[1].set_ylabel('CL/CD')
        axes[1].grid(True)

        axes[2].set_title('Cpmin vs Alpha')
        axes[2].set_xlabel('Alpha (deg)')
        axes[2].set_ylabel('Cpmin')
        axes[2].grid(True)
        
        # Handling Legend - maybe only for one plot or outside
        # If there are <= 10 airfoils, show legend. Else, maybe too crowded.
        if len(unique_airfoils) <= 15:
            axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        # Retrieve the string representation of Re from the dataframe for this group
        try:
             current_re_str = group['Re_str'].iloc[0]
        except:
             current_re_str = str(re_val)
             
        plot_filename = f"polars_Re{current_re_str}.png"
        plot_path = os.path.join(RESULTS_DIR, plot_filename)
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Saved plot to {plot_path}")

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_FILE, index=False)
    print(f"\nAnalysis complete. Summary saved to {SUMMARY_FILE}")
    
    # Print a nice table for the user
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    print("\n--- Summary of Best Performers ---")
    print(summary_df)

if __name__ == "__main__":
    main()
