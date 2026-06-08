"""
Polar Comparison Plots for Airfoil Selection
=============================================
Graph 1: Cl/Cd vs alpha, faceted by Re
Graph 2: Cl vs Cd (drag polar), faceted by Re

Airfoils: franky2, franky, naca4412, s1223, sd7037
Re range: 0.040 to 0.200 (×10^6)
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = "xflr_polars"

# Airfoil name mapping: display_name -> filename prefix
AIRFOILS = {
    "Franky 2":  "franky 2",
    "Franky":    "franky",
    "NACA 4412": "NACA 4412",
    "S1223":     "S1223",
    "SD7037":    "SD7037-092-88",
}

# Re values to plot (×10^6, matching filename convention e.g. Re0.080)
RE_VALUES = [0.040, 0.080, 0.100]

# Alpha range of interest
ALPHA_MIN = -2
ALPHA_MAX = 15

# ─── Styling ─────────────────────────────────────────────────────────────────
# High-contrast colorblind-safe palette
COLORS = {
    "Franky 2":  "#0077BB",  # Strong blue
    "Franky":    "#EE3377",  # Magenta/pink
    "NACA 4412": "#009988",  # Teal
    "S1223":     "#EE7733",  # Orange
    "SD7037":    "#332288",  # Indigo
}

# ─── Parsing ─────────────────────────────────────────────────────────────────
FILENAME_REGEX = re.compile(
    r"^(?P<airfoil>.*)_T\d+_Re(?P<re>\d+\.\d+)_M.*\.txt$"
)

def parse_polar(filepath):
    """Parse a single XFLR5 polar file into a DataFrame."""
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
        df = pd.read_csv(
            filepath, sep=r'\s+', skiprows=header_idx + 2,
            header=None, engine='python'
        )
        if df.shape[1] < 8:
            return None

        df.rename(columns={0: 'alpha', 1: 'CL', 2: 'CD', 7: 'Cpmin'}, inplace=True)
        df = df[['alpha', 'CL', 'CD', 'Cpmin']]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        # Compute CL/CD
        df['CL_CD'] = df['CL'] / df['CD']

        return df
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return None


def load_all_polars():
    """Load polar data for all airfoils and Re values."""
    data = {}  # data[(display_name, re_val)] = DataFrame
    base = os.path.dirname(os.path.abspath(__file__))
    polar_dir = os.path.join(base, DATA_DIR)

    for display_name, file_prefix in AIRFOILS.items():
        for re_val in RE_VALUES:
            # Build expected filename
            re_str = f"{re_val:.3f}"
            pattern = f"{file_prefix}_T1_Re{re_str}_M0.00_N9.0.txt"
            fpath = os.path.join(polar_dir, pattern)

            if not os.path.isfile(fpath):
                # Try alternate Re formats (e.g. 0.051)
                print(f"  Missing: {pattern}")
                continue

            df = parse_polar(fpath)
            if df is not None and len(df) > 0:
                # Filter alpha range
                df = df[(df['alpha'] >= ALPHA_MIN) & (df['alpha'] <= ALPHA_MAX)]
                data[(display_name, re_val)] = df

    return data


def re_label(re_val):
    """Human-readable Re label."""
    return f"Re = {int(re_val * 1e6 / 1000)}k"


# ─── Graph 1: Cl/Cd vs Alpha, faceted by Re ─────────────────────────────────
def plot_clcd_vs_alpha(data):
    n_re = len(RE_VALUES)
    fig, axes = plt.subplots(n_re, 1, figsize=(10, 3.0 * n_re),
                              sharex=True, squeeze=False)
    fig.suptitle("Aerodynamic Efficiency (Cl/Cd) vs Angle of Attack",
                 fontsize=16, fontweight='bold', y=0.995)

    for idx, re_val in enumerate(RE_VALUES):
        ax = axes[idx, 0]

        for display_name in AIRFOILS:
            key = (display_name, re_val)
            if key not in data:
                continue
            df = data[key]
            ax.plot(df['alpha'], df['CL_CD'],
                    color=COLORS[display_name],
                    linewidth=2.0, alpha=0.9,
                    label=display_name)

            # Mark peak CL/CD
            peak_idx = df['CL_CD'].idxmax()
            peak_alpha = df.loc[peak_idx, 'alpha']
            peak_val = df.loc[peak_idx, 'CL_CD']
            ax.plot(peak_alpha, peak_val, marker='*', color=COLORS[display_name],
                    markersize=12, zorder=5, markeredgecolor='black',
                    markeredgewidth=0.5)

        # Formatting
        label = re_label(re_val)
        # Highlight primary Re range
        if re_val >= 0.080:
            ax.set_facecolor('#f8f8ff')
            label += "  ★"

        ax.set_ylabel("Cl / Cd", fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold', loc='left',
                     color='#333')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which='minor', alpha=0.1, linewidth=0.3)

        if idx == 0:
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9,
                      ncol=3, handlelength=2.5)

    axes[-1, 0].set_xlabel("Angle of Attack α (°)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


# ─── Graph 2: Cl vs Cd (Drag Polar), faceted by Re ──────────────────────────
def plot_cl_vs_cd(data):
    n_re = len(RE_VALUES)
    fig, axes = plt.subplots(n_re, 1, figsize=(10, 3.0 * n_re),
                              sharex=False, squeeze=False)
    fig.suptitle("Drag Polar (Cl vs Cd)",
                 fontsize=16, fontweight='bold', y=0.995)

    for idx, re_val in enumerate(RE_VALUES):
        ax = axes[idx, 0]

        for display_name in AIRFOILS:
            key = (display_name, re_val)
            if key not in data:
                continue
            df = data[key]
            ax.plot(df['CD'], df['CL'],
                    color=COLORS[display_name],
                    linewidth=2.0, alpha=0.9,
                    label=display_name)

            # Mark best L/D point (tangent from origin)
            peak_idx = df['CL_CD'].idxmax()
            ax.plot(df.loc[peak_idx, 'CD'], df.loc[peak_idx, 'CL'],
                    marker='*', color=COLORS[display_name],
                    markersize=12, zorder=5, markeredgecolor='black',
                    markeredgewidth=0.5)

        # Formatting
        label = re_label(re_val)
        if re_val >= 0.080:
            ax.set_facecolor('#f8f8ff')
            label += "  ★"

        ax.set_ylabel("Cl", fontsize=11)
        ax.set_xlabel("Cd", fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold', loc='left',
                     color='#333')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which='minor', alpha=0.1, linewidth=0.3)

        if idx == 0:
            ax.legend(loc='lower right', fontsize=9, framealpha=0.9,
                      ncol=3, handlelength=2.5)

    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


# ─── Graph 3: Cpmin vs Alpha, faceted by Re ──────────────────────────────────
def plot_cpmin_vs_alpha(data):
    n_re = len(RE_VALUES)
    fig, axes = plt.subplots(n_re, 1, figsize=(10, 3.0 * n_re),
                              sharex=True, squeeze=False)
    fig.suptitle("Minimum Pressure Coefficient (−Cp,min) vs Angle of Attack",
                 fontsize=16, fontweight='bold', y=0.995)

    for idx, re_val in enumerate(RE_VALUES):
        ax = axes[idx, 0]

        for display_name in AIRFOILS:
            key = (display_name, re_val)
            if key not in data:
                continue
            df = data[key]
            # Plot -Cpmin (positive = stronger suction peak)
            ax.plot(df['alpha'], -df['Cpmin'],
                    color=COLORS[display_name],
                    linewidth=2.0, alpha=0.9,
                    label=display_name)

        # Formatting
        label = re_label(re_val)
        if re_val >= 0.080:
            ax.set_facecolor('#f8f8ff')
            label += "  ★"

        ax.set_ylabel("−Cp,min", fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold', loc='left',
                     color='#333')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which='minor', alpha=0.1, linewidth=0.3)

        if idx == 0:
            ax.legend(loc='upper left', fontsize=9, framealpha=0.9,
                      ncol=3, handlelength=2.5)

    axes[-1, 0].set_xlabel("Angle of Attack α (°)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


# ─── Compact combined figure for paper ────────────────────────────────────────
def plot_combined_compact(data):
    """3-column × N-row compact figure for paper inclusion."""
    n_re = len(RE_VALUES)
    fig, axes = plt.subplots(n_re, 3, figsize=(7.0, 2.2 * n_re),
                              squeeze=False)

    for idx, re_val in enumerate(RE_VALUES):
        ax_eff  = axes[idx, 0]  # CL/CD vs α
        ax_drag = axes[idx, 1]  # CL vs CD
        ax_cp   = axes[idx, 2]  # −Cpmin vs α

        for display_name in AIRFOILS:
            key = (display_name, re_val)
            if key not in data:
                continue
            df = data[key]
            c = COLORS[display_name]
            lbl = display_name if idx == 0 else None

            # Col 1: CL/CD vs alpha
            ax_eff.plot(df['alpha'], df['CL_CD'], color=c,
                        linewidth=1.4, alpha=0.9, label=lbl)
            peak_idx = df['CL_CD'].idxmax()
            ax_eff.plot(df.loc[peak_idx, 'alpha'], df.loc[peak_idx, 'CL_CD'],
                        marker='*', color=c, markersize=8, zorder=5,
                        markeredgecolor='black', markeredgewidth=0.3)

            # Col 2: CL vs CD
            ax_drag.plot(df['CD'], df['CL'], color=c,
                         linewidth=1.4, alpha=0.9, label=lbl)
            ax_drag.plot(df.loc[peak_idx, 'CD'], df.loc[peak_idx, 'CL'],
                         marker='*', color=c, markersize=8, zorder=5,
                         markeredgecolor='black', markeredgewidth=0.3)

            # Col 3: -Cpmin vs alpha
            ax_cp.plot(df['alpha'], -df['Cpmin'], color=c,
                       linewidth=1.4, alpha=0.9, label=lbl)

        # Row label
        label = re_label(re_val)
        if re_val >= 0.080:
            for ax in [ax_eff, ax_drag, ax_cp]:
                ax.set_facecolor('#f8f8ff')
            label += " ★"
        ax_eff.set_ylabel(label, fontsize=11, fontweight='bold')

        # Tick styling
        for ax in [ax_eff, ax_drag, ax_cp]:
            ax.tick_params(labelsize=9)
            ax.grid(True, alpha=0.25, linewidth=0.4)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Column headers
    axes[0, 0].set_title("Cl/Cd vs α", fontsize=12, fontweight='bold')
    axes[0, 1].set_title("Cl vs Cd", fontsize=12, fontweight='bold')
    axes[0, 2].set_title("−Cp,min vs α", fontsize=12, fontweight='bold')

    # Bottom axis labels
    axes[-1, 0].set_xlabel("α (°)", fontsize=11)
    axes[-1, 1].set_xlabel("Cd", fontsize=11)
    axes[-1, 2].set_xlabel("α (°)", fontsize=11)

    # Legend once at top
    axes[0, 1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.95),
                       fontsize=13, ncol=3, framealpha=0.9,
                       handlelength=1.5, columnspacing=1.2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(hspace=0.30, wspace=0.30)
    return fig


# ─── 2x2 Grid Plot with 3 rows per quadrant (one per Re) ─────────────────────
def plot_2x2_grid(data):
    """2×2 grid where each quadrant has 3 rows for each Re value."""
    from matplotlib.gridspec import GridSpec
    
    # Create figure with extra height for multiple rows
    # 252pt × 756pt = 3.5" × 10.5" (72 pt/inch)
    fig = plt.figure(figsize=(3.5, 10.5))
    
    # Create 6 rows × 2 columns grid (3 Re rows per quadrant × 2 quadrant rows)
    gs = GridSpec(6, 2, figure=fig, hspace=0.35, wspace=0.25,
                  top=0.96, bottom=0.04, left=0.08, right=0.98)
    
    n_re = len(RE_VALUES)
    
    # ═══ QUADRANT 1 (top-left): Cl/Cd vs Alpha ═══
    for idx, re_val in enumerate(RE_VALUES):
        ax = fig.add_subplot(gs[idx, 0])
        
        for display_name in AIRFOILS:
            key = (display_name, re_val)
            if key not in data:
                continue
            df = data[key]
            ax.plot(df['alpha'], df['CL_CD'],
                   color=COLORS[display_name],
                   linewidth=2.0, alpha=0.9,
                   label=display_name)
            
            # Mark peak CL/CD
            peak_idx = df['CL_CD'].idxmax()
            peak_alpha = df.loc[peak_idx, 'alpha']
            peak_val = df.loc[peak_idx, 'CL_CD']
            ax.plot(peak_alpha, peak_val, marker='*', 
                   color=COLORS[display_name],
                   markersize=10, zorder=5, 
                   markeredgecolor='black',
                   markeredgewidth=0.5)
        
        # Formatting
        label = re_label(re_val)
        if re_val >= 0.080:
            ax.set_facecolor('#f8f8ff')
            label += "  ★"
        
        ax.set_ylabel("Cl / Cd", fontsize=11)
        ax.set_title(label, fontsize=11, fontweight='bold', loc='left', color='#333')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which='minor', alpha=0.1, linewidth=0.3)
        
        if idx == 0:
            ax.set_title("Aerodynamic Efficiency (Cl/Cd)", fontsize=13, 
                        fontweight='bold', loc='center', pad=10)
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9, ncol=3)
        if idx == n_re - 1:
            ax.set_xlabel("Angle of Attack α (°)", fontsize=11)
    
    # ═══ QUADRANT 2 (top-right): Cl vs Cd (Drag Polar) ═══
    for idx, re_val in enumerate(RE_VALUES):
        ax = fig.add_subplot(gs[idx, 1])
        
        for display_name in AIRFOILS:
            key = (display_name, re_val)
            if key not in data:
                continue
            df = data[key]
            ax.plot(df['CD'], df['CL'],
                   color=COLORS[display_name],
                   linewidth=2.0, alpha=0.9,
                   label=display_name)
            
            # Mark best L/D point
            peak_idx = df['CL_CD'].idxmax()
            ax.plot(df.loc[peak_idx, 'CD'], df.loc[peak_idx, 'CL'],
                   marker='*', color=COLORS[display_name],
                   markersize=10, zorder=5, 
                   markeredgecolor='black',
                   markeredgewidth=0.5)
        
        # Formatting
        label = re_label(re_val)
        if re_val >= 0.080:
            ax.set_facecolor('#f8f8ff')
            label += "  ★"
        
        ax.set_ylabel("Cl", fontsize=11)
        ax.set_title(label, fontsize=11, fontweight='bold', loc='left', color='#333')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which='minor', alpha=0.1, linewidth=0.3)
        
        if idx == 0:
            ax.set_title("Drag Polar (Cl vs Cd)", fontsize=13, 
                        fontweight='bold', loc='center', pad=10)
            ax.legend(loc='lower right', fontsize=8, framealpha=0.9, ncol=3)
        if idx == n_re - 1:
            ax.set_xlabel("Cd", fontsize=11)
    
    # ═══ QUADRANT 3 (bottom-left): -Cpmin vs Alpha ═══
    for idx, re_val in enumerate(RE_VALUES):
        ax = fig.add_subplot(gs[idx + 3, 0])
        
        for display_name in AIRFOILS:
            key = (display_name, re_val)
            if key not in data:
                continue
            df = data[key]
            ax.plot(df['alpha'], -df['Cpmin'],
                   color=COLORS[display_name],
                   linewidth=2.0, alpha=0.9,
                   label=display_name)
        
        # Formatting
        label = re_label(re_val)
        if re_val >= 0.080:
            ax.set_facecolor('#f8f8ff')
            label += "  ★"
        
        ax.set_ylabel("−Cp,min", fontsize=11)
        ax.set_title(label, fontsize=11, fontweight='bold', loc='left', color='#333')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which='minor', alpha=0.1, linewidth=0.3)
        
        if idx == 0:
            ax.set_title("Minimum Pressure Coefficient (−Cp,min)", fontsize=13, 
                        fontweight='bold', loc='center', pad=10)
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9, ncol=3)
        if idx == n_re - 1:
            ax.set_xlabel("Angle of Attack α (°)", fontsize=11)
    
    # ═══ QUADRANT 4 (bottom-right): Cl vs Alpha ═══
    for idx, re_val in enumerate(RE_VALUES):
        ax = fig.add_subplot(gs[idx + 3, 1])
        
        for display_name in AIRFOILS:
            key = (display_name, re_val)
            if key not in data:
                continue
            df = data[key]
            ax.plot(df['alpha'], df['CL'],
                   color=COLORS[display_name],
                   linewidth=2.0, alpha=0.9,
                   label=display_name)
        
        # Formatting
        label = re_label(re_val)
        if re_val >= 0.080:
            ax.set_facecolor('#f8f8ff')
            label += "  ★"
        
        ax.set_ylabel("Cl", fontsize=11)
        ax.set_title(label, fontsize=11, fontweight='bold', loc='left', color='#333')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(True, which='minor', alpha=0.1, linewidth=0.3)
        
        if idx == 0:
            ax.set_title("Lift Coefficient (Cl vs α)", fontsize=13, 
                        fontweight='bold', loc='center', pad=10)
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9, ncol=3)
        if idx == n_re - 1:
            ax.set_xlabel("Angle of Attack α (°)", fontsize=11)
    
    fig.suptitle("Airfoil Performance Comparison", 
                 fontsize=18, fontweight='bold', y=0.99)
    
    # Add thick lines to demarcate quadrants
    # Vertical line between left and right quadrants
    fig.add_artist(plt.Line2D([0.53, 0.53], [0.04, 0.96], 
                              transform=fig.transFigure, 
                              color='black', linewidth=3, zorder=10))
    
    # Horizontal line between top and bottom quadrants
    fig.add_artist(plt.Line2D([0.08, 0.98], [0.50, 0.50], 
                              transform=fig.transFigure, 
                              color='black', linewidth=3, zorder=10))
    
    return fig


# ─── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading polars...")
    data = load_all_polars()
    print(f"Loaded {len(data)} polar files")

    # Print summary
    for (name, re_val), df in sorted(data.items()):
        peak_clcd = df['CL_CD'].max()
        peak_alpha = df.loc[df['CL_CD'].idxmax(), 'alpha']
        peak_cl = df.loc[df['CL_CD'].idxmax(), 'CL']
        print(f"  {name:12s} Re={int(re_val*1e6):>6d}  "
              f"Peak Cl/Cd={peak_clcd:6.1f} @ α={peak_alpha:5.1f}°  "
              f"(Cl={peak_cl:.3f})")

    # 2x2 Grid plot with all Re values
    print("\nPlotting 2x2 Grid with all Re values...")
    fig_grid = plot_2x2_grid(data)
    fig_grid.savefig("results/airfoil_2x2_grid.pdf",
                     bbox_inches='tight', facecolor='white')
    print("  Saved: results/airfoil_2x2_grid.png / .pdf")

    # Individual full-size plots
    print("\nPlotting Graph 1: Cl/Cd vs Alpha...")
    fig1 = plot_clcd_vs_alpha(data)
    fig1.savefig("results/clcd_vs_alpha_comparison.png", dpi=200,
                 bbox_inches='tight', facecolor='white')
    print("  Saved: results/clcd_vs_alpha_comparison.png")

    print("Plotting Graph 2: Cl vs Cd (Drag Polar)...")
    fig2 = plot_cl_vs_cd(data)
    fig2.savefig("results/cl_vs_cd_comparison.png", dpi=200,
                 bbox_inches='tight', facecolor='white')
    print("  Saved: results/cl_vs_cd_comparison.png")

    print("Plotting Graph 3: Cpmin vs Alpha...")
    fig3 = plot_cpmin_vs_alpha(data)
    fig3.savefig("results/cpmin_vs_alpha_comparison.png", dpi=200,
                 bbox_inches='tight', facecolor='white')
    print("  Saved: results/cpmin_vs_alpha_comparison.png")

    # Compact combined figure for paper
    print("Plotting combined compact figure...")
    fig4 = plot_combined_compact(data)
    fig4.savefig("results/airfoil_comparison_compact.png", dpi=300,
                 bbox_inches='tight', facecolor='white')
    fig4.savefig("results/airfoil_comparison_compact.pdf",
                 bbox_inches='tight', facecolor='white')
    print("  Saved: results/airfoil_comparison_compact.png / .pdf")

    print("Done!")
