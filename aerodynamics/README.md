# Aerodynamics

Airfoil selection and aerodynamic characterisation for the spinning wing surfaces. Polar data was generated using **XFOIL** via XFLR5, covering multiple candidate airfoils across the operational Reynolds number range of the drone's wing tips.

---

## Airfoil Polars (`airfoil_polars/`)

XFOIL-format polar files (`.txt`) for each airfoil at each Reynolds number tested. File naming convention:

```
<AIRFOIL_NAME>_T1_Re<Re_value>_M0.00_N9.0.txt
```

**Airfoils evaluated:**

| Airfoil | Notes |
|---------|-------|
| AG14 | Low-Re glider section |
| NACA 4412 | Classic cambered section |
| RG 14 | Riegels-modified low-Re |
| RG-15 8.9% | Low-Re, thin, high Cl/Cd |
| S1223 | High-lift, low-Re |
| SD7032 | Smooth-stall low-Re section |
| SD7037 | Low-Re sailplane section |
| SD7062 14% | Thicker low-Re section |
| franky / franky 2 | Custom-designed sections |

**Reynolds numbers tested:** 0.001 – 0.207 × 10⁶ (matching wing-tip speeds at hover RPM)

The `.xfl` file (`jan1.xfl`) is an XFLR5 project file containing the full session.

---

## Analysis Scripts (`analysis/`)

| Script | Description |
|--------|-------------|
| `analyze_polars.py` | Load and parse all polar .txt files, compute Cl/Cd across α sweep |
| `generate_corrected_polar.py` | Apply corrections for low-Re effects |
| `plot_polar_comparison.py` | Side-by-side polar plots across airfoils |
| `plot_optimal_clcd.py` | Plot peak Cl/Cd for each airfoil vs Re |
| `plot_optimal_operating_points.py` | Overlay optimal operating points on polars |
| `plot_weighted_performance.py` | Weighted performance metric combining Cl, Cl/Cd, and stall margin |
| `plot_sim_forces.py` | Estimated aerodynamic forces at hover conditions |
| `verify_polar_data.py` | Data integrity checks on polar files |

**To run analysis:**
```bash
cd spinning-wing-drone
pip install numpy matplotlib pandas
python aerodynamics/analysis/analyze_polars.py
```

Scripts expect polar files to be in `aerodynamics/airfoil_polars/`. Update the path constant at the top of each script if running from a different working directory.

---

## Results (`results/`)

Pre-generated plots from the analysis pipeline:

| File | Description |
|------|-------------|
| `polars_Re*.png` | Polar plots at each Reynolds number |
| `airfoil_comparison_compact.png/.pdf` | Summary comparison across all airfoils |
| `airfoil_2x2_grid.png/.pdf` | 2×2 grid of the top 4 candidate airfoils |
| `cl_vs_cd_comparison.png` | Cl vs Cd overlay |
| `clcd_vs_alpha_comparison.png` | Cl/Cd ratio vs angle of attack |
| `cpmin_vs_alpha_comparison.png` | Minimum pressure coefficient (stall indicator) |
| `optimal_cl_comparison.png` | Peak Cl per airfoil |
| `optimal_clcd_comparison.png` | Peak Cl/Cd per airfoil |
| `optimal_operating_points_comparison.png` | Combined operating point map |
| `weighted_performance_comparison.png` | Weighted score across airfoils |
| `local_velocity_optima_*.json` | Optimal operating point data (machine-readable) |

> These are output files — re-running the analysis scripts will regenerate them.
