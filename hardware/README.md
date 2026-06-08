# Hardware

Physical design files for the spinning-wing drone prototype.

---

## CAD (`cad/`)

### Wings (`cad/wings/`)

Wing surface designs iterated over the project. All designed in Autodesk Fusion 360.

| File | Description |
|------|-------------|
| `feb_wing.stl` | February iteration wing (final prototype) |
| `feb_wing.3mf` | 3MF version for direct slicer import |
| `feb_wing_26.step` | STEP file for CAD interchange |
| `jan_wing.stl` | January iteration wing |

The `wing_design_archive/` subdirectory contains earlier design iterations.

**Key design parameters (final wing):**
- Airfoil: RG-15 (selected from aerodynamics analysis)
- Mounting: Servo-driven tilt for thrust vectoring
- Material: PLA/PETG (FDM printed)

### Drone Assembly (`cad/drone_assembly/`)

Full drone frame and arm assembly files.

### Propellers (`cad/propellers/`)

| File | Description |
|------|-------------|
| `PROPELLER - 8x4.5 CCW.STEP` | 8×4.5" CCW propeller (main rotors) |
| `Gemfan 3in 2blade.stl` | 3" 2-blade propeller model |
| `Emax ECO Micro 1404 2~4S 6000KV v3.step` | Motor STEP file for assembly alignment |
| `blade_for+_print.f3d` | Custom blade Fusion 360 source |
| `Trial1.f3z` | Earlier propeller trial |
| `8-x-4-5-propellers-1.snapshot.5.zip` | Snapshot of 8×4.5 propeller project |

---

## Firmware (`firmware/`)

PX4 firmware binaries for the **Holybro KakuteH7 Mini** flight controller.

| File | Description |
|------|-------------|
| `holybro_kakuteh7mini_default.px4` | Main PX4 flight firmware |
| `holybro_kakuteh7mini_bootloader.hex` | Bootloader |
| `holybro_kakuteh7mini_bootloader_ID_1058.hex` | Bootloader (board ID 1058 variant) |

**To flash:** Use QGroundControl (Firmware tab → Advanced → Custom firmware) or the PX4 `px_uploader.py` script.

> These are pre-built binaries. Source code is maintained in the upstream [PX4-Autopilot](https://github.com/PX4/PX4-Autopilot) repository.

---

## 3D Printing Notes

- Wings: printed flat, 3 perimeters, 20% gyroid infill
- Material: PETG preferred for flex tolerance at wing root
- Layer height: 0.2 mm
- Filament specs are in `05_hardware/3d_printing/filament/` (parent archive)
