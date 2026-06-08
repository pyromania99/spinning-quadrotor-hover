import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Controller gains
kp_pos = 0.10
kd_pos = 1.0
g = 9.81

# Initial values
initial_target_x = -1.0
initial_target_y = -1.0
initial_yaw = 90.0  # degrees

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9, top=0.95)

# Text display
text_str = ""
text_display = ax.text(0.5, 0.5, text_str, transform=ax.transAxes, 
                       fontsize=12, verticalalignment='center', 
                       horizontalalignment='center', family='monospace')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# Create sliders
ax_target_x = plt.axes([0.15, 0.25, 0.7, 0.03])
ax_target_y = plt.axes([0.15, 0.20, 0.7, 0.03])
ax_yaw = plt.axes([0.15, 0.15, 0.7, 0.03])
ax_pos_x = plt.axes([0.15, 0.10, 0.7, 0.03])
ax_pos_y = plt.axes([0.15, 0.05, 0.7, 0.03])

slider_target_x = Slider(ax_target_x, 'Target X', -2.0, 2.0, valinit=initial_target_x, valstep=0.1)
slider_target_y = Slider(ax_target_y, 'Target Y', -2.0, 2.0, valinit=initial_target_y, valstep=0.1)
slider_yaw = Slider(ax_yaw, 'Yaw (deg)', 0.0, 360.0, valinit=initial_yaw, valstep=5.0)
slider_pos_x = Slider(ax_pos_x, 'Current X', -2.0, 2.0, valinit=0.0, valstep=0.1)
slider_pos_y = Slider(ax_pos_y, 'Current Y', -2.0, 2.0, valinit=0.0, valstep=0.1)

def update(val):
    # Get slider values
    target_x = slider_target_x.val
    target_y = slider_target_y.val
    yaw_deg = slider_yaw.val
    pos_x = slider_pos_x.val
    pos_y = slider_pos_y.val
    
    # Convert yaw to radians
    yaw = np.deg2rad(yaw_deg)
    
    # Calculate position errors (world frame)
    pos_error_x = target_x - pos_x
    pos_error_y = target_y - pos_y
    
    # Desired accelerations in world frame (simplified - no derivative term)
    acc_world_x_desired = kp_pos * pos_error_x
    acc_world_y_desired = kp_pos * pos_error_y
    
    # Transform to body frame and calculate desired roll/pitch
    desired_roll = (np.cos(yaw)*acc_world_x_desired + np.sin(yaw)*acc_world_y_desired) / g
    desired_pitch = (-np.sin(yaw)*acc_world_x_desired + np.cos(yaw)*acc_world_y_desired) / g
    
    # Calculate body frame accelerations for verification
    acc_body_x = np.cos(yaw)*acc_world_x_desired + np.sin(yaw)*acc_world_y_desired
    acc_body_y = -np.sin(yaw)*acc_world_x_desired + np.cos(yaw)*acc_world_y_desired
    
    # Body frame axes in world coordinates
    body_x_world = np.array([np.cos(yaw), np.sin(yaw)])
    body_y_world = np.array([-np.sin(yaw), np.cos(yaw)])
    
    # Update text display
    text_str = f"""
╔══════════════════════════════════════════════════════════════╗
║                  COORDINATE TRANSFORMATION                    ║
╚══════════════════════════════════════════════════════════════╝

CURRENT STATE:
  Position (world):     [{pos_x:6.2f}, {pos_y:6.2f}]
  Target (world):       [{target_x:6.2f}, {target_y:6.2f}]
  Yaw angle:            {yaw_deg:6.1f}°  ({yaw:6.3f} rad)

POSITION ERRORS (World Frame):
  Error X:              {pos_error_x:7.3f} m
  Error Y:              {pos_error_y:7.3f} m

DESIRED ACCELERATIONS (World Frame):
  acc_world_x:          {acc_world_x_desired:7.3f} m/s²
  acc_world_y:          {acc_world_y_desired:7.3f} m/s²

BODY FRAME ORIENTATION (at yaw={yaw_deg:.0f}°):
  Body +X (right)   →   World [{body_x_world[0]:6.3f}, {body_x_world[1]:6.3f}]
  Body +Y (forward) →   World [{body_y_world[0]:6.3f}, {body_y_world[1]:6.3f}]

DESIRED ACCELERATIONS (Body Frame):
  acc_body_x:           {acc_body_x:7.3f} m/s²  (roll control)
  acc_body_y:           {acc_body_y:7.3f} m/s²  (pitch control)

╔══════════════════════════════════════════════════════════════╗
║                   DESIRED ATTITUDE ANGLES                     ║
╚══════════════════════════════════════════════════════════════╝

  desired_roll:         {desired_roll:7.4f} rad  =  {np.rad2deg(desired_roll):7.2f}°
  desired_pitch:        {desired_pitch:7.4f} rad  =  {np.rad2deg(desired_pitch):7.2f}°

INTERPRETATION:
  {"Positive" if desired_roll > 0 else "Negative"} roll  → Tilt {"LEFT" if desired_roll > 0 else "RIGHT"}
  {"Positive" if desired_pitch > 0 else "Negative"} pitch → Nose {"DOWN" if desired_pitch > 0 else "UP"}
"""
    
    text_display.set_text(text_str)
    fig.canvas.draw_idle()

# Connect sliders to update function
slider_target_x.on_changed(update)
slider_target_y.on_changed(update)
slider_yaw.on_changed(update)
slider_pos_x.on_changed(update)
slider_pos_y.on_changed(update)

# Initial update
update(None)

plt.show()
