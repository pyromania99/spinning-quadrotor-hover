import csv
from pymavlink import mavutil
import threading
import time
import math
import pandas as pd
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Connect to MAVLink
master = mavutil.mavlink_connection('COM4', baud=57600)

print("Waiting for heartbeat...")
master.wait_heartbeat()
print("Heartbeat received. System is alive.")

# Request messages
def request_message_interval(message_id, frequency_hz):
    interval_us = int(1e6 / frequency_hz)
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
        0,
        message_id,
        interval_us,
        0, 0, 0, 0, 0
    )

# Request both attitude and RC channels
request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 50)
request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_RC_CHANNELS, 50)

# Store latest values
latest_yaw = 0.0
latest_pitch_setpoint = 0.0
latest_roll_setpoint = 0.0
latest_body_pitch = 0.0
latest_body_roll = 0.0

# Data storage for Excel
data_rows = []
data_lock = threading.Lock()

# Generate filename
def get_filename():
    folder_path = r"c:\Users\anike\Desktop\RESEARCH\TELEM\yaw_dat"
    
    # Create folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    # Find existing files to get next number
    existing_files = [f for f in os.listdir(folder_path) if f.startswith('mavlink_data_') and f.endswith('.xlsx')]
    if existing_files:
        numbers = []
        for f in existing_files:
            try:
                num = int(f.split('_')[-1].split('.')[0])
                numbers.append(num)
            except:
                continue
        next_num = max(numbers) + 1 if numbers else 1
    else:
        next_num = 1
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mavlink_data_{timestamp}_{next_num:03d}.xlsx"
    return os.path.join(folder_path, filename)

def mavlink_listener():
    global latest_yaw, latest_pitch_setpoint, latest_roll_setpoint
    global latest_body_pitch, latest_body_roll, data_rows
    
    try:
        while True:
            msg = master.recv_match(blocking=True)
            if msg:
                msg_type = msg.get_type()
                timestamp = time.time()
                
                if msg_type == 'ATTITUDE':
                    with data_lock:
                        latest_yaw = math.degrees(msg.yaw)
                    
                elif msg_type == 'RC_CHANNELS':
                    pitch_raw = msg.chan2_raw
                    roll_raw = msg.chan1_raw
                    
                    # Convert from RC range to setpoint range
                    ned_pitch = (pitch_raw - 1500) / 500.0
                    ned_roll = (roll_raw - 1500) / 500.0
                    
                    # Transform to body frame
                    yaw_rad = math.radians(latest_yaw)
                    body_pitch = ned_pitch * math.cos(yaw_rad) - ned_roll * math.sin(yaw_rad)
                    body_roll = ned_pitch * math.sin(yaw_rad) + ned_roll * math.cos(yaw_rad)
                    
                    with data_lock:
                        latest_pitch_setpoint = ned_pitch
                        latest_roll_setpoint = ned_roll
                        latest_body_pitch = body_pitch
                        latest_body_roll = body_roll
                    
                    # Store data row
                    row = {
                        'Timestamp': timestamp,
                        'DateTime': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                        'Yaw_degrees': latest_yaw,
                        'NED_Pitch': ned_pitch,
                        'NED_Roll': ned_roll,
                        'Body_Pitch': body_pitch,
                        'Body_Roll': body_roll,
                        'Pitch_Raw': pitch_raw,
                        'Roll_Raw': roll_raw,
                    }
                    
                    data_rows.append(row)
                    
    except KeyboardInterrupt:
        print("Stopped by user.")

# Start MAVLink listener thread
listener_thread = threading.Thread(target=mavlink_listener, daemon=True)
listener_thread.start()

# Set up the visualization
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.set_xlabel('East (Roll)')
ax.set_ylabel('North (Pitch)')
ax.set_title('Vehicle Coordinate Frame Transformation\nRed=NED Input, Blue=Body Vector, Black=Vehicle Heading')

# Initialize plot elements - store references
vehicle_heading = None
ned_vector = None
body_vector = None

# Text displays
yaw_text = ax.text(-1.8, 1.8, '', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
ned_text = ax.text(-1.8, 1.5, '', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7))
body_text = ax.text(-1.8, 1.2, '', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.7))

def animate(frame):
    global vehicle_heading, ned_vector, body_vector
    
    with data_lock:
        yaw = latest_yaw
        ned_pitch = latest_pitch_setpoint
        ned_roll = latest_roll_setpoint
        body_pitch = latest_body_pitch
        body_roll = latest_body_roll
    
    # Remove old arrows properly
    if vehicle_heading is not None:
        vehicle_heading.remove()
    if ned_vector is not None:
        ned_vector.remove()
    if body_vector is not None:
        body_vector.remove()
    
    # Vehicle heading (yaw direction)
    yaw_rad = math.radians(yaw)
    heading_x = 0.8 * math.sin(yaw_rad)  # East component
    heading_y = 0.8 * math.cos(yaw_rad)  # North component
    vehicle_heading = ax.arrow(0, 0, heading_x, heading_y, 
                              head_width=0.05, head_length=0.05, 
                              fc='black', ec='black', linewidth=3)
    
    # NED input vector (red)
    ned_scale = 0.5
    ned_x = ned_roll * ned_scale  # East (roll right is positive)
    ned_y = ned_pitch * ned_scale  # North (pitch forward is positive)
    if abs(ned_x) > 0.01 or abs(ned_y) > 0.01:
        ned_vector = ax.arrow(0, 0, ned_x, ned_y, 
                             head_width=0.03, head_length=0.03, 
                             fc='red', ec='red', linewidth=2, alpha=0.8)
    else:
        ned_vector = None
    
    # Body vector (blue) - transformed
    body_scale = 0.5
    body_x = body_roll * body_scale
    body_y = body_pitch * body_scale
    if abs(body_x) > 0.01 or abs(body_y) > 0.01:
        body_vector = ax.arrow(0, 0, body_x, body_y, 
                              head_width=0.03, head_length=0.03, 
                              fc='blue', ec='blue', linewidth=2, alpha=0.8)
    else:
        body_vector = None
    
    # Update text
    yaw_text.set_text(f'Yaw: {yaw:.1f}°')
    ned_text.set_text(f'NED: P={ned_pitch:.2f}, R={ned_roll:.2f}')
    body_text.set_text(f'Body: P={body_pitch:.2f}, R={body_roll:.2f}')
    
    return []

# Start animation
ani = animation.FuncAnimation(fig, animate, interval=50, blit=False, cache_frame_data=False)

plt.tight_layout()

try:
    plt.show()
except KeyboardInterrupt:
    pass
finally:
    print("Saving data...")
    if data_rows:
        filename = get_filename()
        df = pd.DataFrame(data_rows)
        df.to_excel(filename, index=False)
        print(f"Data saved to: {filename}")
        print(f"Total rows: {len(data_rows)}")
