import csv
from pymavlink import mavutil
import os
from datetime import datetime
import time

# Connect to the MAVLink device
master = mavutil.mavlink_connection('COM4', baud=57600)

# Wait for the first heartbeat to ensure connection is establishedN
print("Waiting for heartbeat...")
master.wait_heartbeat()
print("Heartbeat received. System is alive.")

# Open the CSV file for writing the output
# Create a directory if it doesn't exist
output_dir = 'test'
os.makedirs(output_dir, exist_ok=True)

# Generate a unique filename based on current time
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'mavlink_data_{timestamp_str}.csv'
filepath = os.path.join(output_dir, filename)

with open(filepath, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header to the CSV file
    writer.writerow([
        'Timestamp', 'Message Type', 
        'Voltage (V)', 'Current (A)', 'Remaining (%)',  # Battery Status
        'Servo1 Output', 'Servo2 Output', 'Servo8 Output', 'Servo4 Output',  # Servo Outputs
        'Altitude (m)',  # Altitude
        'Roll', 'Pitch', 'Yaw', 'Yaw Rate (rad/s)',  # Attitude
        'Acceleration Z (m/s^2)',  # Highres IMU
        'Local X (m)', 'Local Y (m)', 'Local Z (m)',  # Local Position NED
        'Vibration X', 'Vibration Y', 'Vibration Z',  # Vibration
        'ESC Info'  # ESC Info
        ])

    # Function to request a specific MAVLink message at a given interval
    def request_message_interval(message_id, frequency_hz):
        interval_us = int(1e6 / frequency_hz)  # Convert frequency to interval in microseconds
        master.mav.command_long_send(
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
            0,  # Confirmation
            message_id,  # The MAVLink message ID
            interval_us,  # Interval in microseconds
            0, 0, 0, 0, 0  # Unused parameters
        )

    # Request messages at desired frequencies (in Hz)
    request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_BATTERY_STATUS, 20)
    request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_SERVO_OUTPUT_RAW, 200)
    request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_ALTITUDE, 20)
    request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE, 100)
    # ESC_INFO cannot be requested - it's sent automatically if available

    print(f"\n=== Writing MAVLink data to: {os.path.abspath(filepath)} ===\n")
    print("Listening for MAVLink messages. Press Ctrl+C to stop.")

    try:
        while True:
            # Receive the next MAVLink message
            msg = master.recv_match(blocking=True)
            if msg:
                msg_type = msg.get_type()
                timestamp = msg._timestamp
                row = [timestamp, msg_type] + [''] * 20  # Initialize row with empty values

                if msg_type == 'BATTERY_STATUS':
                    voltage = msg.voltages[0] / 1000.0  # Convert mV to V
                    current = msg.current_battery / 100.0  # Convert cA to A
                    remaining = msg.battery_remaining  # Percentage
                    print(f"Battery - Voltage: {voltage} V, Current: {current} A, Remaining: {remaining}%")
                    row[2:5] = [voltage, current, remaining]
                elif msg_type == 'SERVO_OUTPUT_RAW':
                    servo_outputs = (msg.servo1_raw, msg.servo2_raw, msg.servo8_raw, msg.servo4_raw)
                    print(f"Servo Outputs: {servo_outputs}")
                    row[5:9] = servo_outputs
                elif msg_type == 'ALTITUDE':
                    print(f"Altitude - Monotonic: {msg.altitude_monotonic} m, AMSL: {msg.altitude_amsl} m, Local: {msg.altitude_local} m")
                    row[9] = msg.altitude_amsl
                elif msg_type == 'ATTITUDE':
                    roll = msg.roll
                    pitch = msg.pitch
                    yaw = msg.yaw
                    yaw_rate = msg.yawspeed
                    print(f"Attitude - Roll: {roll:.3f}, Pitch: {pitch:.3f}, Yaw: {yaw:.3f}, Yaw Rate: {yaw_rate:.3f} rad/s")
                    row[10:14] = [roll, pitch, yaw, yaw_rate]
                elif msg_type == 'HIGHRES_IMU':
                    az = msg.zacc
                    print(f"Accel Z: {az:.3f} m/s^2")
                    row[14] = az
                elif msg_type == 'LOCAL_POSITION_NED':
                    x, y, z = msg.x, msg.y, msg.z
                    print(f"Local Position - X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}")
                    row[15:18] = [x, y, z]
                elif msg_type == 'VIBRATION':
                    vib_x, vib_y, vib_z = msg.vibration_x, msg.vibration_y, msg.vibration_z
                    print(f"Vibration - X: {vib_x:.3f}, Y: {vib_y:.3f}, Z: {vib_z:.3f}")
                    row[18:21] = [vib_x, vib_y, vib_z]
                elif msg_type == 'ESC_INFO':
                    esc_info = []
                    for i in range(msg.esc_count):
                        esc_info.append(f"ESC{i}:T={msg.esc[i].temperature}C,E={msg.esc[i].error_count}")
                    info_str = "; ".join(esc_info)
                    print(f"ESC Info: {info_str}")
                    row[21] = info_str
                else:
                    continue  # Skip unwanted message types

                writer.writerow(row)
                file.flush()  # Ensure data is written immediately

    except KeyboardInterrupt:
        print("Stopped by user.")
