from pymavlink import mavutil

# Connect to the MAVLink device
master = mavutil.mavlink_connection('COM4', baud=57600)

# Wait for the first heartbeat to ensure connection is established
print("Waiting for heartbeat...")
master.wait_heartbeat()
print("Heartbeat received. System is alive.")

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
request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_BATTERY_STATUS, 10)
request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_SERVO_OUTPUT_RAW, 100)
request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_ALTITUDE, 5)

print("Listening for MAVLink messages. Press Ctrl+C to stop.")

try:
    while True:
        # Receive the next MAVLink message
        msg = master.recv_match(blocking=True)
        if msg:
            msg_type = msg.get_type()
            if msg_type == 'BATTERY_STATUS':
                voltage = msg.voltages[0] / 1000.0  # Convert mV to V
                current = msg.current_battery / 100.0  # Convert cA to A
                remaining = msg.battery_remaining  # Percentage
                print(f"Battery - Voltage: {voltage} V, Current: {current} A, Remaining: {remaining}%")
            elif msg_type == 'SERVO_OUTPUT_RAW':
                servo_outputs = msg.servo1_raw, msg.servo2_raw, msg.servo3_raw, msg.servo4_raw
                print(f"Servo Outputs: {servo_outputs}")
            elif msg_type == 'ESC_STATUS':
                for i in range(msg.esc_count):
                    print(f"ESC {i} - RPM: {msg.esc[i].rpm}, Voltage: {msg.esc[i].voltage} V, Current: {msg.esc[i].current} A")
            elif msg_type == 'ESC_INFO':
                for i in range(msg.esc_count):
                    print(f"ESC {i} - Error Count: {msg.esc[i].failure_flags}, Temperature: {msg.esc[i].temperature} C")
            elif msg_type == 'ALTITUDE':
                print(f"Altitude - Monotonic: {msg.altitude_monotonic} m, AMSL: {msg.altitude_amsl} m, Local: {msg.altitude_local} m")
            else:
                print(f"Received message type: {msg_type}")

except KeyboardInterrupt:
    print("Stopped by user.")
