import csv
import os
import time
from datetime import datetime
from pymavlink import mavutil

# Connect to the MAVLink device
master = mavutil.mavlink_connection('COM4', baud=57600)

# Wait for the first heartbeat to ensure connection is established
print("Waiting for heartbeat...")
master.wait_heartbeat()
print("Heartbeat received. System is alive.")

# ===============================
# Output file
# ===============================
output_dir = 'test'
os.makedirs(output_dir, exist_ok=True)

# Generate a unique filename based on current time
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
filename = f'servo_log_{timestamp_str}.csv'
filepath = os.path.join(output_dir, filename)

file = open(filepath, 'w', newline='')
writer = csv.writer(file)

writer.writerow([
    'timestamp',
    'servo1',
    'servo2',
    'servo3',
    'servo4',
    'servo5',
    'servo6',
    'servo7',
    'servo8'
])

print(f"Logging to: {os.path.abspath(filepath)}")

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
    # Wait for ACK
    ack = master.recv_match(type='COMMAND_ACK', blocking=True, timeout=3)
    if ack:
        if ack.result == 0:
            print(f"  -> MSG {message_id} requested at {frequency_hz} Hz: ACCEPTED")
        else:
            print(f"  -> MSG {message_id} requested at {frequency_hz} Hz: REJECTED (result={ack.result})")
    else:
        print(f"  -> MSG {message_id} requested at {frequency_hz} Hz: NO ACK (timeout)")

# Request servo output
print("Requesting SERVO_OUTPUT_RAW...")
request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_SERVO_OUTPUT_RAW, 200)

# Also try requesting via REQUEST_DATA_STREAM as fallback
print("Requesting RC_CHANNELS data stream as fallback...")
master.mav.request_data_stream_send(
    master.target_system,
    master.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS,
    50,   # Hz
    1     # start
)
time.sleep(0.5)

# Rate monitor
count = 0
t0 = time.time()

# First check what messages are actually arriving
print("\nChecking what messages arrive (5 seconds)...")
check_start = time.time()
msg_types = {}
while time.time() - check_start < 5:
    msg = master.recv_match(blocking=True, timeout=1)
    if msg:
        mt = msg.get_type()
        msg_types[mt] = msg_types.get(mt, 0) + 1
if msg_types:
    print("Messages received:")
    for mt, cnt in sorted(msg_types.items(), key=lambda x: -x[1]):
        print(f"  {mt}: {cnt}")
else:
    print("  NO messages received at all!")

print("\nListening for SERVO_OUTPUT_RAW messages. Press Ctrl+C to stop.\n")

# Main loop
try:
    while True:
        msg = master.recv_match(type="SERVO_OUTPUT_RAW", blocking=True)
        if msg is None:
            continue

        outputs = [
            msg.servo1_raw,
            msg.servo2_raw,
            msg.servo3_raw,
            msg.servo4_raw,
            msg.servo5_raw,
            msg.servo6_raw,
            msg.servo7_raw,
            msg.servo8_raw
        ]

        print(f"Servo Outputs: {outputs}")

        writer.writerow([msg._timestamp, *outputs])
        file.flush()

        count += 1
        if time.time() - t0 >= 1.0:
            print(f"Servo rate: {count} Hz")
            count = 0
            t0 = time.time()

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    file.close()
