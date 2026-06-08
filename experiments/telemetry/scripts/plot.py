import csv
from pymavlink import mavutil
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time

# Data storage for plotting
servo_data = {
    'servo1': [],
    'servo2': [],
    'servo3': [],
    'servo4': [],
    'timestamps': []
}

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
request_message_interval(mavutil.mavlink.MAVLINK_MSG_ID_HIGHRES_IMU, 100)  # Request accelerometer data

def mavlink_listener():
    try:
        while True:
            msg = master.recv_match(blocking=True)
            if msg:
                msg_type = msg.get_type()
                timestamp = msg._timestamp

                if msg_type == 'SERVO_OUTPUT_RAW':
                    servo_outputs = (
                        msg.servo1_raw, 
                        msg.servo2_raw, 
                        msg.servo8_raw, 
                        msg.servo4_raw
                    )
                    # Keep last 50 values for each servo
                    servo_data['servo1'].append(servo_outputs[0])
                    servo_data['servo2'].append(servo_outputs[1])
                    servo_data['servo3'].append(servo_outputs[2])
                    servo_data['servo4'].append(servo_outputs[3])
                    servo_data['timestamps'].append(timestamp)
                    for key in servo_data:
                        servo_data[key] = servo_data[key][-50:]
    except KeyboardInterrupt:
        print("Stopped by user.")

# Start MAVLink listener in a separate thread
listener_thread = threading.Thread(target=mavlink_listener, daemon=True)
listener_thread.start()

# Set up matplotlib for live plotting
plt.style.use('seaborn-v0_8')
fig, ax = plt.subplots()
lines = [
    ax.plot([], [], label='Servo 1')[0],
    ax.plot([], [], label='Servo 2')[0],
    ax.plot([], [], label='Servo 3')[0],
    ax.plot([], [], label='Servo 4')[0]
]
ax.set_xlabel('Time (s)')
ax.set_ylabel('Servo Output')
ax.set_title('Live Servo Outputs')
ax.legend()

def animate(frame):
    if len(servo_data['timestamps']) > 0:
        times = servo_data['timestamps']
        for i, key in enumerate(['servo1', 'servo2', 'servo3', 'servo4']):
            lines[i].set_data(times, servo_data[key])
        ax.set_xlim(times[0], times[-1])
        ymin = min([min(servo_data[k]) for k in ['servo1', 'servo2', 'servo3', 'servo4']]) - 100
        ymax = max([max(servo_data[k]) for k in ['servo1', 'servo2', 'servo3', 'servo4']]) + 100
        ax.set_ylim(ymin, ymax)
    return lines

ani = FuncAnimation(fig, animate, interval=5, cache_frame_data=False)
plt.show()