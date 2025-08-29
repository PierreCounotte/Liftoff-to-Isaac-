import pyvjoy
import time
import math

# -------------------------------
# Configuration
# -------------------------------
j = pyvjoy.VJoyDevice(1)
AXIS_MAX = 32767
UPDATE_RATE = 0.05  # 20 Hz

THROTTLE_START = -1.0  # Drone au sol
THROTTLE_TARGET = 0.3  # Altitude de croisière

ROLL_AMPLITUDE = 0.2
PITCH_AMPLITUDE = 0.2
YAW_AMPLITUDE = 0.2  # Tour complet

def normalize_axis(value):
    """Converts a value [-1,1] in vJoy [0,32767]"""
    value = max(min(value, 1.0), -1.0)
    return int((value + 1) * (AXIS_MAX / 2))

def send_axes(roll=0.0, pitch=0.0, throttle=0.0, yaw=0.0):
    j.set_axis(pyvjoy.HID_USAGE_X, normalize_axis(roll))
    j.set_axis(pyvjoy.HID_USAGE_Y, normalize_axis(pitch))
    j.set_axis(pyvjoy.HID_USAGE_Z, normalize_axis(throttle))
    j.set_axis(pyvjoy.HID_USAGE_RZ, normalize_axis(yaw))
    print(f"Roll={roll:.2f}, Pitch={pitch:.2f}, Throttle={throttle:.2f}, Yaw={yaw:.2f}")

# -------------------------------
# Phase 1 : Vertical takeoff
# -------------------------------
try:
    print("[INFO] Takeoff : throttle initialized at -1")
    send_axes(throttle=THROTTLE_START)
    time.sleep(3)

    print("[INFO] Vertical ascent...")
    throttle = THROTTLE_START
    while throttle < THROTTLE_TARGET:
        throttle += 0.02
        send_axes(roll=0.0, pitch=0.0, throttle=throttle, yaw=0.0)
        send_axes(roll=0.0, pitch=0.0, throttle=throttle, yaw=0.0)
        time.sleep(UPDATE_RATE)

    print("[INFO] Drone reach target altitude. Activation of commands Roll, Pitch, Yaw...")

    # -------------------------------
    # Phase 2 : Flight with commands
    # -------------------------------
    t = 0.0
    while True:
        roll = math.sin(t * 0.8) * ROLL_AMPLITUDE
        pitch = math.sin(t * 1.0) * PITCH_AMPLITUDE
        yaw = math.sin(t * 0.5) * YAW_AMPLITUDE
        send_axes(roll=roll, pitch=pitch, throttle=THROTTLE_TARGET, yaw=yaw)
        t += 0.05
        time.sleep(UPDATE_RATE)
        if t > 10:
            send_axes(roll=0, pitch=0, throttle=0, yaw=0)
            time.sleep(UPDATE_RATE)
            break

except KeyboardInterrupt:
    print("\n[INFO] User interruption detected. Stopping...")
finally:
    print("[INFO] Resetting axes to zero.")
