import socket
import struct
import threading
import keyboard
import os
import argparse
import csv

recording = False
terminate = False
lock = threading.Lock()
filename = None

# Colonnes du CSV
columns = [
    'drone_id', 'timestamp', 'position_x', 'position_y', 'position_z',
    'quaternion_x', 'quaternion_y', 'quaternion_z', 'quaternion_w',
    'velocity_x', 'velocity_y', 'velocity_z',
    'gyro_pitch', 'gyro_roll', 'gyro_yaw',
    'input_throttle', 'input_yaw', 'input_pitch', 'input_roll',
    'battery_percentage', 'battery_voltage', 'num_motors',
    'left_front_rpm', 'right_front_rpm', 'left_back_rpm', 'right_back_rpm'
]

def toggle_recording_keypress(output_path):
    global recording, terminate, filename
    print("Press 'a' to start / end recording.")
    while True:
        keyboard.wait('a')
        with lock:
            recording = not recording
            if recording:
                filename = output_path
                print(f"Recording started -> {filename}")
            else:
                print("Recording ended")
                terminate = True
                break  

def listen_trajectory(port, drone_id):
    global recording, terminate, filename
    csv_file = None
    csv_writer = None

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
        udp_socket.bind(('0.0.0.0', port))
        print(f"Listening for UDP packets on port {port}...")

        while not terminate:
            try:
                udp_socket.settimeout(1.0)
                data, _ = udp_socket.recvfrom(1024)
            except socket.timeout:
                continue

            if not recording:
                # Si l'enregistrement est arrêté, fermer le fichier si ouvert
                if csv_file:
                    csv_file.close()
                    csv_file = None
                    csv_writer = None
                continue

            # Si on commence à enregistrer et que le fichier n'est pas ouvert, l'ouvrir
            if recording and csv_file is None:
                file_exists = os.path.exists(filename)
                csv_file = open(filename, mode='a', newline='')
                csv_writer = csv.DictWriter(csv_file, fieldnames=columns)
                if not file_exists:
                    csv_writer.writeheader()

            bytes_telemetry = bytearray(data)
            if len(bytes_telemetry) < 97:
                continue

            try:
                row = {
                    'drone_id': drone_id,
                    'timestamp': struct.unpack('f', bytes_telemetry[0:4])[0],
                    'position_x': struct.unpack('f', bytes_telemetry[4:8])[0],
                    'position_y': struct.unpack('f', bytes_telemetry[8:12])[0],
                    'position_z': struct.unpack('f', bytes_telemetry[12:16])[0],
                    'quaternion_x': struct.unpack('f', bytes_telemetry[16:20])[0],
                    'quaternion_y': struct.unpack('f', bytes_telemetry[20:24])[0],
                    'quaternion_z': struct.unpack('f', bytes_telemetry[24:28])[0],
                    'quaternion_w': struct.unpack('f', bytes_telemetry[28:32])[0],
                    'velocity_x': struct.unpack('f', bytes_telemetry[32:36])[0],
                    'velocity_y': struct.unpack('f', bytes_telemetry[36:40])[0],
                    'velocity_z': struct.unpack('f', bytes_telemetry[40:44])[0],
                    'gyro_pitch': struct.unpack('f', bytes_telemetry[44:48])[0],
                    'gyro_roll': struct.unpack('f', bytes_telemetry[48:52])[0],
                    'gyro_yaw': struct.unpack('f', bytes_telemetry[52:56])[0],
                    'input_throttle': struct.unpack('f', bytes_telemetry[56:60])[0],
                    'input_yaw': struct.unpack('f', bytes_telemetry[60:64])[0],
                    'input_pitch': struct.unpack('f', bytes_telemetry[64:68])[0],
                    'input_roll': struct.unpack('f', bytes_telemetry[68:72])[0],
                    'battery_percentage': struct.unpack('f', bytes_telemetry[72:76])[0],
                    'battery_voltage': struct.unpack('f', bytes_telemetry[76:80])[0],
                    'num_motors': struct.unpack('B', bytes_telemetry[80:81])[0],
                    'left_front_rpm': struct.unpack('f', bytes_telemetry[81:85])[0],
                    'right_front_rpm': struct.unpack('f', bytes_telemetry[85:89])[0],
                    'left_back_rpm': struct.unpack('f', bytes_telemetry[89:93])[0],
                    'right_back_rpm': struct.unpack('f', bytes_telemetry[93:97])[0]
                }

                if csv_writer:
                    csv_writer.writerow(row)
                    csv_file.flush()

            except Exception as e:
                print(f"Parsing error: {e}")

        # À la fin, fermer le fichier CSV s'il est ouvert
        if csv_file:
            csv_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run drone telemetry trajectory")
    parser.add_argument("--drone_id", default="1", help="Type of drone : 1 = DJI FPV Drone, 2 = Luma 5, 3 = ...")
    parser.add_argument("--output_file", default="Liftoff_trajectory.csv", help="Output file name for the trajectory data.")
    args = parser.parse_args()
    port = 9001

    thread_listener = threading.Thread(target=listen_trajectory, args=(port, args.drone_id))
    thread_toggle = threading.Thread(target=toggle_recording_keypress, args=(args.output_file,))

    thread_listener.start()
    thread_toggle.start()

    thread_toggle.join()
    thread_listener.join()

    print("Program finished.")
