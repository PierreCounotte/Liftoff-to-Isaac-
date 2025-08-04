import socket
import struct
import pandas as pd
import threading
import keyboard
import os
import argparse

recording = False
terminate = False
lock = threading.Lock()
filename = None
trajectory_df = pd.DataFrame(columns=
                             
                             [
    'drone_id', 'timestamp', 'position_x', 'position_y', 'position_z',
    'quaternion_x', 'quaternion_y', 'quaternion_z', 'quaternion_w',
    'velocity_x', 'velocity_y', 'velocity_z',
    'gyro_pitch', 'gyro_roll', 'gyro_yaw',
    'input_throttle', 'input_yaw', 'input_pitch', 'input_roll',
    'battery_percentage', 'battery_voltage', 'num_motors',
    'left_front_rpm', 'right_front_rpm', 'left_back_rpm', 'right_back_rpm'
])

def toggle_recording_keypress():
    global recording, terminate, filename
    print("Press 'enter' to start / end recording.")
    while True:
        keyboard.wait('enter')
        with lock:
            recording = not recording
            if recording:
                filename = f"Big_trajectory_corrected.csv"
                print(f"Recording started -> {filename}")
            else:
                print("Recording ended")
                terminate = True
                break  
def listen_trajectory(port, drone_id):
    global recording, terminate, trajectory_df
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
                continue

            bytes_telemetry = bytearray(data)
            if len(bytes_telemetry) < 97:
                continue

            try:
                row = {
                    'drone_id' :           drone_id,
                    'timestamp':           struct.unpack('f', bytes_telemetry[0:4])[0], # sec
                    'position_x':          struct.unpack('f', bytes_telemetry[4:8])[0], # m (points to the right)
                    'position_y':          struct.unpack('f', bytes_telemetry[8:12])[0], # m (points up)
                    'position_z':          struct.unpack('f', bytes_telemetry[12:16])[0], # m (points forward)
                    'quaternion_x':        struct.unpack('f', bytes_telemetry[16:20])[0], 
                    'quaternion_y':        struct.unpack('f', bytes_telemetry[20:24])[0],
                    'quaternion_z':        struct.unpack('f', bytes_telemetry[24:28])[0],
                    'quaternion_w':        struct.unpack('f', bytes_telemetry[28:32])[0],
                    'velocity_x':          struct.unpack('f', bytes_telemetry[32:36])[0], # m/s
                    'velocity_y':          struct.unpack('f', bytes_telemetry[36:40])[0], # m/s
                    'velocity_z':          struct.unpack('f', bytes_telemetry[40:44])[0], # m/s
                    'gyro_pitch':          struct.unpack('f', bytes_telemetry[44:48])[0], # deg/s
                    'gyro_roll':           struct.unpack('f', bytes_telemetry[48:52])[0], # deg/s
                    'gyro_yaw':            struct.unpack('f', bytes_telemetry[52:56])[0], # deg/s
                    'input_throttle':      struct.unpack('f', bytes_telemetry[56:60])[0],
                    'input_yaw':           struct.unpack('f', bytes_telemetry[60:64])[0],
                    'input_pitch':         struct.unpack('f', bytes_telemetry[64:68])[0],
                    'input_roll':          struct.unpack('f', bytes_telemetry[68:72])[0],
                    'battery_percentage':  struct.unpack('f', bytes_telemetry[72:76])[0],
                    'battery_voltage':     struct.unpack('f', bytes_telemetry[76:80])[0],
                    'num_motors':          struct.unpack('B', bytes_telemetry[80:81])[0],
                    'left_front_rpm':      struct.unpack('f', bytes_telemetry[81:85])[0],
                    'right_front_rpm':     struct.unpack('f', bytes_telemetry[85:89])[0],
                    'left_back_rpm':       struct.unpack('f', bytes_telemetry[89:93])[0],
                    'right_back_rpm':      struct.unpack('f', bytes_telemetry[93:97])[0]
                }
                with lock:
                    trajectory_df.loc[len(trajectory_df)] = row
            except Exception as e:
                print(f"Parsing error: {e}")
                


def split_csv_with_chunks(csv_path, chunk_size=1000, output_path=None):
    
    if not os.path.exists(csv_path):
        print(f"{csv_path} does not exist.")
        return

    df = pd.read_csv(csv_path)
    total_rows = len(df)
    chunks = []

    for i in range(0, total_rows, chunk_size):
        chunk_id = (i // chunk_size) + 1
        chunk = df.iloc[i:i + chunk_size].copy()
        chunk.insert(0, 'chunk_id', chunk_id)  
        chunks.append(chunk)

    combined_df = pd.concat(chunks, ignore_index=True)

    if output_path is None:
        output_path = "Chunked_trajectory_corrected.csv"

    combined_df.to_csv(output_path, index=False)
    print(f"Output file : {output_path}")


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run drone telemetry trajectory")
    parser.add_argument("--drone_id", default="1", help="Type of drone : 1 = DJI FPV Drone, 2 = Luma 5, 3 = ...")
    parser.add_argument("--chunks", default="0", choices=["0","1"], help="1 for a chunked csv")
    args = parser.parse_args()
    port = 9001

    thread_listener = threading.Thread(target=listen_trajectory, args=(port,args.drone_id))
    thread_toggle = threading.Thread(target=toggle_recording_keypress)

    thread_listener.start()
    thread_toggle.start()

    thread_toggle.join()
    thread_listener.join()
    
    if filename and not trajectory_df.empty:
        total_new_samples = len(trajectory_df)
        samples_to_write = (total_new_samples // 1000) * 1000

        if samples_to_write == 0:
            print("Not enough samples to complete a trajectory of 10 sec.")
        else:
            filtered_df = trajectory_df.iloc[:samples_to_write]
            file_exists = os.path.exists(filename)

            filtered_df.to_csv(
                filename,
                mode='a',           
                header=not file_exists,  
                index=False
            )
            print(f"{samples_to_write} samples added to : {filename}")
    else:
        print("No recording has been made.")


    if(args.chunks == '1'):
        split_csv_with_chunks(f"{filename}")
    




