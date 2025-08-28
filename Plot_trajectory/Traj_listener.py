import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import socket
import struct
import queue
import threading




def udp_listener(port, udp_queue):
    """
    Listens for UDP packets and puts new rows into the udp_queue.

    Parameters
    ----------
    port : int
        The UDP port to listen on.

    Returns
    -------
    None
    """
    for row in listen_trajectory(port):
        udp_queue.put(row)  



def listen_trajectory(port):
    """
    Listens for UDP packets containing drone trajectory data and yields rows of data as pandas Series.

    Parameters
    ----------
    port : int
        The UDP port to listen on for trajectory data.

    Yields
    ------
    pandas.Series
        A row of trajectory data.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
        udp_socket.bind(('0.0.0.0', port))
        print(f"Listening for UDP packets on port {port}...")

        while True:
            try:
                udp_socket.settimeout(1.0)
                data, _ = udp_socket.recvfrom(1024)
            except socket.timeout:
                continue

            bytes_telemetry = bytearray(data)
            if len(bytes_telemetry) < 97:
                continue

            try:
                row = {
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

                row_df = pd.DataFrame([row])
                yield row_df.iloc[0]

            except Exception as e:
                print(f"Parsing error: {e}")
