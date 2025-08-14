""" This script runs a drone trajectory in Isaac Lab using either a CSV file or UDP data stream."""


import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run drone trajectory in Isaac Lab.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--input_file", default="test.csv", help="CSV file containing the trajectory data")
parser.add_argument("--data_source", choices=["csv", "udp"], help="Source of the trajectory data")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import time
import socket
import struct
import threading
import keyboard
import queue

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg, Articulation, ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pxr import Usd

""" Global variables for threading """
recording = False
terminate = False
lock = threading.Lock()
udp_stack = []

""" Liftoff sampling frequency"""
sampling_frequency = 100 # Hz


def design_scene():
    """
    Design the scene by adding assets to it. Creates a ground plane, a distant light,
    and spawns a custom drone model.

    Parameters
    ----------
    None
    
    Returns
    -------
    scene_entities : dict
        A dictionary containing the drone object.
    """
    
    """ Ground plane and distant light"""
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),  
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    prim_utils.create_prim("/World/Objects", "Xform")

    """ Drone object"""
    drone_cfg = ArticulationCfg(
        prim_path="/World/Objects/Vapor_X5",
        spawn=sim_utils.UsdFileCfg(
            usd_path = f"C:/Users/Administrateur/Documents/DroneProject/Liftoff-to-Isaac-/Vapor_X5_along_x/my_drone/my_drone.usd",
        ),
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={},  
    )

    drone_object = Articulation(cfg=drone_cfg)

    scene_entities = {"drone": drone_object}
    return scene_entities



def LiftoffToIsaacCoordinates(df):
    """
    Converts the coordinates from Liftoff to Isaac conventions.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the trajectory data with at least the following columns:
        - position_x, position_y, position_z
        - quaternion_x, quaternion_y, quaternion_z, quaternion_w
        -velocity_x, velocity_y, velocity_z
    
    Returns
    -------
    None
    Modifies the Dataframe in place to create a new one.
    """
    x_unity = df['position_x'].copy()
    df.loc[:, 'position_x'] = df['position_z']
    df.loc[:, 'position_z'] = df['position_y']
    df.loc[:, 'position_y'] = -x_unity

    vx_unity = df['velocity_x'].copy()
    df.loc[:, 'velocity_x'] = df['velocity_z']
    df.loc[:, 'velocity_z'] = df['velocity_y']
    df.loc[:, 'velocity_y'] = -vx_unity


    """ Unity format quaternions """
    quat_unity = df[['quaternion_x', 'quaternion_y', 'quaternion_z', 'quaternion_w']].values

    """ Converts Unity format quaternions to Isaac format"""
    r_unity = R.from_quat(quat_unity)
    P = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, 1, 0]
    ])
    rot_mats = r_unity.as_matrix()
    rot_mats_converted = P @ rot_mats @ P.T
    angles = R.from_matrix(rot_mats_converted).as_euler('xyz', degrees=False)
    angles[:, [0, 2]] = angles[:, [2, 0]]
    angles[:, 0] *= -1  # Roll
    angles[:, 2] *= -1  # Yaw

    r_converted = R.from_euler('xyz', angles, degrees=False)
    r_z_180 = R.from_euler('x', -180, degrees=True)   
    r_final = r_z_180 * r_converted

    """ Isaac format quaternions """
    quat_converted = r_final.as_quat()
    df['quaternion_x'] = quat_converted[:, 0]
    df['quaternion_y'] = quat_converted[:, 1]
    df['quaternion_z'] = quat_converted[:, 2]
    df['quaternion_w'] = quat_converted[:, 3]




def run_simulator_from_csv(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], csv_path: str):
    """
    Run the simulation using a CSV file containing Liftoff trajectory data.

    Parameters
    ----------
    sim : sim_utils.SimulationContext
        The simulation context.
    entities : dict[str, Articulation]
        Dictionary containing the drone object.
    csv_path : str
        Path to the CSV file containing the trajectory data.
    
    Returns
    -------
    None  
    """
    drone_object = entities["drone"]
    sim_dt = sim.get_physics_dt()

    if not os.path.exists(csv_path):
        print(f"{csv_path} does not exist.")
        return
    
    """ Read CSV file """
    row_stream = live_csv_reader(csv_path)

    sim_time = 0.0
    count = 0
    while simulation_app.is_running():
        try:
            row = next(row_stream)
        except StopIteration:
            print("[INFO] Data flux end.")
            return

        """ Recover drone position and orientation from the DataFrame row"""
        x_pos = row['position_x']-1000 # Center drone in the map
        y_pos = row['position_y']+1070 # Center drone in the map
        z_pos = row['position_z']
        qx = row['quaternion_x']
        qy = row['quaternion_y']
        qz = row['quaternion_z']
        qw = row['quaternion_w']

        """ Write the drone state into the simulation"""
        root_state = drone_object.data.default_root_state.clone()
        root_state[:, :3] = torch.tensor([x_pos, y_pos, z_pos], device=root_state.device)
        root_state[:, 3:7] = torch.tensor([qx, qy, qz, qw], device=root_state.device)
        drone_object.write_root_pose_to_sim(root_state[:, :7])
        drone_object.write_root_velocity_to_sim(root_state[:, 7:])
        drone_object.reset()
        drone_object.write_data_to_sim()

        sim.step()

        drone_object.update(sim_dt)
                
        """ Update camera position and target to follow the drone """
        drone_pose = drone_object.data.root_state_w.clone() 
        drone_pose_np = drone_pose.cpu().numpy()     
        drone_pos = drone_pose_np[0, :3]           
        offset_local = np.array([-1.5, 0.0, 1.5])  

        camera_position = drone_pos + offset_local 
        camera_target = drone_pos

        sim.set_camera_view(camera_position.tolist(), camera_target.tolist())


        if count % 50 == 0:
            print(f"Timestamp : {sim_time:.2f} s, Root position (in world): {drone_object.data.root_pos_w}")
            
        count += 1
        sim_time += sim_dt
    

    


def live_csv_reader(file_path, start_row=0, sleep_time=0.1, timeout=30.0):
    """
    Reads a CSV file in real-time, yielding new rows as they are added.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    start_row : int, optional
        Row number to start reading from (default is 0).
    sleep_time : float, optional
        Time to wait between checks for new data (default is 0.1 seconds).
    timeout : float, optional
        Time to wait before stopping if no new data is received (default is 30 seconds).
    Returns
    -------
    generator
        A generator that yields new rows as they are added to the CSV file. 
    """
    last_row_read = start_row
    waited_time = 0.0
    print(f"[INFO] Starting to read from {file_path}...")

    while True:
        try:
            df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()

        total_rows = len(df)

        if total_rows > last_row_read:
            waited_time = 0.0
            new_rows = df.iloc[last_row_read:total_rows]
            LiftoffToIsaacCoordinates(new_rows)
            for _, row in new_rows.iterrows():
                yield row
            last_row_read = total_rows
        else:
            time.sleep(sleep_time)
            waited_time += sleep_time
            if waited_time >= timeout:
                print(f"[WARNING] No data received for {timeout} seconds. Ending stream.")
                raise StopIteration  


def toggle_recording_keypress():
    """
    Toggles recording state on keypress ('a').

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    global recording, terminate
    print("Press 'a' to start / end recording.")
    while True:
        keyboard.wait('a')
        with lock:
            recording = not recording
            if recording:
                print(f"Recording started")
            else:
                print("Recording ended")
                terminate = True
                break  


def udp_listener(port):
    """
    Listens for UDP packets and appends new rows to the udp_stack.
    Add new rows to the udp_stack.

    Parameters
    ----------
    port : int
        The UDP port to listen on.

    Returns
    -------
    None
    """
    for row in listen_trajectory(port):
        if udp_stack:
            udp_stack.pop()
        udp_stack.append(row)


def run_simulator_from_udp(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], port):
    """
    Run the simulation using UDP data stream from Liftoff.

    Parameters
    ----------
    sim : sim_utils.SimulationContext
        The simulation context.
    entities : dict[str, Articulation]
        Dictionary containing the drone object.
    port : int
        The UDP port to listen on for trajectory data.
    
    Returns
    -------
    None
    """
    drone_object = entities["drone"]
    sim_dt = sim.get_physics_dt()

    """ Start the UDP listener in a separate thread """
    threading.Thread(target=udp_listener, args=(port,), daemon=True).start()

    sim_time = 0.0
    count = 0
    while simulation_app.is_running():
        row = None
        if recording and udp_stack:
            row = udp_stack.pop()  
        udp_stack.clear()

        """ If recording is enabled and a new row is available, update the drone state."""
        if recording and row is not None:

            x_pos = row['position_x'] - 1000
            y_pos = row['position_y'] + 1070
            z_pos = row['position_z']
            qx = row['quaternion_x']
            qy = row['quaternion_y']
            qz = row['quaternion_z']
            qw = row['quaternion_w']

            root_state = drone_object.data.default_root_state.clone()
            root_state[:, :3] = torch.tensor([x_pos, y_pos, z_pos], device=root_state.device)
            root_state[:, 3:7] = torch.tensor([qx, qy, qz, qw], device=root_state.device)

            drone_object.write_root_pose_to_sim(root_state[:, :7])
            drone_object.write_root_velocity_to_sim(root_state[:, 7:])
            drone_object.reset()
            drone_object.write_data_to_sim()

            drone_object.update(sim_dt)

            if count % 50 == 0:
                print(f"Timestamp : {sim_time:.2f} s, Root position (in world): {drone_object.data.root_pos_w}")

        sim.step()

        """ Update camera position and target to follow the drone """
        drone_pose = drone_object.data.root_state_w.clone()
        drone_pose_np = drone_pose.cpu().numpy()
        drone_pos = drone_pose_np[0, :3]
        offset_local = np.array([-1.5, 0.0, 1.5])
        camera_position = drone_pos + offset_local
        camera_target = drone_pos
        sim.set_camera_view(camera_position.tolist(), camera_target.tolist())

        count += 1
        sim_time += sim_dt




def listen_trajectory(port):
    """
    Listens for UDP packets containing drone trajectory data and yields rows of data.

    Parameters
    ----------
    port : int
        The UDP port to listen on for trajectory data.
    
    Returns
    -------
    generator
        A generator that yields rows of trajectory data as pandas Series.
    """

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
        udp_socket.bind(('0.0.0.0', port))
        print(f"Listening for UDP packets on port {port}...")

        df = pd.DataFrame()

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
                LiftoffToIsaacCoordinates(row_df)
                row_serie = row_df.iloc[0]
                yield row_serie

            except Exception as e:
                print(f"Parsing error: {e}")

    

def main(args_cli):
    """
    Main function to run the drone trajectory in Isaac Lab.

    Parameters
    ----------
    args_cli : argparse.Namespace
        Command line arguments parsed by argparse.
    
    Returns
    -------
    None
    """
    
    """ Initialize the simulation application and set up the simulation context. """
    sim_cfg = sim_utils.SimulationCfg(dt=1/sampling_frequency, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    simulation_app.set_setting("/app/asyncRendering", True)
    sim.set_camera_view([-2, 0, 4], [3, 0.0, 0.0])

    scene_entities = design_scene()
    sim.reset()

    print("[INFO]: Setup complete...")

    """ Run the simulation based on the data source specified in the command line arguments. """

    if args_cli.data_source == "csv":
        csv_path = args_cli.input_file
        run_simulator_from_csv(sim, scene_entities, csv_path)
        simulation_app.close()
        exit(0)
    
    else:
        port = 9001
        thread_toggle = threading.Thread(target=toggle_recording_keypress, args=())
        thread_toggle.start()
        
        run_simulator_from_udp(sim, scene_entities, port)
        
        thread_toggle.join()
        simulation_app.close()
        exit(0)


if __name__ == "__main__":

    main(args_cli)
    simulation_app.close()
    