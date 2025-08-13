import argparse
from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Run drone trajectory in Isaac Lab.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# add custom argument
parser.add_argument("--input_file", default="test.csv", help="CSV file containing the trajectory data")
parser.add_argument("--data_source", choices=["csv", "udp"], help="Source of the trajectory data")
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

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
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg, Articulation, ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pxr import Usd

recording = False
terminate = False
lock = threading.Lock()

udp_stack = []


sampling_frequency = 10000 # Hz


def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)
    # spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),  
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    

    # create a new xform prim for all objects to be spawned under
    prim_utils.create_prim("/World/Objects", "Xform")

    """drone_cfg = RigidObjectCfg(
        prim_path="/World/Objects/Vapor_X5",
        spawn=sim_utils.UsdFileCfg(
            #usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Crazyflie/cf2x.usd",
            usd_path = f"C:/Users/Administrateur/Documents/DroneProject/Liftoff-to-Isaac-/Vapor_X5/my_drone/my_drone.usd",
            #rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg()
    )"""

    


    drone_cfg = ArticulationCfg(
        prim_path="/World/Objects/Vapor_X5",
        spawn=sim_utils.UsdFileCfg(
            #usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Crazyflie/cf2x.usd",
            usd_path = f"C:/Users/Administrateur/Documents/DroneProject/Liftoff-to-Isaac-/Vapor_X5_along_x/my_drone/my_drone.usd",
            #rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={},  
    )

    


    drone_object = Articulation(cfg=drone_cfg)
    #drone_object = RigidObject(cfg=drone_cfg)

    # return the scene information
    scene_entities = {"drone": drone_object}
    return scene_entities



def LiftoffToIsaacCoordinates(df):
    x_unity = df['position_x'].copy()
    df.loc[:, 'position_x'] = df['position_z']
    df.loc[:, 'position_z'] = df['position_y']
    df.loc[:, 'position_y'] = -x_unity

    vx_unity = df['velocity_x'].copy()
    df.loc[:, 'velocity_x'] = df['velocity_z']
    df.loc[:, 'velocity_z'] = df['velocity_y']
    df.loc[:, 'velocity_y'] = -vx_unity




    
    # Unity format: (x, y, z, w)
    quat_unity = df[['quaternion_x', 'quaternion_y', 'quaternion_z', 'quaternion_w']].values  # shape = (N, 4)

    # Étape 1 : Conversion en rotation
    r_unity = R.from_quat(quat_unity)

    P = np.array([
        [0, 0, 1],
        [-1, 0, 0],
        [0, 1, 0]
    ])


    rot_mats = r_unity.as_matrix()
    rot_mats_converted = P @ rot_mats @ P.T

    angles = R.from_matrix(rot_mats_converted).as_euler('xyz', degrees=False)

    # Swap roll et yaw
    angles[:, [0, 2]] = angles[:, [2, 0]]
    angles[:, 0] *= -1  # Roll
    angles[:, 2] *= -1  # Yaw
    

    r_converted = R.from_euler('xyz', angles, degrees=False)

    r_z_180 = R.from_euler('x', -180, degrees=True)   
    r_final = r_z_180 * r_converted


    quat_converted = r_final.as_quat()
    df['quaternion_x'] = quat_converted[:, 0]
    df['quaternion_y'] = quat_converted[:, 1]
    df['quaternion_z'] = quat_converted[:, 2]
    df['quaternion_w'] = quat_converted[:, 3]




def run_simulator_from_csv(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject], csv_path: str):
    drone_object = entities["drone"]
    sim_dt = sim.get_physics_dt()

    if not os.path.exists(csv_path):
        print(f"{csv_path} does not exist.")
        return
    row_stream = live_csv_reader(csv_path)

    sim_time = 0.0
    count = 0


    while simulation_app.is_running():
        try:
            row = next(row_stream)
        except StopIteration:
            print("[INFO] Data flux end.")
            return


        x_pos = row['position_x']-1000
        y_pos = row['position_y']+1070
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
        sim.step()

        drone_object.update(sim_dt)

                
        
        drone_pose = drone_object.data.root_state_w.clone() 
        drone_pose_np = drone_pose.cpu().numpy()     # shape: (1, 13)
        drone_pos = drone_pose_np[0, :3]              # (x, y, z)
        drone_rot = drone_pose_np[0, 3:7]   


        
        offset_local = np.array([-1.5, 0.0, 1.5])  


        camera_position = drone_pos + offset_local 
        camera_target = drone_pos

        sim.set_camera_view(camera_position.tolist(), camera_target.tolist())


        if count % 50 == 0:
            print(f"Timestamp : {sim_time:.2f} s, Root position (in world): {drone_object.data.root_pos_w}")
            
        if count % 1000 == 0 and count != 0:
            print(f"[INFO] --- Resetting sim_time at count = {count} ---")
            sim_time = 0.0
            print(f"Timestamp : {sim_time:.2f} s, Root position (in world): {drone_object.data.root_pos_w}")
        count += 1

        sim_time += sim_dt
    

    


def live_csv_reader(file_path, start_row=0, sleep_time=0.1, timeout=30.0):
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

# --- Thread : lit l'UDP et stocke les lignes ---
def udp_listener(port):
    for row in listen_trajectory(port):
        # Remplace complètement l'élément précédent
        if udp_stack:
            udp_stack.pop()
        udp_stack.append(row)

# --- Boucle principale : lit la queue et met à jour la simulation ---
def run_simulator_from_udp(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject], port):
    drone_object = entities["drone"]
    sim_dt = sim.get_physics_dt()

    # Lancer l'écoute UDP dans un thread séparé
    threading.Thread(target=udp_listener, args=(port,), daemon=True).start()

    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        row = None
        if recording and udp_stack:
            row = udp_stack.pop()  # Prend la dernière position
        # Ne garde rien d'autre dans la stack pour éviter le retard
        udp_stack.clear()

        # --- Mise à jour UDP et drone seulement si recording ---
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

        # --- Faire avancer la simulation toujours ---
        sim.step()

        # --- Mise à jour caméra toujours ---
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
    #global recording, terminate

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

        # À la fin, fermer le fichier CSV s'il est ouvert
    

def main(args_cli):
    """Main function."""
    
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=1/sampling_frequency, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([-2, 0, 4], [3, 0.0, 0.0])
    # Design scene by adding assets to it
    scene_entities = design_scene()
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

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
    # run the main function
    main(args_cli)

    simulation_app.close()
    