import argparse
from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Run drone trajectory in Isaac Lab.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# add custom argument
parser.add_argument("--input_file", default="test.csv", help="CSV file containing the trajectory data")
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

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets import Articulation, ArticulationCfg

sampling_frequency = 100 # Hz


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
    

    drone_cfg = ArticulationCfg(
        prim_path="/World/Objects/Drone",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Crazyflie/cf2x.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={},  # <--- ajoute cette ligne pour éviter l'erreur
    )

    cube_cfg = RigidObjectCfg(
        prim_path="/World/Objects/GhostCube",
        spawn=sim_utils.CuboidCfg(  # Cube généré à la volée
            size=(2, 2, 2),  # Ajuste selon ton drone
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 1.0), metallic=0.2, opacity=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg()
    )


    drone_object = Articulation(cfg=drone_cfg)
    cube_object = RigidObject(cfg=cube_cfg)

    # return the scene information
    scene_entities = {"cube": cube_object, "drone": drone_object}
    return scene_entities


    cfg_cuboid.func("/World/Objects/CuboidRigid", cfg_cuboid, translation=(0.0, 0.0, 0.1))

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
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ])

    # Appliquer ce changement à toutes les quaternions
    # Soit R la rotation dans l’ancien repère.
    # Dans le nouveau repère, la rotation est : R' = P⁻¹ * R * P = P.T * R * P
    rot_mats = r_unity.as_matrix()            # (N, 3, 3)
    rot_mats_permuted = P.T @ rot_mats @ P    # changement de base
    r_permuted = R.from_matrix(rot_mats_permuted)

    # Étape 4 : Appliquer rotation de +90° autour du **nouvel axe Z**
    r_z_90 = R.from_euler('z', 90, degrees=True)   # dans le nouveau repère
    r_adjusted = r_z_90 * r_permuted



    # Étape 5 : Mise à jour dans le DataFrame (x, y, z, w)
    quat_adjusted = r_adjusted.as_quat()  # shape = (N, 4)
    df.loc[:, 'quaternion_x'] = quat_adjusted[:, 0]
    df.loc[:, 'quaternion_y'] = quat_adjusted[:, 1]
    df.loc[:, 'quaternion_z'] = quat_adjusted[:, 2]
    df.loc[:, 'quaternion_w'] = quat_adjusted[:, 3]




def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject], csv_path: str):
    cube_object, drone_object = entities["cube"], entities["drone"]
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

        # Lecture et transformation des données
        x_pos = row['position_x']-950
        y_pos = row['position_y']+1000
        z_pos = row['position_z']
        qx = row['quaternion_x']
        qy = row['quaternion_y']
        qz = row['quaternion_z']
        qw = row['quaternion_w']

        root_state = cube_object.data.default_root_state.clone()
        root_state[:, :3] = torch.tensor([x_pos, y_pos, z_pos], device=root_state.device)
        root_state[:, 3:7] = torch.tensor([qx, qy, qz, qw], device=root_state.device)

        # Mise à jour du drone et du cube
        cube_object.write_root_pose_to_sim(root_state[:, :7])
        cube_object.write_root_velocity_to_sim(root_state[:, 7:])
        drone_object.write_root_pose_to_sim(root_state[:, :7])
        drone_object.write_root_velocity_to_sim(root_state[:, 7:])

        cube_object.reset()
        drone_object.reset()
        cube_object.write_data_to_sim()
        drone_object.write_data_to_sim()
        sim.step()

        cube_object.update(sim_dt)
        drone_object.update(sim_dt)

        if count % 50 == 0:
            print(f"Timestamp : {sim_time:.2f} s, Root position (in world): {cube_object.data.root_pos_w}")
            
        if count % 1000 == 0 and count != 0:
            print(f"[INFO] --- Resetting sim_time at count = {count} ---")
            sim_time = 0.0
            print(f"Timestamp : {sim_time:.2f} s, Root position (in world): {cube_object.data.root_pos_w}")
        count += 1

        sim_time += sim_dt
    

    


def live_csv_reader(file_path, start_row=0, sleep_time=0.1, timeout=10.0):
    last_row_read = start_row
    waited_time = 0.0

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

    

def main(args_cli):
    """Main function."""
    
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=1/sampling_frequency, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([50, 50, 50], [-10, 0.0, 10])


    # Design scene by adding assets to it
    scene_entities = design_scene()

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    csv_path = args_cli.input_file
    run_simulator(sim, scene_entities, csv_path)
    simulation_app.close()
    exit(0)


if __name__ == "__main__":
    # run the main function
    main(args_cli)

    simulation_app.close()
    