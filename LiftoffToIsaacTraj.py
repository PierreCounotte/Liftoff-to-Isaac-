import argparse
from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on spawning prims into the scene.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import pandas as pd
import numpy as np

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext

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
    

    # Rigid Object
    cube_cfg = RigidObjectCfg(
        prim_path="/World/Objects/CuboidRigid",
        spawn=sim_utils.MeshCuboidCfg(
            size=(2, 2, 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    cube_object = RigidObject(cfg=cube_cfg)

    # return the scene information
    scene_entities = {"cube": cube_object}
    return scene_entities


    cfg_cuboid.func("/World/Objects/CuboidRigid", cfg_cuboid, translation=(0.0, 0.0, 0.1))

def LiftoffToIsaacCoordinates(df):
    x_unity = df['position_x'].copy()
    df.loc[:, 'position_x'] = df['position_z']
    df.loc[:, 'position_z'] = df['position_y']
    df.loc[:, 'position_y'] = -x_unity

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject]):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    cube_object = entities["cube"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    csv_path = "Big_trajectory_corrected.csv"
    df = pd.read_csv(csv_path)
    LiftoffToIsaacCoordinates(df)

    max_count = len(df)


    

    while simulation_app.is_running():
        if count < max_count:
                

            x_pos = df['position_x'].iloc[count]-np.mean(df['position_x'].values)
            y_pos = df['position_y'].iloc[count]-np.mean(df['position_y'].values)
            z_pos = df['position_z'].iloc[count]

            root_state = cube_object.data.default_root_state.clone()
            offset = torch.tensor([x_pos, y_pos, z_pos], device=root_state.device)
            root_state[:, :3] = offset  
            cube_object.write_root_pose_to_sim(root_state[:, :7])
            cube_object.write_root_velocity_to_sim(root_state[:, 7:])

            cube_object.reset()
            cube_object.write_data_to_sim()

            sim.step()
            cube_object.update(sim.get_physics_dt())

            if count % 50 == 0:
                print(f"Timestamp : {sim_time:.2f} s, Root position (in world): {cube_object.data.root_pos_w}")
            
            if count % 1000 == 0 and count != 0:
                print(f"[INFO] --- Resetting sim_time at count = {count} ---")
                sim_time = 0.0
                print(f"Timestamp : {sim_time:.2f} s, Root position (in world): {cube_object.data.root_pos_w}")

            sim_time += sim_dt
            count += 1

        else:
            print("End of the CSV file, simulation ended.")
            simulation_app.close()


    

def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=1/sampling_frequency, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([50, 50, 50], [-0.5, 0.0, 0.5])


    # Design scene by adding assets to it
    scene_entities = design_scene()

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # run the main function
    main()

    simulation_app.close()
    