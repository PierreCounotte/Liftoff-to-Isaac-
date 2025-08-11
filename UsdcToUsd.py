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

from pxr import Usd, UsdPhysics

def print_hierarchy(prim, indent=0):
    print("  " * indent + str(prim.GetName()))
    for child in prim.GetChildren():
        print_hierarchy(child, indent + 1)

stage = Usd.Stage.Open("C:/Users/Administrateur/Documents/DroneProject/Liftoff-to-Isaac-/Vapor_X5/my_drone/my_drone.usd")

root = stage.GetPseudoRoot()  # récupère la racine du stage
print_hierarchy(root)