import pandas as pd
import argparse
from scipy.spatial.transform import Rotation as R
import numpy as np



def LiftoffToIsaac(Liftoff_file, Isaac_file):
    """
    Converts a Liftoff telemetry CSV file to Isaac-compatible format. Isaac Sim is Z-Up, right-handed (Up:Z, forward:X, left:Y).
    """
    try:
        # Load the Liftoff file
        df = pd.read_csv(Liftoff_file)
        
        x_unity = df['position_x'].copy()
        df.loc[:, 'position_x'] = df['position_z']
        df.loc[:, 'position_z'] = df['position_y']
        df.loc[:, 'position_y'] = -x_unity
        
        vx_unity = df['velocity_x'].copy()
        df.loc[:, 'velocity_x'] = df['velocity_z']
        df.loc[:, 'velocity_z'] = df['velocity_y']
        df.loc[:, 'velocity_y'] = -vx_unity
        
        quat_unity = df[['quaternion_x', 'quaternion_y', 'quaternion_z', 'quaternion_w']].values
        r_unity = R.from_quat(quat_unity)  

        change_basis = R.from_matrix(np.array([
            [ 0,  0,  1],   # X_isaac ← Z_unity
            [-1,  0,  0],   # Y_isaac ← -X_unity
            [ 0,  1,  0]    # Z_isaac ← Y_unity
        ]))
        
        r_isaac = change_basis * r_unity
        quat_isaac = r_isaac.as_quat()
        
        df.loc[:, ['quaternion_x', 'quaternion_y', 'quaternion_z', 'quaternion_w']] = quat_isaac

        # Write to the Isaac file
        df.to_csv(Isaac_file, index=False)

        print(f"The Liftoff file '{Liftoff_file}' has been successfully converted to the Isaac file: '{Isaac_file}'")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Liftoff to IsaacSim conversion")
    parser.add_argument("--Liftoff_file", default="Chunked_trajectory_corrected.csv", help="Input file containing data with Liftoff convention")
    parser.add_argument("--Isaac_file", default="Chunked_trajectory_corrected_Isaac.csv", help="Output file containing data with IsaacSim convention")
    args = parser.parse_args()
    
    LiftoffToIsaac(args.Liftoff_file, args.Isaac_file)