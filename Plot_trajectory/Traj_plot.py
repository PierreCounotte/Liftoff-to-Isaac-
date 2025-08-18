from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import numpy as np
import threading 
import pandas as pd
from matplotlib.cm import ScalarMappable




def plot_3d_trajectory(udp_queue, vmax=40.0):
    """
    Plot 3D trajectory of the drone with color representing velocity norm.
    Trajectory does not erase previous points.
    Colors: blue (slow) -> red (fast)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    lock = threading.Lock()
    
    drone_data_df = pd.DataFrame(columns=[
        'timestamp', 'position_x', 'position_y', 'position_z',
        'quaternion_x', 'quaternion_y', 'quaternion_z', 'quaternion_w',
        'velocity_x', 'velocity_y', 'velocity_z',
        'gyro_pitch', 'gyro_roll', 'gyro_yaw',
        'input_throttle', 'input_yaw', 'input_pitch', 'input_roll',
        'battery_percentage', 'battery_voltage',
        'num_motors', 'left_front_rpm', 'right_front_rpm',
        'left_back_rpm', 'right_back_rpm'
    ])

    # Stockage de la trajectoire complète
    all_x, all_y, all_z, all_velocities = [], [], [], []

    # Scatter pour la trajectoire
    trajectory_scatter = ax.scatter([], [], [], c=[], cmap='jet', s=10, vmin=0, vmax=vmax)
    cbar = fig.colorbar(trajectory_scatter, ax=ax, pad=0.1)
    cbar.set_label('Velocity norm (m/s)')

    # Point bleu pour le drone actuel
    drone_point = ax.scatter([], [], [], color='blue', s=50, label='Drone')

    first_point = None

    def update(frame):
        nonlocal first_point
        with lock:
            while not udp_queue.empty():
                row = udp_queue.get()
                drone_data_df.loc[len(drone_data_df)] = row

            if len(drone_data_df) < 2:
                return

            new_data = drone_data_df.iloc[len(all_x):]
            x = new_data['position_x'].to_numpy()
            y = new_data['position_y'].to_numpy()
            z = new_data['position_z'].to_numpy()
            vx = new_data['velocity_x'].to_numpy()
            vy = new_data['velocity_y'].to_numpy()
            vz = new_data['velocity_z'].to_numpy()
            velocities = np.sqrt(vx**2 + vy**2 + vz**2)

            all_x.extend(x)
            all_y.extend(y)
            all_z.extend(z)
            all_velocities.extend(velocities)

            if first_point is None:
                first_point = {'x': all_x[0], 'y': all_y[0], 'z': all_z[0]}

        # Mettre à jour scatter
        trajectory_scatter._offsets3d = (all_x, all_y, all_z)
        trajectory_scatter.set_array(np.array(all_velocities))

        # Mettre à jour le point actuel
        drone_point._offsets3d = ([all_x[-1]], [all_y[-1]], [all_z[-1]])

        # Ajuster les limites dynamiquement
        ax.set_xlim(first_point['x'] - 250, first_point['x'] + 250)
        ax.set_ylim(first_point['y'] - 250, first_point['y'] + 250)
        ax.set_zlim(first_point['z'] - 100, first_point['z'] + 100)

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Drone 3D Trajectory (color = velocity norm)")

    ani = FuncAnimation(fig, update, interval=50)
    plt.show()



def plot_2d_trajectory(udp_queue, vmax=40.0):
    """
    Plot X-Z trajectory with color representing velocity norm.
    Trajectory does not erase previous points.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    lock = threading.Lock()
    
    drone_data_df = pd.DataFrame(columns=[
        'timestamp', 'position_x', 'position_y', 'position_z',
        'velocity_x', 'velocity_y', 'velocity_z'
    ])
    
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Drone X-Z Trajectory (color = velocity norm)")

    # Listes pour stocker la trajectoire complète
    all_x = []
    all_z = []
    all_velocities = []

    # Scatter pour la trajectoire
    trajectory_scatter = ax.scatter([], [], c=[], cmap='jet', s=10, vmin=0, vmax=vmax)
    cbar = fig.colorbar(trajectory_scatter, ax=ax)
    cbar.set_label('Velocity norm (m/s)')

    # Point bleu pour le drone actuel
    drone_point = ax.scatter([], [], color='blue', s=50, label='Drone')

    def update(frame):
        with lock:
            while not udp_queue.empty():
                row = udp_queue.get()
                drone_data_df.loc[len(drone_data_df)] = row

            if len(drone_data_df) < 2:
                return

            # Tous les nouveaux points depuis la dernière frame
            new_data = drone_data_df.iloc[len(all_x):]
            x = new_data['position_x'].to_numpy()
            z = new_data['position_z'].to_numpy()
            vx = new_data['velocity_x'].to_numpy()
            vz = new_data['velocity_z'].to_numpy()
            velocities = np.sqrt(vx**2 + vz**2)

            # Ajouter aux listes globales
            all_x.extend(x)
            all_z.extend(z)
            all_velocities.extend(velocities)

        # Mettre à jour scatter pour la trajectoire complète
        trajectory_scatter.set_offsets(np.column_stack((all_x, all_z)))
        trajectory_scatter.set_array(np.array(all_velocities))

        # Mettre à jour le drone actuel
        drone_point.set_offsets([all_x[-1], all_z[-1]])

        # Ajuster limites dynamiquement
        ax.set_xlim(min(all_x)-50, max(all_x)+50)
        ax.set_ylim(min(all_z)-50, max(all_z)+50)

    ani = FuncAnimation(fig, update, interval=50)
    plt.show()
