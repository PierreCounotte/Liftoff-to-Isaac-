import sys
import os
import argparse
import threading
import queue

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Traj_listener import udp_listener
from Traj_plot import plot_3d_trajectory, plot_2d_trajectory 

def main():
    parser = argparse.ArgumentParser(description="Run drone telemetry visualization")
    parser.add_argument("--plot", choices=["3d", "2d"], default="2d", help="Type of visualization")
    parser.add_argument("--window", type=int, default=200, help="Window size for visualization")
    parser.add_argument("--port", type=int, default=9001, help="UDP port to listen on")
    args = parser.parse_args()

    # Queue partagée
    udp_queue = queue.Queue()

    # Lancer le listener UDP dans un thread
    listener_thread = threading.Thread(target=udp_listener, args=(args.port, udp_queue))
    listener_thread.daemon = True
    listener_thread.start()

    # Lancer le plot 3D
    if args.plot == "3d":
        plot_3d_trajectory(udp_queue)
    elif args.plot == "2d":
        plot_2d_trajectory(udp_queue)


if __name__ == "__main__":
    main()