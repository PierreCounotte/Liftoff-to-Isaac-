

@echo off
echo [INFO] Launching the scripts ...

REM Lancer le premier script dans une nouvelle fenêtre
start "LiftoffToIsaacTraj" cmd /k "echo [INFO] Running LiftoffToIsaacTraj.py... && python LiftoffToIsaacTraj.py --input_file Liftoff_trajectory_recorded.csv && echo [INFO] LiftoffToIsaacTraj.py done && pause"

REM Lancer le deuxième script dans une autre fenêtre
start "TrajectoryLiftoff" cmd /k "echo [INFO] Running TrajectoryLiftoff.py... && python TrajectoryRecordingFromLiftoff.py --drone_id 1 --output_file Liftoff_trajectory_recorded.csv  && echo [INFO] TrajectoryLiftoff.py done && pause"




