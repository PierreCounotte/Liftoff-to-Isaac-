

@echo off
echo [INFO] Launching the scripts ...

REM Lancer le premier script dans une nouvelle fenêtre
start "LiftoffToIsaacTraj" cmd /k "echo [INFO] Running LiftoffToIsaacTraj.py... && python LiftoffToIsaacTraj.py --input_file Test.csv && echo [INFO] LiftoffToIsaacTraj.py done && pause"

REM Lancer le deuxième script dans une autre fenêtre
start "TrajectoryLiftoff" cmd /k "echo [INFO] Running TrajectoryLiftoff.py... && python TrajectoryLiftoff.py --drone_id 1 --output_file Liftoff_trajectory.csv --chunks 0 && echo [INFO] TrajectoryLiftoff.py done && pause"




