@echo off
:: run_humanoid_amp.bat - Script to run Humanoid AMP Walk experiment
echo Starting Humanoid AMP Walk experiment...


:: Run the first experiment
echo Running experiment 2...
call isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task=Isaac-Simon-Walk-Direct-v0-normal --algorithm=AMP --headless

:: Check for errors
if %ERRORLEVEL% neq 0 (
    echo Error: Experiment failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

:: Run the first experiment
echo Running experiment 3...
call isaaclab.bat -p scripts\reinforcement_learning\skrl\train.py --task=Isaac-Simon-Walk-Direct-v0-old --algorithm=AMP --headless

:: Check for errors
if %ERRORLEVEL% neq 0 (
    echo Error: Experiment failed with error code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)


echo Experiment completed successfully.
pause

:: isaaclab.bat -p scripts\reinforcement_learning\skrl\biomech_eval.py --algorithm=AMP --num_envs=1 --task=Isaac-Simon-Walk-Direct-v0-normal --checkpoint=