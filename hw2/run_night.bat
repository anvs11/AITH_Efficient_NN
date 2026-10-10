@echo off
echo Night run: SVP CIFAR-10 experiments
echo ---------------------------------------------------
echo.

echo [1/7] Run: Random 30%% ...
call python cifar/run_coreset.py --method random --fraction 0.3
echo.

echo [2/7] Run: Random 50%% ...
call python cifar/run_coreset.py --method random --fraction 0.5
echo.

echo [3/7] Run: Forgetting Events 30%% ...
call python cifar/run_coreset.py --method forgetting_events --fraction 0.3
echo.

echo [4/7] Run: Forgetting Events 50%% ...
call python cifar/run_coreset.py --method forgetting_events --fraction 0.5
echo.

echo [5/7] Run: Entropy 50%% ...
call python cifar/run_coreset.py --method entropy --fraction 0.5
echo.

echo [6/7] Run: Forgetting Events 70%% ...
call python cifar/run_coreset.py --method forgetting_events --fraction 0.7
echo.

echo [7/7] Run: Full Dataset (100%) ...
call python cifar/run_coreset.py --method random --fraction 1.0
echo.

echo ---------------------------------------------------
echo All experiments completed!
pause
