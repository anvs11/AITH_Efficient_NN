@echo off
echo [1/4] Spearman correlation (CIFAR-100)
call python cifar/run_spearman.py --workers 2
echo [2/4] Partial training: proxy 25 ep
call python cifar/run_partial_training.py --proxy-epochs 25 --fraction 0.5 --target-epochs 100 --workers 2
echo [3/4] Partial training: proxy 50 ep
call python cifar/run_partial_training.py --proxy-epochs 50 --fraction 0.5 --target-epochs 100 --workers 2
echo [4/4] Partial training: proxy 100 ep
call python cifar/run_partial_training.py --proxy-epochs 100 --fraction 0.5 --target-epochs 100 --workers 2
rem опционально референс-линия: call python cifar/run_partial_training.py --mode random --fraction 0.5 --target-epochs 100 --workers 2
echo ALL CIFAR-100 DONE
pause
