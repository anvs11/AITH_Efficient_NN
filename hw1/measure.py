import torch
import torch.nn as nn
import time
import numpy as np
import pandas as pd
import pynvml
from models import SmallCNN
from equations import flops, memory, bytes_moved

# вызовет OOM на конфигурациях
torch.cuda.set_per_process_memory_fraction(0.285)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)


base_S = [32, 64, 128, 224, 256, 384, 512]
random_S = [48, 144, 272, 416]
all_S = sorted(base_S + random_S)

base_B = [1, 2, 4, 8, 16, 32, 64, 128, 256]
random_B = [3, 11, 201]
all_B = sorted(base_B + random_B)

device = torch.device("cuda")
model = SmallCNN().to(device).eval()

results = []

for S in all_S:
    for B in all_B:
        print(f"Running S={S}, B={B}")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        dummy_input = torch.randn(B, 3, S, S).to(device)

        try:
            with torch.inference_mode():
                # прогрев, чтобы первый запуск не искажал статистику
                _ = model(dummy_input)

                num_runs = 15
                times_ms = []

                power_before = pynvml.nvmlDeviceGetPowerUsage(handle)

                with torch.inference_mode():
                    for _ in range(num_runs):
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)

                        start_event.record()
                        _ = model(dummy_input)
                        end_event.record()

                        torch.cuda.synchronize()
                        times_ms.append(start_event.elapsed_time(end_event))

                power_after = pynvml.nvmlDeviceGetPowerUsage(handle)

                median_time_ms = np.median(times_ms)
                latency_sec = median_time_ms / 1000.0

                peak_mem_bytes = torch.cuda.max_memory_allocated()

                avg_power_watts = ((power_before + power_after) / 2) / 1000.0
                energy_joules = avg_power_watts * latency_sec

            results.append({
                'S': S,
                'B': B,
                'latency': latency_sec,
                'memory': peak_mem_bytes,
                'energy': energy_joules,
                'is_oom': False,
                'is_validation': (S in random_S) or (B in random_B)
            })

        except torch.cuda.OutOfMemoryError:
            print(f" -> OOM at S={S}, B={B}")
            results.append({
                'S': S,
                'B': B,
                'latency': np.nan,
                'memory': np.nan,
                'energy': np.nan,
                'is_oom': True,
                'is_validation': (S in random_S) or (B in random_B)
            })

df = pd.DataFrame(results)
df.to_csv('results/measurements.csv', index=False)
print("Measurements saved to results/measurements.csv")

pynvml.nvmlShutdown()
