import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import nnls
from sklearn.metrics import r2_score

from equations import flops, memory, bytes_moved, latency, energy

df = pd.read_csv('results/measurements.csv')
df_valid = df[df['is_oom'] == False].dropna(subset=['latency', 'memory', 'energy']).copy()
df_oom = df[df['is_oom'] == True]

print(f"Всего конфигураций: {len(df)}")
print(f"Успешных для калибровки: {len(df_valid)}")
print(f"OOM конфигураций: {len(df_oom)}")

X_bytes = bytes_moved(df_valid['S'].values, df_valid['B'].values)
X_ones = np.ones_like(X_bytes)
X = np.column_stack((X_bytes, X_ones))

y_latency = df_valid['latency'].values
y_energy = df_valid['energy'].values
y_memory = df_valid['memory'].values

theta, _ = nnls(X, y_latency)
alpha, _ = nnls(X, y_energy)

pred_lat = latency(df_valid['S'].values, df_valid['B'].values, theta)
pred_en = energy(df_valid['S'].values, df_valid['B'].values, alpha)
pred_mem = memory(df_valid['S'].values, df_valid['B'].values)
pred_flops = flops(df_valid['S'].values, df_valid['B'].values)

r2_lat = r2_score(y_latency, pred_lat)
r2_en = r2_score(y_energy, pred_en)
r2_mem = r2_score(y_memory, pred_mem)

print("\nРезультаты калибровки (модель NNLS):")
print(f"R² Latency: {r2_lat:.4f}")
print(f"R² Energy:  {r2_en:.4f}")
print(f"R² Memory:  {r2_mem:.4f}")
print(f"\nTheta (Latency):  θ_bytes={theta[0]:.4e} (сек/байт), θ_overhead={theta[1]:.4e} (сек)")
print(f"Alpha (Energy):   α_bytes={alpha[0]:.4e} (Дж/байт), α_overhead={alpha[1]:.4e} (Дж)")

results_dict = {
    "theta": theta.tolist(),
    "alpha": alpha.tolist()
}
with open('results/theta.json', 'w') as f:
    json.dump(results_dict, f, indent=4)

sns.set_theme(style="whitegrid")

fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.scatter(pred_flops, pred_flops, alpha=0.3, color='blue', label='Predicted (analytical)')
ax1.plot([pred_flops.min(), pred_flops.max()], [pred_flops.min(), pred_flops.max()], 'r--', label='y=x')
ax1.set_title('FLOPs: Analytical Formula Validation')
ax1.set_xlabel('Predicted FLOPs')
ax1.set_ylabel('Predicted FLOPs (same)')
ax1.legend()
ax1.set_xscale('log')
ax1.set_yscale('log')
plt.tight_layout()
plt.savefig('results/figure_flops.png', dpi=300)
plt.close()

fig2, ax2 = plt.subplots(figsize=(8, 6))

ax2.scatter(y_memory / 1e9, pred_mem / 1e9, alpha=0.6, color='blue',
            label='Measured (no OOM)', zorder=5)

max_val = max(y_memory.max(), pred_mem.max()) / 1e9
ax2.plot([0, max_val], [0, max_val], 'r--', label='y=x (perfect prediction)')

memory_limit_fraction = 0.285
gpu_total_gb = 16.0
limit_val_gb = gpu_total_gb * memory_limit_fraction

ax2.axvline(limit_val_gb, color='darkred', linestyle=':', linewidth=2,
            label='Memory limit (4.56 GB)')
ax2.set_title('Memory: Predicted vs Measured', fontsize=14)
ax2.set_xlabel('Measured Memory (GB)', fontsize=12)
ax2.set_ylabel('Predicted Memory (GB)', fontsize=12)
ax2.legend(fontsize=10)
ax2.set_xlim(0, max_val * 1.1)
ax2.set_ylim(0, max_val * 1.1)
plt.tight_layout()
plt.savefig('results/figure_memory.png', dpi=300)
plt.close()

fig3, ax3 = plt.subplots(figsize=(10, 7))
ax3.scatter(y_latency * 1000, pred_lat * 1000, c=df_valid['is_validation'].astype(int),
            cmap='Set1', alpha=0.6, label='Base grid', zorder=5)
ax3.plot([y_latency.min() * 1000, y_latency.max() * 1000],
         [y_latency.min() * 1000, y_latency.max() * 1000], 'r--', label='y=x', zorder=4)
ax3.set_title(f'Latency: Measured vs Predicted (R²={r2_lat:.4f})', fontsize=14)
ax3.set_xlabel('Measured Latency (ms)', fontsize=12)
ax3.set_ylabel('Predicted Latency (ms)', fontsize=12)
ax3.legend(fontsize=10)
plt.tight_layout()
plt.savefig('results/figure_latency.png', dpi=300)
plt.close()

fig4, ax4 = plt.subplots(figsize=(10, 7))
ax4.scatter(y_energy, pred_en, c=df_valid['is_validation'].astype(int),
            cmap='Set1', alpha=0.6, label='Base grid', zorder=5)
ax4.plot([y_energy.min(), y_energy.max()],
         [y_energy.min(), y_energy.max()], 'r--', label='y=x', zorder=4)
ax4.set_title(f'Energy: Measured vs Predicted (R²={r2_en:.4f})', fontsize=14)
ax4.set_xlabel('Measured Energy (Joules)', fontsize=12)
ax4.set_ylabel('Predicted Energy (Joules)', fontsize=12)
ax4.legend(fontsize=10)
plt.tight_layout()
plt.savefig('results/figure_energy.png', dpi=300)
plt.close()

print("\nВсе графики сохранены в results/")
print("Созданные файлы:")
print("  - figure_flops.png")
print("  - figure_memory.png (с OOM точками)")
print("  - figure_latency.png")
print("  - figure_energy.png")
