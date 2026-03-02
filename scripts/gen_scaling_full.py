"""Generate scaling plots from the scaling study results."""
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("experiments/results/scaling_study.yaml") as f:
    data = yaml.safe_load(f)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Plot 1: Runtime vs K
ks = [d['K'] for d in data['scaling_vs_K']]
times_k = [d['epoch_time'] for d in data['scaling_vs_K']]
mems_k = [d['peak_memory_mb'] for d in data['scaling_vs_K']]

ax1 = axes[0]
color1 = '#2196F3'
color2 = '#FF5722'
ax1.plot(ks, times_k, 'o-', color=color1, linewidth=2, markersize=6, label='Runtime (s)')
ax1.set_xlabel('Number of Prototypes $K$', fontsize=12)
ax1.set_ylabel('Epoch Time (s)', color=color1, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xscale('log')

ax1b = ax1.twinx()
ax1b.plot(ks, mems_k, 's--', color=color2, linewidth=2, markersize=6, label='Memory (MB)')
ax1b.set_ylabel('Peak Memory (MB)', color=color2, fontsize=12)
ax1b.tick_params(axis='y', labelcolor=color2)
ax1.set_title('Scaling vs $K$', fontsize=13, fontweight='bold')

# Plot 2: Runtime vs T
ts = [d['T'] for d in data['scaling_vs_T']]
times_t = [d['epoch_time'] for d in data['scaling_vs_T']]

ax2 = axes[1]
ax2.plot(ts, times_t, 'o-', color='#4CAF50', linewidth=2, markersize=6)
ax2.set_xlabel('Sinkhorn Iterations $T$', fontsize=12)
ax2.set_ylabel('Epoch Time (s)', fontsize=12)
ax2.set_title('Scaling vs $T$', fontsize=13, fontweight='bold')

# Plot 3: Runtime vs B
bs = [d['B'] for d in data['scaling_vs_B']]
times_b = [d['epoch_time'] for d in data['scaling_vs_B']]
mems_b = [d['peak_memory_mb'] for d in data['scaling_vs_B']]

ax3 = axes[2]
ax3.plot(bs, times_b, 'o-', color='#9C27B0', linewidth=2, markersize=6, label='Runtime')
ax3.set_xlabel('Batch Size $B$', fontsize=12)
ax3.set_ylabel('Epoch Time (s)', color='#9C27B0', fontsize=12)
ax3.tick_params(axis='y', labelcolor='#9C27B0')

ax3b = ax3.twinx()
ax3b.plot(bs, mems_b, 's--', color='#FF9800', linewidth=2, markersize=6, label='Memory')
ax3b.set_ylabel('Peak Memory (MB)', color='#FF9800', fontsize=12)
ax3b.tick_params(axis='y', labelcolor='#FF9800')
ax3.set_title('Scaling vs Batch Size $B$', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('paper/figures/scaling_full.png', dpi=200, bbox_inches='tight')
print("Saved paper/figures/scaling_full.png")
