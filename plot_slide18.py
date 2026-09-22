import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set a clean, professional style for presentations
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 12

# Load the benchmark data
df = pd.read_csv('benchmark_results.csv')

# Create a 1x2 figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# ==========================================
# PLOT 1: Throughput vs Chunk Size
# ==========================================
# Extract the 40k (Real Space) and 100k (Stress Test) data
df_40k = df[df['N_run'] == 40000].sort_values('chunk_size')
df_100k = df[df['N_run'] == 100000].sort_values('chunk_size')

ax1.plot(df_40k['chunk_size'], df_40k['throughput_sps'], marker='o', markersize=8, 
         linewidth=2.5, color='#2ca02c', label='40k (Real Space Ceiling)')
ax1.plot(df_100k['chunk_size'], df_100k['throughput_sps'], marker='s', markersize=8, 
         linewidth=2.5, color='#1f77b4', label='100k (Stress Test)')

# Formatting
ax1.set_xscale('log')
ax1.set_xticks([500, 1000, 2000, 5000, 10000, 20000])
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.set_xlabel('Chunk Size (Batch Size)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Throughput (samples / sec)', fontsize=14, fontweight='bold')
ax1.set_title('Throughput vs Chunk Size\n(Saturates at ~10k-20k)', fontsize=16, fontweight='bold', pad=15)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(fontsize=12, loc='lower right')

# Highlight the optimal setting
opt_chunk = 20000
opt_thr = df_100k[df_100k['chunk_size'] == opt_chunk]['throughput_sps'].values[0]
ax1.annotate(f'Optimal: {opt_thr:.0f} samples/sec\n@ chunk = 20,000', 
             xy=(opt_chunk, opt_thr), xytext=(1500, opt_thr - 60),
             arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10),
             fontsize=13, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.9))

# ==========================================
# PLOT 2: Time vs Sample Count (Linearity)
# ==========================================
# Extract data for the optimal chunk size (20,000)
df_opt = df[df['chunk_size'] == 20000].sort_values('N_run')
ax2.plot(df_opt['N_run'], df_opt['inf_time_s'], marker='D', markersize=8, 
         linewidth=2.5, color='#d62728', label='Actual Inference Time')

# Add a linear regression line to prove perfect scaling
z = np.polyfit(df_opt['N_run'], df_opt['inf_time_s'], 1)
p = np.poly1d(z)
x_lin = np.linspace(df_opt['N_run'].min(), df_opt['N_run'].max(), 100)
ax2.plot(x_lin, p(x_lin), "--", color='gray', linewidth=2.5, label='Linear Fit ($R^2 \\approx 1.0$)')

# Formatting
ax2.set_xlabel('Total Samples (N)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Inference Time (seconds)', fontsize=14, fontweight='bold')
ax2.set_title('Inference Time vs Sample Count\n(Perfect Linear Scaling)', fontsize=16, fontweight='bold', pad=15)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(fontsize=12, loc='upper left')

# Format X-axis to show 'k' (e.g., 20k, 40k)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k' if x >= 1000 else str(int(x))))

# Add a text box extrapolating to 1 Million samples
extrap_1M = p(1000000)
textstr = f'Extrapolation to 1M samples:\n~{extrap_1M/60:.0f} minutes'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black')
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=13,
         verticalalignment='top', bbox=props, fontweight='bold')

# ==========================================
# SAVE AND SHOW
# ==========================================
plt.tight_layout()
plt.savefig('slide18_throughput_visuals.png', dpi=300, bbox_inches='tight')
print("Success! Saved 'slide18_throughput_visuals.png'")
plt.show()