import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# ENTER YOUR TESTED CSV FILES HERE
# ==========================================
# Bhai, yahan apne un models ki testing CSV files ka exact naam daalna 
# jo tune alag-alag Kappa pe train aur test kiye hain.
models = {
    "Kappa = 10 (High Risk)": "results/testing/NSE_K10_Model.csv",  # Change this name!
    "Kappa = 50 (Optimal)": "results/testing/NSE_14000.csv",        # Tera champion model
    "Kappa = 100 (Safe)": "results/testing/NSE_K100_Model.csv",     # Change this name!
    "Kappa = 200 (BS Clone)": "results/testing/NSE_K200_Model.csv"  # Change this name!
}

risks = []
returns = []
labels = []

bs_risk = 0
bs_return = 0

print("Calculating Efficient Frontier Metrics...")

for label, file_path in models.items():
    try:
        df = pd.read_csv(file_path)
        e = df.groupby(['episode']).sum()
        
        # Calculate AI Metrics
        mean_pnl = e['A PnL'].mean() * 100
        std_pnl = e['A PnL'].std() * 100
        
        # Calculate BS Metrics (Only need to capture this once)
        bs_return = e['B PnL'].mean() * 100
        bs_risk = e['B PnL'].std() * 100
        
        risks.append(std_pnl)
        returns.append(mean_pnl)
        labels.append(label)
        
        print(f"{label} -> Risk: {std_pnl:.2f}, Return: {mean_pnl:.2f}")
        
    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}. Please run testing_nse.py for this model first!")

# Plotting the Master Graph
plt.figure(figsize=(10, 6))

# Plot AI Efficient Frontier Curve
plt.plot(risks, returns, marker='o', linestyle='-', color='blue', linewidth=2, markersize=8, label='Deep Hedging (AI)')

# Add labels to AI points
for i, label in enumerate(labels):
    plt.annotate(label, (risks[i], returns[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

# Plot Black-Scholes Benchmark
plt.scatter([bs_risk], [bs_return], color='orange', s=150, edgecolors='black', label='Black-Scholes (Math)', zorder=5)
plt.annotate('Black-Scholes\nBenchmark', (bs_risk, bs_return), textcoords="offset points", xytext=(0,-20), ha='center', color='darkorange', weight='bold')

plt.title('Empirical Deep Hedging: Efficient Frontier (NSE 20% Volatility)', fontsize=14, weight='bold')
plt.xlabel('Risk (Standard Deviation of PnL)', fontsize=12)
plt.ylabel('Expected Return (Mean Net Profit)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

save_path = 'results/testing/Efficient_Frontier_NSE_Final.png'
plt.savefig(save_path, dpi=300)
plt.show()

print(f"\nBOOM! Master Graph saved as {save_path}")