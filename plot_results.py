import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def plot_performance(model_name):
    file_path = f"results/testing/{model_name}.csv"
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} nahi mila bhai!")
        return

    print(f"Loading REAL data from {file_path}...")
    
    # Is baar hum proper headers ke sath padh rahe hain
    df = pd.read_csv(file_path, low_memory=False) 
    
    # THE REAL FIX: Har episode ke profit ko jodna (Groupby) aur 100 se multiply karna (jaise testing.py karta hai)
    episode_sums = df.groupby(['episode']).sum() * 100
    
    # Asli Profit wale columns uthana
    if 'A PnL' in episode_sums.columns and 'B PnL' in episode_sums.columns:
        agent_pnl = episode_sums['A PnL'].dropna()
        bs_pnl = episode_sums['B PnL'].dropna()
    else:
        print("Error: Asli Profit wale columns nahi mile!")
        return

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(14, 6))
    
    # ==========================================
    # Plot 1: Histogram (Risk & Volatility Distribution)
    # ==========================================
    plt.subplot(1, 2, 1)
    sns.kdeplot(agent_pnl, color='blue', label='AI Agent (TD3)', fill=True, alpha=0.3)
    sns.kdeplot(bs_pnl, color='orange', label='Black-Scholes (Math)', fill=True, alpha=0.3)
    plt.title('Real PnL Distribution (Episode Level)', fontsize=14, fontweight='bold')
    plt.xlabel('Profit / Loss ($)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    
    # Adding vertical lines for Mean values
    plt.axvline(agent_pnl.mean(), color='blue', linestyle='dashed', linewidth=1)
    plt.axvline(bs_pnl.mean(), color='orange', linestyle='dashed', linewidth=1)
    plt.legend()

    # ==========================================
    # Plot 2: Bar Chart (Mean Profit Comparison)
    # ==========================================
    plt.subplot(1, 2, 2)
    means = [agent_pnl.mean(), bs_pnl.mean()]
    bars = plt.bar(['AI Agent (TD3)', 'Black-Scholes'], means, color=['blue', 'orange'], width=0.5)
    plt.title('Real Average Profit (Mean)', fontsize=14, fontweight='bold')
    plt.ylabel('Profit ($)', fontsize=12)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (abs(yval)*0.01), f"${round(yval, 2)}", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    
    save_path = f"results/testing/{model_name}_Performance_Graph.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nBoom! Asli Graph save ho gaya hai yahan: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    model = "Crypto_19000"
    if len(sys.argv) > 1:
        model = sys.argv[1]
    plot_performance(model)