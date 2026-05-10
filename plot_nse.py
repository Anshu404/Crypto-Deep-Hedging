import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def plot_nse(model_name):
    # Extension handle karna
    if not model_name.endswith('.csv'):
        file_path = f"results/testing/{model_name}.csv"
    else:
        file_path = f"results/testing/{model_name}"
        model_name = model_name.replace(".csv", "")
    
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} nahi mila bhai! Naam check kar.")
        return

    print(f"📂 Loading REAL NSE data from {file_path}...")
    
    df = pd.read_csv(file_path, low_memory=False) 
    
    # 🛡️ THE ARRAY-BRACKET CLEANER (0.0000 bug se bachayega) 🛡️
    core_cols = ['A PnL', 'B PnL']
    for col in core_cols:
        if col in df.columns:
            clean_str = df[col].astype(str).str.replace(r'[\[\]]', '', regex=True)
            df[col] = pd.to_numeric(clean_str, errors='coerce').fillna(0)
        else:
            print(f"⚠️ Error: '{col}' column nahi mila CSV mein!")
            return

    # Har episode ke profit ko jodna aur 100 se multiply karna
    episode_sums = df.groupby(['episode'])[core_cols].sum() * 100
    
    agent_pnl = episode_sums['A PnL']
    bs_pnl = episode_sums['B PnL']

    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(14, 6))
    
    # ==========================================
    # Plot 1: Histogram (Risk & Volatility Distribution)
    # ==========================================
    plt.subplot(1, 2, 1)
    sns.kdeplot(agent_pnl, color='blue', label='AI Agent (NSE)', fill=True, alpha=0.3)
    sns.kdeplot(bs_pnl, color='orange', label='Black-Scholes', fill=True, alpha=0.3)
    plt.title('NSE PnL Distribution (Episode Level)', fontsize=14, fontweight='bold')
    plt.xlabel('Profit / Loss ($)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    
    # Adding vertical lines for Mean values
    plt.axvline(agent_pnl.mean(), color='blue', linestyle='dashed', linewidth=1.5)
    plt.axvline(bs_pnl.mean(), color='orange', linestyle='dashed', linewidth=1.5)
    plt.legend()

    # ==========================================
    # Plot 2: Bar Chart (Mean Profit Comparison)
    # ==========================================
    plt.subplot(1, 2, 2)
    means = [agent_pnl.mean(), bs_pnl.mean()]
    bars = plt.bar(['AI Agent\n(NSE)', 'Black-Scholes'], means, color=['blue', 'orange'], width=0.5)
    plt.title('NSE Average Profit (Mean)', fontsize=14, fontweight='bold')
    plt.ylabel('Total Profit ($)', fontsize=12)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (abs(yval)*0.01), f"${round(yval, 2)}", ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    
    save_path = f"results/testing/{model_name}_NSE_Graph.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n🚀 BOOM! NSE Graph save ho gaya hai yahan: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # Strict logic: Agar terminal se naam pass nahi kiya toh seedha error dega
    if len(sys.argv) < 2:
        print("⚠️ Error: Model ka naam pass karna zaroori hai!")
        print("👉 Sahi Command: python plot_nse.py NSE_XXXX")
        sys.exit(1)
        
    model_input = sys.argv[1]
    plot_nse(model_input)