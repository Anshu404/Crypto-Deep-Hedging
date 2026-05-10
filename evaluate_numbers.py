import pandas as pd
import numpy as np
import os
import sys

def calculate_hard_numbers(model_name):
    file_path = f"results/testing/{model_name}.csv"
    
    if not os.path.exists(file_path):
        print(f"❌ Error: File {file_path} nahi mili bhai!")
        return

    print(f"📂 Loading data from {file_path}...\n")
    df = pd.read_csv(file_path, low_memory=False)

    # 1. Episode-level par PnL aur Transaction Costs ka sum nikalna (Multiply by 100 for scaling)
    episode_sums = df.groupby('episode').sum() * 100

    if 'A PnL' not in episode_sums.columns or 'B PnL' not in episode_sums.columns:
        print("❌ Error: PnL columns missing hain CSV mein!")
        return

    agent_pnl = episode_sums['A PnL']
    bs_pnl = episode_sums['B PnL']

    # 2. Calculate Expected PnL (Mean)
    agent_mean = agent_pnl.mean()
    bs_mean = bs_pnl.mean()

    # 3. Calculate Risk (Standard Deviation) -> Yahi curve ki width define karta hai
    agent_std = agent_pnl.std()
    bs_std = bs_pnl.std()

    # 4. The Paper's Objective Function -> Mean - (Xi * StdDev)
    # Paper mein Xi (risk-aversion parameter) ko 1, 2, ya 3 rakhte hain. Hum yahan 1 le rahe hain.
    xi = 1.0 
    agent_score = agent_mean - (xi * agent_std)
    bs_score = bs_mean - (xi * bs_std)

    print("="*55)
    print("        📊 THE HARD NUMBERS (AI vs Black-Scholes) 📊")
    print("="*55)
    print(f"1. Mean PnL (Expected Return) | AI: ${agent_mean:7.2f} | BS: ${bs_mean:7.2f}")
    print(f"2. Risk (Standard Deviation)  | AI: {agent_std:8.2f} | BS: {bs_std:8.2f}")
    print("-" * 55)
    print(f"3. Risk-Adjusted Score        | AI: {agent_score:8.2f} | BS: {bs_score:8.2f}")
    print(f"   (Formula: Mean - {xi} * Std Dev)")
    print("="*55)

    # Automated Verdict
    print("\n🔍 THE VERDICT:")
    if agent_std < bs_std:
        print("✅ PROOF: AI ka Standard Deviation kam hai! Isliye graph mein Green curve patla aur lamba tha. AI ne risk successfully kam kiya.")
    else:
        print("❌ AI failed to reduce risk compared to BS.")

    if agent_score > bs_score:
        print("✅ PROOF: Paper ke math ke hisaab se tera AI Black-Scholes ko out-perform kar raha hai (Better Risk-Adjusted Return)!")
    else:
        print("⚠️ AI score is slightly lower. Try increasing Kappa in settings to punish variance even more.")

if __name__ == "__main__":
    # Apne test file ka naam yahan daal de (bina .csv ke)
    model_to_test = "Crypto_19000" 
    
    if len(sys.argv) > 1:
        model_to_test = sys.argv[1]
        
    calculate_hard_numbers(model_to_test)