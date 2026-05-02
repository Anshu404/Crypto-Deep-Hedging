import numpy as np
import pandas as pd
import os
import argparse
import torch
from collections import deque

from include.settings import getSettings, setSettings
from include.env_nse import Env
from include.utility import StatePrepare, maybe_make_dirs
from include.transformer_agent import TransformerTD3Agent

def build_window(state_deque, seq_len, state_dim):
    history = list(state_deque)
    if len(history) < seq_len:
        pad = [np.zeros(state_dim, dtype=np.float32)] * (seq_len - len(history))
        history = pad + history
    return np.array(history, dtype=np.float32)

def test_run(env, agent, scaler, state_dim, seq_len, episode_number, empirical):
    cur_state = env.reset(empirical)
    cur_state = scaler.transform(cur_state).flatten()
    
    state_history = deque(maxlen=seq_len)
    state_history.append(cur_state)
    
    infos = []
    stats = {'rewards':np.zeros(1), 'b rewards':np.zeros(1), 'pnl':np.zeros(1), 'b pnl':np.zeros(1)}
    
    done = False
    while not done:
        window = build_window(state_history, seq_len, state_dim)
        
        action_raw = agent.select_action(window)
        action_raw = np.clip(action_raw, -1.0, 1.0)
        action = np.clip(0.5 * (action_raw + 1), 0, 1)
        
        next_state, reward, done, info = env.step(action)
        info['episode'] = episode_number
        info['ticker'] = getattr(env, 'ticker', 'NSE-EQ')
        
        infos.append(info)
        
        next_state = scaler.transform(next_state).flatten()
        state_history.append(next_state)
        
        stats['rewards'][0] += reward
        stats['b rewards'][0] += info['B Reward']
        stats['pnl'][0] += info['A PnL']
        stats['b pnl'][0] += info['B PnL']
        
    return stats, episode_number, pd.DataFrame(infos)

def validate_all_models(base_name):
    print(f"🔍 Scanning and Validating all checkpoints for {base_name}...")
    setSettings('NSE')
    s = getSettings()
    env = Env(s)
    
    is_nse = s['process'] == 'NSE'
    scaler = StatePrepare(env, 1, base_name)
    scaler.load(base_name)
    state_dim = scaler.state_size
    seq_len = 20
    
    agent = TransformerTD3Agent(state_dim=state_dim, d_model=32, seq_len=seq_len)
    
    best_diff = -float('inf')
    best_epoch = 0
    
    # Check standard epochs from 1000 to 20000
    for epoch in range(1000, 21000, 1000):
        model_path = f"model/{base_name}_{epoch}_transformer.pth"
        if not os.path.exists(model_path):
            continue
            
        agent.load(model_path.replace('.pth', '')) 
        
        a_rewards, b_rewards = 0, 0
        val_runs = 50 
        
        for j in range(val_runs):
            stats, _, _ = test_run(env, agent, scaler, state_dim, seq_len, j, False)
            a_rewards += np.sum(stats['rewards'])
            b_rewards += np.sum(stats['b rewards'])
            
        diff = a_rewards - b_rewards
        print(f"Epoch {epoch:5d} | AI Reward: {a_rewards:8.0f} | BS Reward: {b_rewards:8.0f} | Diff: {diff:8.0f}")
        
        if diff > best_diff:
            best_diff = diff
            best_epoch = epoch
            
    print("-" * 50)
    print(f"🏆 BEST MODEL: {base_name}_{best_epoch} with Diff: {best_diff:.0f}")
    print("-" * 50)
    print(f"Now run: python test_transformer.py --test --model {base_name}_{best_epoch}")

def test_load(model_name):
    maybe_make_dirs()
    print(f"\n--- Initializing Test for {model_name} ---") 
    
    base_name = "NSE"
    sim = 1000
    
    setSettings(base_name)
    s = getSettings()
    env = Env(s)
    
    print("✅ Environment Loaded...") 
    
    scaler = StatePrepare(env, 1, base_name)
    scaler.load(base_name)
    state_dim = scaler.state_size
    seq_len = 20
    
    agent = TransformerTD3Agent(state_dim=state_dim, d_model=32, seq_len=seq_len)
    
    clean_name = model_name.replace(".pth", "").replace("_transformer", "")
    full_model_path = f"model/{clean_name}_transformer"
    
    print(f"📂 Loading weights from: {full_model_path}")
    agent.load(full_model_path)
    print("✅ Weights Loaded Successfully!") 
    
    output_filename = f"results/testing/{clean_name}_results.csv"
    
    a_rewards, b_rewards = 0, 0
    all_infos = []
    
    print(f"🚀 Starting Simulation (1000 Episodes)... Please wait.")
    
    for j in range(sim):
        stats, _, t_info = test_run(env, agent, scaler, state_dim, seq_len, j, (s['process'] == 'NSE'))
        a_rewards += np.sum(stats['rewards'])
        b_rewards += np.sum(stats['b rewards'])
        all_infos.append(t_info)
        
        if (j + 1) % 10 == 0:
            print(f"\rProgress: {j+1}/{sim} episodes completed...", end="")
        
    print(f"\n\n✅ Simulation Finished!")
    print(f"AI Total Reward: {a_rewards:.0f} | BS Total Reward: {b_rewards:.0f}")
    
    final_df = pd.concat(all_infos, ignore_index=True)
    final_df.to_csv(output_filename, index=False)
    
    # Final Eval
    result_eval(f"{clean_name}_results")

def result_eval(result_file):
    fn = f"results/testing/{result_file}"
    if not fn.endswith(".csv"):
        fn += ".csv"
        
    if not os.path.exists(fn):
        print(f"❌ Error: CSV file not found at {fn}")
        return

    df = pd.read_csv(fn)
    
    # 🛡️ THE ARRAY-BRACKET CLEANER 🛡️
    core_cols = ['A PnL', 'B PnL', 'A TC', 'B TC']
    for col in core_cols:
        if col in df.columns:
            # PnL ko string banao, '[' aur ']' hatayao, phir number mein convert karo
            clean_str = df[col].astype(str).str.replace(r'[\[\]]', '', regex=True)
            df[col] = pd.to_numeric(clean_str, errors='coerce').fillna(0)
        else:
            print(f"⚠️ DHYAN DE: '{col}' CSV mein missing hai!")

    e = df.groupby('episode')[core_cols].sum()
    
    print("\n" + "="*45)
    print("        📊 FINAL PERFORMANCE STATS 📊")
    print("="*45)

    val = 100 * e[['A PnL', 'B PnL']].mean()
    print('Mean PnL (%) | AI: {:10.4f} | BS: {:10.4f}'.format(*val))
    
    val = 100 * e[['A TC', 'B TC']].mean()
    print('Mean TC (%)  | AI: {:10.4f} | BS: {:10.4f}'.format(*val))
    
    val = 100 * e[['A PnL', 'B PnL']].std()
    print('Risk Std (%) | AI: {:10.4f} | BS: {:10.4f}'.format(*val))
    print("="*45)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--test', action='store_true')
    p.add_argument('--validate', action='store_true')
    p.add_argument('--model', required=True)
    
    args = vars(p.parse_args())
    
    if args['test']:
        test_load(args['model'])
    elif args['validate']:
        validate_all_models(args['model'])