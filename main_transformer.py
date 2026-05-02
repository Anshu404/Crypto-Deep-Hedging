import numpy as np
import pandas as pd
import os
import argparse
import torch
from collections import deque

# Load your custom environments and settings
from include.env_nse import Env
from include.utility import StatePrepare, get_model_number, maybe_make_dirs
from include.settings import getSettings, saveSettings, setSettings

# Load the newly forged Transformer parts
from include.transformer_agent import TransformerTD3Agent
from include.buffer_transformer import EfficientSequenceBuffer

def build_window(state_deque, seq_len, state_dim):
    """
    Converts the rolling deque into a strict (seq_len, state_dim) numpy array.
    Zero-pads the beginning if the episode just started and we don't have 20 steps yet.
    """
    history = list(state_deque)
    if len(history) < seq_len:
        pad = [np.zeros(state_dim, dtype=np.float32)] * (seq_len - len(history))
        history = pad + history
    return np.array(history, dtype=np.float32)

def main(model_name=None):
    maybe_make_dirs()
    if model_name is None:
        settings = getSettings()
        process = 'NSE'
        settings['process'] = 'NSE'
        model_name = "Transformer_" + process + str(get_model_number(process))
        saveSettings(model_name)
    else:
        setSettings(model_name)
        settings = getSettings()
        process = 'NSE'
        settings['process'] = 'NSE'
        
    print(f'🚀 Training TRANSFORMER model {model_name} in {process} Mode')
    
    # Hyperparameters
    n_steps = settings['n_steps']
    num_episodes = settings['num_episodes']
    min_noise = settings['min_noise']
    max_noise = settings['max_noise']
    batch_size = settings['batch_size']
    seq_len = 20 # Transformer Memory Window
    
    env = Env(settings)
    start_a, start_b = 0.0, 0.0
    current_noise = max_noise

    # Setup Scaler
    scaler = StatePrepare(env, 1, model_name)
    scaler.save()
    state_dim = scaler.state_size
    
    # Initialize the Transformer Brain and the Optimized Buffer
    # THE SPEED FIX: Reduced d_model to 32
    agent = TransformerTD3Agent(state_dim=state_dim, d_model=32, seq_len=seq_len, max_action=1.0)
    buffer = EfficientSequenceBuffer(capacity=50000, seq_len=seq_len, state_dim=state_dim, device=agent.actor.net[0].weight.device)
    
    stats = {'rewards': np.zeros(num_episodes), 'b rewards': np.zeros(num_episodes), 'pnl': np.zeros(num_episodes), 'b pnl': np.zeros(num_episodes)}
    
    for i in range(num_episodes):
        # 1. NOISE DECAY (Make the agent sober for the last 20% of training)
        decay_steps = num_episodes * 0.8
        if i < decay_steps:
            current_noise = max_noise - ((max_noise - min_noise) * (i / decay_steps))
        else:
            current_noise = min_noise

        done = False
        j = 0
        
        # Reset Environment and Sequence Memory for the new episode
        cur_state = env.reset(False, start_a, start_b)
        cur_state = scaler.transform(cur_state).flatten()
        
        state_history = deque(maxlen=seq_len)
        state_history.append(cur_state)
        
        while not done:
            # Build the temporal window (seq_len, state_dim)
            window = build_window(state_history, seq_len, state_dim)
            
            # Agent looks at the whole window to decide the action
            action_raw = agent.select_action(window)
            
            # Add exploration noise
            if current_noise > 0:
                action_raw += np.random.normal(0, current_noise)
            
            # Bound action between [-1, 1]
            action_raw = np.clip(action_raw, -1.0, 1.0)
            
            # Convert to [0, 1] scale if your env specifically expects 0 to 1
            # Assuming standard Delta hedging where we map [-1, 1] to [0, 1] or use raw.
            # Keeping your original transformation:
            action = np.clip(0.5 * (action_raw + 1), 0, 1)

            # Execute step in Environment
            next_state, reward, done, info = env.step(action)
            next_state = scaler.transform(next_state).flatten()
            
            # Push flat state to buffer (Buffer handles the magic internally)
            buffer.push(cur_state, action_raw, reward, done, is_start=(j==0))
            
            # Update rolling window
            state_history.append(next_state)
            cur_state = next_state
            j += 1
            
            # THE SPEED FIX: Train only every 5 steps
            if len(buffer) > batch_size * 2 and j % 5 == 0:
                for _ in range(2):  # Do 2 updates to compensate
                    agent.train_step(buffer, batch_size=batch_size)
                
            # Log stats
            stats['rewards'][i] += reward
            stats['b rewards'][i] += info['B Reward']
            stats['pnl'][i] += info['A PnL']
            stats['b pnl'][i] += info['B PnL']

        # Print progress every 100 episodes
        if i % 100 == 0 and i >= 100:
            ag = np.mean(stats['rewards'][i - 100:i])
            b = np.mean(stats['b rewards'][i - 100:i])
            print(f"{i+1}/{num_episodes} Last 100: {ag:.4f} vs {b:.4f}, noise: {current_noise:.4f}")

        # Save Checkpoints
        if i % settings['validation_interval'] == 0 and i > 0:
            save_path = f"model/{model_name}_{i}_transformer.pth"
            torch.save({
                'encoder_actor': agent.encoder_actor.state_dict(),
                'actor': agent.actor.state_dict(),
                'encoder_critic': agent.encoder_critic.state_dict(),
                'critic': agent.critic.state_dict(),
            }, save_path)
            print(f"\n[+] Checkpoint saved at {save_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--settings', required=False, default='NSE')
    args = vars(p.parse_args())
    main(args['settings'])