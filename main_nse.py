import numpy as np
import pandas as pd
from os import path
import argparse

# ==========================================
# STRICT ISOLATION: Load the Indian Equity Environment & Testing
# ==========================================
from include.env_nse import Env
from testing_nse import test_run  # <-- Changed to testing_nse to keep pipeline pure

from include.actor_critic import ActorCritic
from include.utility import StatePrepare, get_model_number, maybe_make_dirs
from include.settings import getSettings, saveSettings, setSettings


def main(model_name = None):
    maybe_make_dirs()
    if model_name is None:
        settings = getSettings()
        # Ensure we are strictly using the NSE process
        process = 'NSE' 
        settings['process'] = 'NSE'
        model_name = process + str(get_model_number(process))
        saveSettings(model_name)
    else:
        setSettings(model_name)
        settings = getSettings()
        process = 'NSE'
        settings['process'] = 'NSE'
        
    print(f'Training model {model_name} in NSE (Indian Equity) Mode')
    
    n_steps = settings['n_steps']
    showcase_every = settings['showcase_every']
    min_noise = settings['min_noise']
    max_noise = settings['max_noise']
    noise_reward_dividor = settings['noise_reward_dividor']

    # Initialize the NSE Environment
    env = Env(settings)

    # agent's and black-scholes hedge's starting positions in the underlying
    start_a, start_b = 0.0, 0.0
    validation_diff = -100000
    validation_n = 0
    current_noise = 1.0

    # feature scaling, save for later use
    scaler = StatePrepare(env, 1, model_name)
    scaler.save()
    state_size = scaler.state_size
    
    actor_critic = ActorCritic(state_size)
    actor_critic.forget()
    
    num_episodes = settings['num_episodes']
    
    show_example = False
    
    stats = {'rewards':np.zeros(num_episodes), 'b rewards':np.zeros(num_episodes), 'pnl':np.zeros(num_episodes), 'b pnl':np.zeros(num_episodes)}
    
    for i in range(num_episodes):
        j = 0
        done = False        

        cur_state = env.reset(False, start_a, start_b)
        cur_state = scaler.transform(cur_state).reshape((1, state_size))
        
        while not done:
            if i % showcase_every == 0:
                show_example = True
            if show_example:
                pred_action = actor_critic.act(cur_state)
                action = np.clip( 0.5 * (pred_action[0] + 1), 0, 1)
                bs_delta = env.get_bs_delta()
                
                if not j: print('\nAgent |  BS  |  Diff')
                print('{:5.2f} |{:5.2f} | {:5.2f}'.format(action, bs_delta, action - bs_delta))
            else:
                noise = [np.random.normal(0, current_noise)]
                pred_action = actor_critic.act(cur_state) + noise
                pred_action = np.clip(pred_action, -1, 1)
                action = np.clip( 0.5 * (pred_action[0] + 1), 0, 1)

            new_state, reward, done, info = env.step(action)            
            new_state = scaler.transform(new_state)         
            new_state = new_state.reshape((1, state_size))
            actor_critic.remember(cur_state, pred_action, reward, new_state, done)
            
            cur_state = new_state
            
            j += 1
            if j == n_steps:
                show_example = False
                
            stats['rewards'][i] += reward
            stats['b rewards'][i] += info['B Reward']
            stats['pnl'][i] += info['A PnL']
            stats['b pnl'][i] += info['B PnL']

        actor_critic.train()

        # ==========================================
        # UPGRADE: EXPLORATION NOISE DECAY (Make the Agent Sober)
        # ==========================================
        if i % 100 == 0 and i >= 100:
            ag = np.mean(stats['rewards'][i - 100:i])
            b = np.mean(stats['b rewards'][i - 100:i])
            
            # Linear decay over 80% of the training episodes
            decay_steps = num_episodes * 0.8
            if i < decay_steps:
                # Dheere dheere noise kam karo
                current_noise = max_noise - ((max_noise - min_noise) * (i / decay_steps))
            else:
                # Aakhri 20% training mein Agent ekdum SOBER (0 noise) hona chahiye!
                current_noise = min_noise
                
            print(f"{i+1}/{num_episodes} Last 100: {ag:.4f} vs {b:.4f}, noise: {current_noise:.4f}")
        
        if i % settings['validation_interval'] == 0 and i > 0:
            set_count = settings['sim_test_runs']
            
            actor_critic.save('model/' + model_name + '_' + str(i))
            
            a_rewards = 0
            b_rewards = 0
            
            info_df = None
            k = 0
            while k < set_count:
                test_stats, _, t_info = test_run(env, actor_critic, scaler, state_size, k, False)
                a_rewards += np.sum(test_stats['rewards'])
                b_rewards += np.sum(test_stats['b rewards'])
                print("\rValidating Episode {}/{}".format(k + 1, set_count), end="")
                
                if info_df is None:
                    info_df = pd.DataFrame(t_info)
                else:
                    # Fix for pandas warning about append
                    info_df = pd.concat([info_df, pd.DataFrame(t_info)], ignore_index=True)
                
                k += 1
            
            info_df.to_csv('results/' + model_name + '_' + str(i) + '.csv')
            diff = a_rewards - b_rewards
            
            print('\nValidation: {:.0f} vs {:.0f}'.format(a_rewards, b_rewards))
            
            if diff > validation_diff:
                validation_diff = diff
                validation_n = 0
            else:
                validation_n += 1
            
            if validation_n >= settings['validation_limit']:
                break

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    g = p.add_argument_group('mode')
    # Default to loading the NSE settings JSON
    p.add_argument('--settings', required=False, default='NSE')
    args = vars(p.parse_args())
    main(args['settings'])