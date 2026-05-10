import numpy as np
import pandas as pd
import os
import argparse
from include.settings import getSettings, setSettings
from include.utility import StatePrepare, maybe_make_dirs
from include.actor_critic import ActorCritic

# ---------------------------------------------
# STRICTLY ISOLATED IMPORT FOR CRYPTO
# ---------------------------------------------
from include.env_crypto import Env

def test_run(env, actor_critic, scaler, state_size, episode_number, empirical):
    cur_state = env.reset(empirical)
    cur_state = scaler.transform(cur_state)
    cur_state = cur_state.reshape((1, state_size))
    
    infos_list = []
    stats = {'rewards':np.zeros(1), 'b rewards':np.zeros(1), 'pnl':np.zeros(1), 'b pnl':np.zeros(1)}

    done = False
    while not done:
        pred_action = actor_critic.act(cur_state)
        action = np.clip( 0.5 * (pred_action[0] + 1), 0, 1) 

        new_state, reward, done, info = env.step(action)
        info['episode'] = episode_number
        info['cr1'] = 0
        info['cr2'] = 0
        info['ticker'] = getattr(env, 'ticker', 'BTC-SIM')
        
        infos_list.append(info)
        
        new_state = scaler.transform(new_state) 
        new_state = new_state.reshape((1, state_size))
        cur_state = new_state
        
        stats['rewards'][0] += reward
        stats['b rewards'][0] += info['B Reward']
        stats['pnl'][0] += info['A PnL']
        stats['b pnl'][0] += info['B PnL']
        
    return stats, episode_number, pd.DataFrame(infos_list)

def test_load(model_name):
    maybe_make_dirs()

    model_name, i = model_name.rsplit('_', 1)
    sim = 1000

    setSettings(model_name)
    s = getSettings()
    env = Env(s)

    empirical = s['process'] == 'Real'

    scaler = StatePrepare(env, 1, model_name)
    scaler.load(model_name)
    state_size = scaler.state_size
    
    actor_critic = ActorCritic(state_size)
    actor_critic.load('model/' + model_name + '_' + i)

    folder = 'results/testing/'
    output_filename = folder + model_name + '_' + i + '.csv'
    
    a_rewards = 0
    b_rewards = 0
    j = 0
    
    all_infos = []
    
    if empirical:
        env.data_keeper.switch_to_test()
        env.data_keeper.reset(soft=False)
        set_count = env.data_keeper.set_count
        while not env.data_keeper.no_more_sets:
            stats, _, t_info = test_run(env, actor_critic, scaler, state_size, j, empirical)
            a_rewards += np.sum(stats['rewards'])
            b_rewards += np.sum(stats['b rewards'])
            print("\rEpisode {}/{}".format(j, set_count), end="")
            all_infos.append(t_info)
            j += 1
    else:
        while j < sim:
            stats, _, t_info = test_run(env, actor_critic, scaler, state_size, j, empirical)
            a_rewards += np.sum(stats['rewards'])
            b_rewards += np.sum(stats['b rewards'])
            print("\rEpisode {}/{}".format(j + 1, sim), end="")
            all_infos.append(t_info)
            j += 1

    if all_infos:
        info_df = pd.concat(all_infos, ignore_index=True)
        info_df.to_csv(output_filename, index=False)
        
    print('\nTesting: {:.0f} vs {:.0f}'.format(a_rewards, b_rewards))

def read_validation_files(model):
    folder = 'results/'
    best = -1000
    best_name = ''
    
    directory = os.fsencode(folder)
    for file in os.listdir(directory):
        fname = os.fsdecode(file)
        if model in fname:
            df = pd.read_csv(folder + fname)
            sums = df.sum(axis=0)
            sums = sums.loc[['A Reward','B Reward']]
            diff = sums.loc['A Reward']-sums.loc['B Reward']
            if diff > best: 
                best = diff
                best_name = fname
            print(fname, ':', diff)

    best_name = best_name.replace('.csv', '')
    print(f'Best Crypto Model: {best_name}, Diff: {best}')
    
def result_eval(model):
    fn = f'results/testing/{model}'
    if '.csv' not in fn:
        fn += '.csv'
        
    df = pd.read_csv(fn, low_memory=False)
    
    # 🛡️ THE BRACKET CLEANER FIX 🛡️
    core_cols = ['A PnL', 'B PnL', 'A TC', 'B TC', 'A Reward', 'B Reward']
    for col in core_cols:
        if col in df.columns:
            clean_str = df[col].astype(str).str.replace(r'[\[\]]', '', regex=True)
            df[col] = pd.to_numeric(clean_str, errors='coerce').fillna(0)
            
    e = df.groupby(['episode'])[core_cols].sum()
    
    print("\n" + "="*45)
    print("      📊 FINAL CRYPTO PERFORMANCE STATS 📊")
    print("="*45)
    val = 100*e[['A PnL', 'B PnL']].mean()
    print('Mean PnL (%) | AI: {:10.4f} | BS: {:10.4f}'.format(*val))
    val = 100*e[['A TC','B TC']].mean()
    print('Mean TC (%)  | AI: {:10.4f} | BS: {:10.4f}'.format(*val))
    val = 100*e[['A PnL', 'B PnL']].std()
    print('Risk Std (%) | AI: {:10.4f} | BS: {:10.4f}'.format(*val))    
    val =  e[['A Reward', 'B Reward']].mean()
    print('Reward       | AI: {:10.2f} | BS: {:10.2f}'.format(*val))
    print("="*45)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    g = p.add_argument_group('mode')
    g.add_argument('--test', action='store_true')
    g.add_argument('--validate', action='store_true')
    g.add_argument('--results', action='store_true')
    p.add_argument('--model', required=True)
    args = vars(p.parse_args())
    
    print(f"🚀 PIPELINE: CRYPTO ISOLATED RUN")
    if args['test']:
        print(f'Testing model {args["model"]}')
        test_load(args['model'])
        result_eval(args['model'])
    elif args['validate']:
        print(f'Validating model {args["model"]}')
        read_validation_files(args['model'])
    elif args['results']:
        print(f'Result eval model {args["model"]}')
        result_eval(args['model'])