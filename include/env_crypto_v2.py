from include.option_functions import calc_impl_volatility
import include.option_functions as option_functions
import numpy as np
import pandas as pd
import include.data_keeper as data_keeper
import include.simulation as simulation

from datetime import datetime, timedelta
from scipy.stats import norm
from include.settings import getSettings

class Env():
    def __init__(self, s=getSettings()):
        sim_type = 'GBM' if s['process'] == 'CryptoV2' else s['process']
        self.sim = simulation.Simulator(sim_type, periods_in_day=s['D'])
        
        # Core Params
        self.transaction_cost = s['transaction_cost']
        self.kappa = s['kappa'] * 5.0 # Keep the overdrive, but apply it to Welford/Sortino
        self.SIGMA = s['SIGMA']
        self.process = s['process']
        self.q = s['q']
        
        self.r_df = pd.read_csv('data/1yr_treasury.csv')
        self.heston_params = pd.read_csv('data/heston_params.csv')

        # Load and FILTER Bitcoin Data
        if self.process == 'CryptoV2':
            print("🚀 LOADING CRYPTO V2 (QUANT UPGRADED) ENVIRONMENT...")
            df = pd.read_csv('data/btc_live_options.csv')
            df['days_to_exp'] = (pd.to_datetime(df['expiration']) - pd.to_datetime(df['quote_datetime'])).dt.days
            self.crypto_data = df[df['days_to_exp'] > 7].copy()
            if len(self.crypto_data) == 0:
                self.crypto_data = df

        self.D, self.steps = s['D'], s['n_steps']
        
        if self.process == 'Real':
            self.data_keeper = data_keeper.DataKeeper(self.steps)
            
        self.data_set = pd.DataFrame()
        self.t, self.v, self.date_idx = 0, 0.0, 0
        self.option = {}
        self.S = []
        
        # ==========================================
        # V2 UPGRADE: WELFORD VARIANCE TRACKERS
        # ==========================================
        self.running_mean_pnl = 0.0
        self.running_var_pnl = 0.0
        self.step_count_ep = 0
        
        self.b_running_mean_pnl = 0.0
        self.b_running_var_pnl = 0.0

    def get_bs_delta(self):
        d1, _ = option_functions._d(self.option['S/K']*self.K, self.K, self.r, self.q, self.v, self.option['T']/365)
        return norm.cdf(d1)

    def __concat_state(self):
        # PURE 4-STATE (Unchanged for compatibility)
        return np.array([self.option['S/K'], self.option['T']/30, self.stockOwned, self.v])

    def __update_option(self):
        row = self.data_set.loc[self.t, :]

        spot = row['underlying_bid']
        P = 0.5 * (row['bid'] + row['ask'])
        self.expiry = row['expiration'][0:10]
        self.K = float(row['strike'])
        self.S[self.t] = spot
        self.cur_date = row['quote_datetime'][0:10]
        self.ticker = row['ticker']
        self.option['P'] = P

        if self.process == 'CryptoV2':
            self.r = 0.05
        else:
            try:
                self.r = self.r_df.loc[self.r_df['Date'] == self.cur_date, '1y'].iloc[0]
            except:
                self.r = 0.01
                
        ttm = (datetime.strptime(self.expiry, '%Y-%m-%d') - \
               datetime.strptime(self.cur_date, '%Y-%m-%d')).days - (1 - (self.D - self.t%self.D) / self.D)

        self.option['T'] = max(ttm, 0.001)
        self.option['S/K'] = spot / self.K
        
        iv = calc_impl_volatility(spot, self.K, self.r, self.q, self.option['T']/365, P)
        
        if iv:
            self.v = iv
        self.v = max(self.v, 0.01)

    def reset(self, testing=False, start_a=0.0, start_b=0.0):
        self.testing = testing
        self.t = 0
        self.S = np.zeros(self.steps + 1)
        self.stockOwned, self.b_stockOwned = start_a, start_b
        
        # Reset Welford Trackers for new episode
        self.running_mean_pnl = 0.0
        self.running_var_pnl = 0.0
        self.step_count_ep = 0
        self.b_running_mean_pnl = 0.0
        self.b_running_var_pnl = 0.0
        
        new_set = None
        
        if testing and self.process == 'Real':
            self.data_set = self.data_keeper.next_test_set()
        else:
            if self.process == 'Real':
                self.data_set = self.data_keeper.next_train_set()
                
            elif self.process == 'CryptoV2':
                if not self.testing:
                    # ==========================================
                    # V2 UPGRADE: HESTON DOMAIN RANDOMIZATION (Fat Tails)
                    # ==========================================
                    spot = 100.0  
                    strike = 100.0 
                    self.r = 0.05
                    self.q = 0.0
                    
                    # Randomize params to simulate wild BTC regimes
                    vol0 = np.random.uniform(0.40, 1.80)
                    v0 = vol0 ** 2
                    
                    vol_lr = np.random.uniform(0.60, 1.40)
                    theta_h = vol_lr ** 2
                    
                    kappa_h = np.random.uniform(0.5, 4.0)
                    sigma_h = np.random.uniform(0.8, 2.5) # High Vol of Vol (Tails)
                    rho_h = np.random.uniform(-0.85, -0.40)
                    
                    # Stress test Feller condition
                    feller = 2 * kappa_h * theta_h
                    volvol_sq = sigma_h ** 2
                    if feller > volvol_sq * 1.5:
                        sigma_h = np.sqrt(feller / np.random.uniform(0.8, 1.2))
                    
                    self.sim.set_properties_heston(v0, kappa_h, theta_h, sigma_h, rho_h, self.q, self.r)
                    self.sim.simulate(spot, self.steps + 1, 1/(365*self.D))
                    
                    prices = self.sim.getS().copy()
                    variances = self.sim.getV().copy()
                    vols = np.sqrt(np.maximum(variances, 0.0001))
                    
                    df = pd.DataFrame()
                    df['underlying_bid'] = prices
                    df['strike'] = [strike] * len(prices)
                    
                    base_date = datetime.now()
                    dates = [base_date + timedelta(hours=i) for i in range(len(prices))]
                    df['quote_datetime'] = [d.strftime("%Y-%m-%d %H:%M:%S") for d in dates]
                    df['expiration'] = [(base_date + timedelta(days=30)).strftime("%Y-%m-%d")] * len(prices)
                    df['ticker'] = ['BTC-SIM-V2'] * len(prices)
                    
                    opt_prices = []
                    for i in range(len(prices)):
                        days_to_exp = (datetime.strptime(df['expiration'].iloc[i], "%Y-%m-%d") - dates[i]).days
                        ttm_years = max(days_to_exp / 365.0, 0.001)
                        p = option_functions.call_price(prices[i], strike, self.r, self.q, vols[i], ttm_years)
                        opt_prices.append(p)
                    
                    df['bid'] = opt_prices
                    df['ask'] = opt_prices
                    self.data_set = df

                else:
                    real_opt = self.crypto_data.sample(1).iloc[0]
                    spot = real_opt['underlying_bid']
                    strike = real_opt['strike']
                    expiry_str = real_opt['expiration']
                    quote_time_str = real_opt['quote_datetime']
                    self.r = 0.05
                    crypto_vol = 0.60
                    self.sim.set_properties_gbm(crypto_vol, self.q, 0.0)
                    self.sim.simulate(spot, self.steps + 1, 1/(365*self.D))
                    
                    df = pd.DataFrame()
                    df['underlying_bid'] = self.sim.getS()
                    df['expiration'] = [expiry_str] * (self.steps + 1)
                    df['strike'] = [strike] * (self.steps + 1)
                    base_date = datetime.strptime(quote_time_str, "%Y-%m-%d %H:%M:%S")
                    dates = [base_date + timedelta(hours=i) for i in range(self.steps + 1)]
                    df['quote_datetime'] = [d.strftime("%Y-%m-%d %H:%M:%S") for d in dates]
                    df['ticker'] = ['BTC-REAL-TEST'] * (self.steps + 1)
                    prices = []
                    for i in range(self.steps + 1):
                        days_to_exp = (datetime.strptime(expiry_str, "%Y-%m-%d") - dates[i]).days
                        ttm_years = max(days_to_exp / 365.0, 0.001)
                        p = option_functions.call_price(df['underlying_bid'].iloc[i], strike, self.r, self.q, crypto_vol, ttm_years)
                        prices.append(p)
                    df['bid'] = prices
                    df['ask'] = prices
                    self.data_set = df
            else:
                while new_set is None:
                    dates = self.r_df['Date']
                    dates = dates[dates >='2013-01-01']
                    dates = sorted(dates.unique())[:-90]
                    quote_datetime = np.random.choice(dates)

                    try:
                        self.r = self.r_df.loc[self.r_df['Date'] == quote_datetime, '1y'].iloc[0]
                    except:
                        self.r = 0.01
                    if self.process == 'GBM':
                        self.sim.set_properties_gbm(self.SIGMA, self.q, .0)
                        T = self.steps + 1
                        dt = 1/(252*self.D)
                    else:
                        params = self.heston_params[self.heston_params['date'] == quote_datetime]
                        if params.empty:
                            continue

                        v0 = params.iloc[0]['v0']
                        kappa = params.iloc[0]['kappa']
                        theta = params.iloc[0]['theta']
                        sigma = params.iloc[0]['sigma']
                        rho = params.iloc[0]['rho']

                        self.sim.set_properties_heston(v0, kappa, theta, sigma, rho, self.q, self.r)
                        T = 5
                        dt = 35

                        self.sim.simulate(1.0, T, dt)
                    new_set = self.sim.return_set(.85, 1.15, quote_datetime, 15, 90, sorted(self.r_df['Date'].unique()), self.r)
                self.data_set = new_set
        
        self.__update_option()
        return self.__concat_state()

    def step(self, delta):
        b_delta = self.get_bs_delta()
        
        delta_change = abs(-delta - self.stockOwned)
        b_delta_change = abs(-b_delta - self.b_stockOwned)
        
        # ==========================================
        # V2 UPGRADE: WELFORD & SORTINO REWARD FUNC
        # ==========================================
        def welford_reward_func(raw_pnl, d_change, is_agent=True):
            scale_factor = self.S[0] / 1000.0 if self.S[0] > 0 else 1.0
            pnl = (raw_pnl / scale_factor) * 100
            
            if is_agent:
                self.step_count_ep += 1
                n = self.step_count_ep
                diff = pnl - self.running_mean_pnl
                self.running_mean_pnl += diff / n
                diff2 = pnl - self.running_mean_pnl
                self.running_var_pnl += diff * diff2
            else:
                n = self.step_count_ep 
                diff = pnl - self.b_running_mean_pnl
                self.b_running_mean_pnl += diff / max(n, 1)
                diff2 = pnl - self.b_running_mean_pnl
                self.b_running_var_pnl += diff * diff2
            
            # Welford Online Variance Penalty
            incremental_var_penalty = self.kappa * (diff * diff2) if n > 1 else 0.0
            
            # Sortino Asymmetric Penalty (Only punish losses)
            downside_penalty = self.kappa * (min(pnl, 0.0) ** 2)
            
            # Action Smoothing
            smoothness_penalty = 1.5 * (d_change ** 2)
            
            reward = pnl - incremental_var_penalty - downside_penalty - smoothness_penalty
            return reward * 10
            
        infos = {'T':self.option['T'], 'S/K':self.option['S/K']}
        infos['Date'] = self.cur_date
        infos['DateStep'] = self.t % self.D
        
        t_cost = -delta_change * self.S[self.t] * self.transaction_cost
        b_t_cost = -b_delta_change * self.S[self.t] * self.transaction_cost

        opt_old_price = self.option['P']
        self.t += 1
        self.__update_option()
        done = self.t >= self.steps

        opt_new_price = self.option['P']

        pnl = -delta * (self.S[self.t] - self.S[self.t - 1])
        b_pnl = -b_delta * (self.S[self.t] - self.S[self.t - 1])
        
        pnl += (opt_new_price - opt_old_price) + t_cost
        b_pnl += (opt_new_price - opt_old_price) + b_t_cost
        
        self.stockOwned = -delta
        self.b_stockOwned = -b_delta

        reward = welford_reward_func(pnl, delta_change, is_agent=True)
        b_reward = welford_reward_func(b_pnl, b_delta_change, is_agent=False)
        
        infos['B Reward'] = b_reward
        infos['A Reward'] = reward
        infos['A PnL'] = pnl
        infos['B PnL'] = b_pnl
        infos['P0'] = opt_new_price
        infos['P-1'] = opt_old_price
        infos['S0'] = self.S[self.t]
        infos['S-1'] = self.S[self.t - 1]
        infos['A Pos'] = self.stockOwned
        infos['B Pos'] = self.b_stockOwned
        infos['A TC'] = t_cost
        infos['B TC'] = b_t_cost
        infos['A PnL - TC'] = pnl - t_cost
        infos['B PnL - TC'] = b_pnl - b_t_cost
        infos['Expiry'] = self.expiry
        infos['v'] = self.v
        
        return self.__concat_state(), reward, done, infos