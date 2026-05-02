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
    def __init__(self, s = getSettings()):
        # Check if settings demand NSE, otherwise default to GBM/Real
        sim_type = 'GBM' if s['process'] == 'NSE' else s['process']
        self.sim = simulation.Simulator(sim_type, periods_in_day = s['D'])
        
        # ==========================================
        # RISK MANAGER FIXES INTACT
        # ==========================================
        self.transaction_cost = s['transaction_cost']
        # Force the AI to fear risk by multiplying penalty by 5
        self.kappa = s['kappa'] * 5.0 
        # Strictly set to 2.0 so penalty is based on Variance
        self.reward_exponent = 2.0 
        
        self.SIGMA = s['SIGMA']
        self.process = s['process']
        self.q = s['q']
        self.r_df = pd.read_csv('data/1yr_treasury.csv')
        self.heston_params = pd.read_csv('data/heston_params.csv')

        # Load and FILTER NSE Data
        if self.process == 'NSE':
            print("LOADING NSE (INDIAN EQUITY) LIVE OPTIONS DATA...")
            try:
                df = pd.read_csv('data/nse_live_options.csv')
                # Safety check if dates need conversion
                if 'quote_datetime' in df.columns and 'expiration' in df.columns:
                    df['days_to_exp'] = (pd.to_datetime(df['expiration']) - pd.to_datetime(df['quote_datetime'])).dt.days
                    self.nse_data = df[df['days_to_exp'] > 7].copy()
                    if len(self.nse_data) == 0:
                        self.nse_data = df
                else:
                    self.nse_data = df
            except FileNotFoundError:
                print("WARNING: nse_live_options.csv not found. Please run fetch_nse_data.py first.")
                self.nse_data = pd.DataFrame()

        self.D, self.steps = s['D'], s['n_steps']
        if self.process == 'Real':
            self.data_keeper = data_keeper.DataKeeper(self.steps)
            
        self.data_set = pd.DataFrame()
        self.t, self.v, self.date_idx = 0, 0.0, 0
        self.option = {}
        self.S = []

    def get_bs_delta(self):
        d1, _ = option_functions._d(self.option['S/K']*self.K, self.K, self.r, self.q, self.v, self.option['T']/365)
        return norm.cdf(d1)

    def __concat_state(self):
        # PURE 4-STATE: Moneyness, Time, Inventory, Implied Volatility
        # FIX: Force everything to be a pure Python float to prevent ragged arrays
        sk = float(np.squeeze(self.option['S/K']))
        t_30 = float(np.squeeze(self.option['T'] / 30.0))
        inventory = float(np.squeeze(self.stockOwned))
        vol = float(np.squeeze(self.v))
        
        return np.array([sk, t_30, inventory, vol], dtype=np.float32)
    

    def __update_option(self):
        row = self.data_set.loc[self.t, :]

        # FIX: Ensure these are scalar values, not Pandas Series
        spot = float(np.squeeze(row['underlying_bid']))
        P = float(np.squeeze(0.5 * (row['bid'] + row['ask'])))
        self.expiry = row['expiration'][0:10]
        self.K = float(row['strike'])
        self.S[self.t] = spot
        self.cur_date = row['quote_datetime'][0:10]
        
        # Handle cases where ticker might be missing in dummy data
        self.ticker = row.get('ticker', 'NSE-EQ')
        self.option['P'] = P

        # INDIAN MARKET SPECIFICS
        if self.process == 'NSE':
            self.r = 0.065 # Approx 6.5% Risk Free Rate in India
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

    def reset(self, testing = False, start_a = 0.0, start_b = 0.0):
        self.testing = testing
        self.t = 0
        self.S = np.zeros(self.steps + 1)
        self.stockOwned, self.b_stockOwned = start_a, start_b
        new_set = None
        
        if testing and self.process == 'Real':
            self.data_set = self.data_keeper.next_test_set()
        else:
            if self.process == 'Real':
                self.data_set = self.data_keeper.next_train_set()
                
            elif self.process == 'NSE':
                if not self.testing:
                    # ==========================================
                    # NSE SIMULATOR (INDIAN EQUITY HESTON)
                    # ==========================================
                    spot = 2900.0  # e.g., Reliance Stock Price
                    strike = 2900.0 
                    self.r = 0.065 # Indian Risk Free Rate
                    self.q = 0.0
                    
                    # Indian Equity Volatility Params
                    v0 = 0.20 ** 2      # 20% initial volatility
                    kappa_h = 3.0       # Faster mean reversion
                    theta_h = 0.22 ** 2 # Long-term vol slightly higher
                    sigma_h = 0.30      # Vol-of-vol (less than crypto)
                    rho_h = -0.60       # Negative correlation
                    
                    self.sim.set_properties_heston(v0, kappa_h, theta_h, sigma_h, rho_h, self.q, self.r)
                    
                    # IMPORTANT: Indian Markets use 252 days
                    self.sim.simulate(spot, self.steps + 1, 1/(252*self.D))
                    
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
                    df['ticker'] = ['NSE-SIM'] * len(prices)
                    
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
                    # REAL WORLD TESTING BLOCK (NSE)
                    if self.nse_data.empty:
                        raise ValueError("NSE data is empty. Cannot run tests.")
                        
                    real_opt = self.nse_data.sample(1).iloc[0]
                    spot = real_opt['underlying_bid']
                    strike = real_opt['strike']
                    expiry_str = real_opt['expiration']
                    quote_time_str = real_opt['quote_datetime']
                    
                    self.r = 0.065
                    nse_vol = 0.20 # Base vol for Indian equity
                    
                    self.sim.set_properties_gbm(nse_vol, self.q, 0.0)
                    self.sim.simulate(spot, self.steps + 1, 1/(252*self.D)) # 252 Days
                    
                    df = pd.DataFrame()
                    df['underlying_bid'] = self.sim.getS()
                    df['expiration'] = [expiry_str] * (self.steps + 1)
                    df['strike'] = [strike] * (self.steps + 1)
                    
                    base_date = datetime.strptime(quote_time_str, "%Y-%m-%d %H:%M:%S")
                    dates = [base_date + timedelta(hours=i) for i in range(self.steps + 1)]
                    df['quote_datetime'] = [d.strftime("%Y-%m-%d %H:%M:%S") for d in dates]
                    df['ticker'] = ['NSE-REAL-TEST'] * (self.steps + 1)
                    
                    prices = []
                    for i in range(self.steps + 1):
                        days_to_exp = (datetime.strptime(expiry_str, "%Y-%m-%d") - dates[i]).days
                        ttm_years = max(days_to_exp / 365.0, 0.001)
                        p = option_functions.call_price(df['underlying_bid'].iloc[i], strike, self.r, self.q, nse_vol, ttm_years)
                        prices.append(p)
                    df['bid'] = prices
                    df['ask'] = prices
                    
                    # THE FIX: Forcefully reset the index and drop the old one
                    self.data_set = df.reset_index(drop=True)
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
        # ACTION SMOOTHING
        # ==========================================
        # ==========================================
        # UPGRADE 3: STRICT PNL NORMALIZATION (The Quant Fix)
        # ==========================================
        # ==========================================
        # UPGRADE 4: THE TRUE QUANT OBJECTIVE (Domain Shift Fixed, Double Penalty Removed)
        # ==========================================
        def reward_func(raw_pnl, d_change):
            pct_pnl = (raw_pnl / self.S[0]) * 100.0
            
            # PAPER UPGRADE: Downside Risk (Semi-Variance) Only!
            if pct_pnl < 0:
                # Agar loss hua, toh hardcore penalty lagao
                risk_penalty = self.kappa * (abs(pct_pnl) ** self.reward_exponent)
            else:
                # Agar profit hua, toh koi risk penalty nahi! Let winners run.
                risk_penalty = 0.0 
            
            reward = pct_pnl - risk_penalty
            return reward * 10.0
            
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

        reward = reward_func(pnl, delta_change)
        b_reward = reward_func(b_pnl, b_delta_change)
        
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