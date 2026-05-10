import numpy as np
import pandas as pd
import random
import include.option_functions as option_functions

# ==========================================
# STRICT ISOLATION: Simulator for NSE V2 Market!
# ==========================================
class Simulator():
    def __init__(self, process, periods_in_day = 1):
        self.process = process
        self.D = periods_in_day

    def set_properties_gbm(self, v, q, mu):
        self.v0 = v
        self.q = q
        self.mu = mu

    def set_properties_heston(self, v0, kappa, theta, sigma, rho, q, r):
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.q = q
        self.r = r
        # Ensure that it forces the custom Heston
        self.process = 'NSE_V2'

    def simulate(self, S0, T = 252, dt = 1/252):
        if self.process == 'GBM':
            self._sim_gbm(S0, self.mu, self.v0, T, dt)
        else:
            # Custom Crash-Proof Log-Euler Heston for NSE V2
            self._sim_heston(S0, self.v0, self.kappa, self.theta, self.sigma, self.rho, self.q, self.r, T, dt)

    def _sim_gbm(self, S0, mu, stdev, T, dt):
        # T idhar steps ke roop me aata hai
        T = int(T)
        self.St = np.zeros(T)
        self.St[0] = S0
        for t in range(1, T):
            self.St[t] = self.St[t-1] * np.exp(mu * dt + stdev * np.sqrt(dt)*np.random.normal())

    def _sim_heston(self, S0, v0, kappa, theta, sigma, rho, q, r, T, dt):
        # INSTITUTIONAL FIX: Log-Euler Maruyama with Full Truncation
        T = int(T)
        self.St = np.zeros(T)
        self.Vt = np.zeros(T)
        
        self.St[0] = S0
        self.Vt[0] = v0

        for t in range(1, T):
            # 1. Generate Correlated Brownian Motions
            Z1 = np.random.normal(0, 1)
            Z2 = np.random.normal(0, 1)
            W1 = Z1
            W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

            # 2. Variance Step (Full Truncation to avoid negative variance)
            v_prev = max(self.Vt[t-1], 0)
            dv = kappa * (theta - v_prev) * dt + sigma * np.sqrt(v_prev * dt) * W2
            self.Vt[t] = max(v_prev + dv, 0.0001)

            # 3. Price Step (Log-Euler Fix - Zero crash guarantee)
            drift = (r - q - 0.5 * v_prev) * dt
            diffusion = np.sqrt(v_prev * dt) * W1
            self.St[t] = self.St[t-1] * np.exp(drift + diffusion)

    def getS(self):
        return self.St
        
    def getV(self):
        # Naya function taaki AI ko Volatility dikhe
        return self.Vt

    def return_set(self, strike_min, strike_max, quote_datetime, min_exp, max_exp, datearray, r):
        # Returns a simulated which looks similar to DataKeeper's sets
        strike = random.uniform(strike_min, strike_max)
        strike = [strike] * len(self.St)
        exp = random.randint(min_exp, max_exp)
        expiration = datearray[datearray.index(quote_datetime) + int(exp)]
        expiration = [expiration] * len(self.St)
        quote_datetimes = []
        i = 0
        while len(quote_datetimes) < len(self.St):
            temp = [datearray[datearray.index(quote_datetime) + int(i)]] * self.D
            quote_datetimes += temp
            i = i + 1
        quote_datetimes = quote_datetimes[:len(self.St)]

        St = self.St / self.St[0]
        Ts = exp - np.arange(0, len(self.St)/(1/self.D), 1/self.D)

        df = pd.DataFrame()
        df['underlying_bid'] = St
        df['expiration'] = expiration
        df['strike'] = strike
        df['quote_datetime'] = quote_datetimes
        df['ticker'] = 'simulated_v2'
        prices = []
        for i in range(len(self.St)):
            if self.process == 'GBM':
                price = option_functions.call_price(St[i], strike[i], r, self.q, self.v0, Ts[i]/252)
            else:
                # Custom Heston approximation for generated paths
                price = option_functions.call_price(St[i], strike[i], r, self.q, np.sqrt(self.Vt[i]), Ts[i]/252)
            prices.append(price)

        df['bid'] = prices
        df['ask'] = prices
        return df