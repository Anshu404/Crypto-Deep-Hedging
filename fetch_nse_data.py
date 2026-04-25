import yfinance as yf
import pandas as pd
from datetime import timedelta
import os

print("Fetching Reliance (RELIANCE.NS) Data from NSE...")
# 1 saal ka data nikal rahe hain Reliance ka
ticker = yf.Ticker("RELIANCE.NS")
hist = ticker.history(period="1y")

options_data = []

for date, row in hist.iterrows():
    spot = row['Close']
    quote_dt = date.strftime('%Y-%m-%d %H:%M:%S')
    
    # 30-day Expiry set kar rahe hain
    exp_dt = (date + timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Round off karke ATM Strike bana rahe hain (e.g., 2943 -> 2940)
    strike = round(spot / 10) * 10 
    
    # Dummy bid/ask dal rahe hain, tera Heston Simulator isko automatically real price se replace kar dega
    dummy_price = spot * 0.05 
    
    options_data.append({
        'quote_datetime': quote_dt,
        'expiration': exp_dt,
        'strike': strike,
        'underlying_bid': spot,
        'bid': dummy_price,
        'ask': dummy_price + 2.0 # 2 rupees ka dummy spread
    })

df = pd.DataFrame(options_data)

# Ensure data folder exists
os.makedirs('data', exist_ok=True)

# Save the file
save_path = 'data/nse_live_options.csv'
df.to_csv(save_path, index=False)
print(f"Success! NSE data saved to {save_path}")
print(f"Total Rows Generated: {len(df)}")