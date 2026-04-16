import requests
import pandas as pd
import datetime

def fetch_deribit_options():
    print("Fetching live BTC options data from Deribit API...")
    
    # Deribit API endpoint jo saare active BTC options ka live data deta hai
    url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"
    params = {
        "currency": "BTC",
        "kind": "option"
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if "result" not in data:
        print("Error fetching data!")
        return
        
    options_list = []
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # API se jo data aaya, usko apne AI model ke format mein convert kar rahe hain
    for item in data['result']:
        instrument = item['instrument_name'] # Example: 'BTC-29MAR24-60000-C'
        parts = instrument.split('-')
        
        # Sirf Call options le rahe hain
        if parts[3] == 'C':
            expiry_str = parts[1]
            strike = float(parts[2])
            
            # Formatting Expiry Date
            expiry_date = datetime.datetime.strptime(expiry_str, "%d%b%y").strftime("%Y-%m-%d")
            
            # FIX: Agar API se 'None' aata hai, toh usko 0 maan lenge (using 'or 0')
            bid = item.get('bid_price') or 0
            ask = item.get('ask_price') or 0
            underlying = item.get('estimated_delivery_price') or item.get('underlying_price') or 0
            
            # Agar bid, ask, aur underlying zero se zyada hain, tabhi save karenge
            if bid > 0 and ask > 0 and underlying > 0:
                options_list.append({
                    "quote_datetime": current_time,
                    "expiration": expiry_date,
                    "strike": strike,
                    "underlying_bid": underlying,
                    "bid": bid * underlying, # Convert to USD
                    "ask": ask * underlying, # Convert to USD
                    "ticker": "BTC"
                })
                
    # DataFrame banakar CSV mein save karna
    df = pd.DataFrame(options_list)
    filename = "data/btc_live_options.csv"
    df.to_csv(filename, index=False)
    
    print(f"Success! {len(df)} active BTC Call options saved to {filename}")

if __name__ == "__main__":
    fetch_deribit_options()