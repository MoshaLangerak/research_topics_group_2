import pickle
import pandas as pd
import yfinance as yf
import datetime
from tqdm import tqdm


file_path = 'stock_data_for_emm.pkl'

with open(file_path, 'rb') as f:
    stock_data = pickle.load(f)

stock_data = pd.DataFrame(stock_data)

# Extract the stock symbols into a list
stock_symbols = stock_data.index.tolist()  # [:n]  if you want to run the first n stocks

# Set the date range for one year
end_date = datetime.datetime.now().date()
start_date = end_date - datetime.timedelta(days=365)

# Create a list to store the final data
final_data = []

# Loop through each stock symbol and download its data, using a progress bar
for symbol in tqdm(stock_symbols, desc="Fetching stock data"):
    try:
        # Fetch the stock data
        stock_df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        # Check if the fetched data is empty
        if stock_df.empty:
            print(len(final_data))
            continue
        
        # Extract the closing prices as a list
        closing_prices = stock_df['Close'].tolist()
        dates = stock_df.index.tolist()
        
        # Append the stock symbol and its time series to the final data list
        final_data.append({'stocks': symbol, 'dates': dates, 'time_series': closing_prices})

    except Exception as e:
        # Skip the stock in case of any error (e.g., delisted stock)
        continue

# Convert the final data list into a DataFrame
result_df = pd.DataFrame(final_data)

# Save the final DataFrame to a CSV file
result_df.to_csv("data/all_stocks_1year_data.csv", index=False)
