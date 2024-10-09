import pickle
import pandas as pd
import yfinance as yf
import datetime
from tqdm import tqdm

from load_data import load_data_from_pickle

def get_stock_data(ticker):
    # Fetch the stock data
    stock_df = yf.download(symbol, start=start_date, end=end_date, progress=False)
    
    # Check if the fetched data is empty
    if stock_df.empty:
        return
    
    # Extract the closing prices as a list
    closing_prices = stock_df['Close'].tolist()
    dates = stock_df.index.tolist()
    
    return {'symbol': symbol, 'dates': dates, 'time_series': closing_prices}

if __name__ == "__main__":
    # Load the stock data from the pickle file
    file_path = 'datasets/stock_data_for_emm.pkl'

    stock_data = load_data_from_pickle(file_path)

    # Extract the stock symbols into a list
    stock_symbols = stock_data.index.tolist()[:10]  # [:n]  if you want to run the first n stocks

    # Set the date range for one year starting at Monday 2 October 2023 with the end data exactly 52 weeks later
    start_date = datetime.datetime(2023, 10, 2)
    end_date = start_date + datetime.timedelta(weeks=52)

    # Create a list to store the final data
    final_data = []

    # Loop through each stock symbol and download its data, using a progress bar
    for symbol in tqdm(stock_symbols, desc="Fetching stock data"):
        try:
            data = get_stock_data(symbol)

            # Check if the data is not empty
            if data:
                final_data.append(data)

        except Exception as e:
            continue

    # Convert the final data list into a DataFrame with symbol as index
    result_df = pd.DataFrame(final_data).set_index('symbol')

    # Save the final DataFrame to a CSV file
    result_df.to_csv("datasets/all_stocks_1_year_data.csv")
