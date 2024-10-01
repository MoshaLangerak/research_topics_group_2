from datetime import datetime
import pandas as pd
import yfinance as yf
import time
import os

def get_price_data(stock):
	""" try to query the iex for a stock, if failed note with print """
	ticker = yf.Ticker(stock)
	stock_df = ticker.history(period="5y")
	output_name = 'data/' + stock + '_data.csv'
	stock_df.to_csv(output_name)

def get_info_data(stock, info_to_keep):
	""" try to query the iex for a stock, if failed note with print """
	ticker = yf.Ticker(stock)
	stock_info = ticker.info
	stock_info = {}

	for key in info_to_keep:
		try:
			stock_info[key] = stock_info[key]
		except:
			stock_info[key] = None

	stock_info['ticker'] = stock
	return stock_info

def read_s_and_p(file):
	""" read in s&p data """
	data = pd.read_csv(file)
	return data['Symbol'].values.tolist()

if __name__ == '__main__':
	""" list of S&P companies """
	s_and_p = read_s_and_p('sp500.csv')
	
	info_to_keep = ['country', 'industry', 'sector', 'fullTimeEmployees', 'auditRisk', 
				 'boardRisk', 'compensationRisk', 'overallRisk', 'dividendRate', 'dividendYield', 
				 'marketCap', 'profitMargins', 'totalCash', 'totalCashPerShare', 'ebitda', 
				 'totalDebt', 'totalRevenue', 'revenuePerShare', 'grossMargins']
	
	info_df = pd.DataFrame(columns=['ticker'] + info_to_keep)
	
	start_time = datetime.now()

	for stock in s_and_p:
		if os.path.exists('data/' + stock + '_data.csv'):
			continue

		get_price_data(stock)
		info = get_info_data(stock, info_to_keep)
		info_df = pd.concat([info_df, pd.DataFrame(info, index=[0])], axis=0)

		time.sleep(.5) #so I don't get booted from the server for an abuse of the API

	info_df.to_csv('data/info.csv', index=False)

	#timing:
	finish_time = datetime.now()
	duration = finish_time - start_time
	minutes, seconds = divmod(duration.seconds, 60)
	print(f'The script took {minutes} minutes and {seconds} seconds to run.')
