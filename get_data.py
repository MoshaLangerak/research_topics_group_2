from datetime import datetime
import pandas as pd
from pandas import DataFrame
import yfinance as yf
import time
import bs4 as bs
import pickle
import requests

def download_stock(stock):
	""" try to query the iex for a stock, if failed note with print """
	try:
		ticker = yf.Ticker(stock)
		stock_df = ticker.history(period="5y")
		output_name = 'data/' + stock + '_data.csv'
		stock_df.to_csv(output_name)

	except Exception as e:
		print(e)
		bad_names.append(stock)
		print('bad: %s' % (stock))

def read_s_and_p(file):
	""" read in s&p data """
	data = pd.read_csv(file)
	return data['Symbol'].values.tolist()

if __name__ == '__main__':
	""" list of S&P companies """
	s_and_p = read_s_and_p('sp500.csv')
		
	start_time = datetime.now()
	bad_names =[] #to keep track of failed queries

	for stock in s_and_p:
		download_stock(stock)
		time.sleep(.5) #so I don't get booted from the server for an abuse of the API
	
	# Save failed queries to a text file to retry
	if len(bad_names) > 0:
		with open('failed_queries.txt','w') as outfile:
			for name in bad_names:
				outfile.write(name+'\n')

	#timing:
	finish_time = datetime.now()
	duration = finish_time - start_time
	minutes, seconds = divmod(duration.seconds, 60)
	print(f'The threaded script took {minutes} minutes and {seconds} seconds to run.')
