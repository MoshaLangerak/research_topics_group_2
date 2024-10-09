import pandas as pd
import pickle
import numpy as np
from scipy.stats import linregress
def convert_to_percentage_growth(price_values):
    """
    Converts a list of price values to a percentage growth dataset.
    Parameters:
        price_values (list or pd.Series): List of stock prices.
    Returns:
        pd.DataFrame: DataFrame containing day numbers, prices, and percentage growth.
    """
    # Create a DataFrame
    df = pd.DataFrame({'price': price_values})
    df['day'] = range(1, len(df) + 1)

    # Compute the difference between consecutive days
    df['delta'] = df['price'].diff()

    # Calculate the percentage growth
    df['percentage_growth'] = (df['delta'] / df['price'].shift(1)) * 100

    # Replace NaN for the first day with 0% using .loc
    df.loc[0, 'percentage_growth'] = 0.0

    # Round the percentage growth to two decimal places
    df['percentage_growth'] = df['percentage_growth'].round(2)

    return df[['day', 'price', 'percentage_growth']]



def make_growth_target_df(file_path):
    """
    Loads stock data from a pickle file and calculates the 'growth_target' for each entry, returning a DataFrame.
    Parameters:
    file_path (str): The file path to the pickle file containing the stock data.
    Returns:
    pd.DataFrame: A DataFrame where each row contains stock data along with the 'growth_target',
                  which represents the percentage growth over time for each stock's target values.
    """

    # Open and load the pickle file containing stock data
    with open(file_path, 'rb') as f:
        stock_data_file = pickle.load(f)

    # Convert the loaded data into a pandas DataFrame
    stock_data = pd.DataFrame(stock_data_file)

    # Calculate the 'growth_target' for each row by applying the 'convert_to_percentage_growth' function
    # to the 'target' column and extract the 'percentage_growth' part for each row
    stock_data['growth_target'] = [list(convert_to_percentage_growth(i)['percentage_growth']) for i in stock_data['target']]

    # Replace infinite values with NaN
    stock_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Extract the growth target column
    growth_data = stock_data.loc[:, "growth_target"]
    # Get the indices of rows with NaN or Inf values
    nan_indices = growth_data[growth_data.apply(lambda x: any(pd.isna(i) for i in x))].index.tolist()
    inf_indices = growth_data[growth_data.apply(lambda x: any(i in [np.inf, -np.inf] for i in x))].index.tolist()
    # Get the union of the two indices
    union = list(set(nan_indices).union(set(inf_indices)))
    # Drop the rows with NaN or Inf values and reset the index
    stock_data = stock_data.drop(union).reset_index(drop=True)
    stock_data.drop(['target'], axis=1, inplace=True)
    stock_data.dropna(inplace=True)
    stock_data.reset_index(drop=True, inplace=True)
    return stock_data