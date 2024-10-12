import pandas as pd
import numpy as np
import ast

from load_data import load_data_from_pickle, load_data_from_csv
from features_time_series_generator import *

def join_dfs(stock_data, time_series_data):
    result = stock_data.join(time_series_data, how='inner')
    result.rename(columns={'time_series': 'target'}, inplace=True)
    # result['target'] = result['target'].apply(ast.literal_eval)
    
    return result

def merge_dicts(d1, d2):
    for key, value in d2.items():
        if key in d1:
            # If the value is already a list, append the new value
            if isinstance(d1[key], list):
                d1[key].append(value)
            else:
                # If it's not a list, convert it to a list with the existing and new value
                d1[key] = [d1[key], value]
        else:
            # If the key is not in the dictionary, simply add it
            d1[key] = value
    return d1

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

def make_growth_target_df(stock_data):
    """
    Loads stock data from a pickle file and calculates the 'growth_target' for each entry, returning a DataFrame.

    Parameters:
    file_path (str): The file path to the pickle file containing the stock data.

    Returns:
    pd.DataFrame: A DataFrame where each row contains stock data along with the 'growth_target',
                  which represents the percentage growth over time for each stock's target values.
    """

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
    stock_data = stock_data.drop(union).reset_index().rename(columns={"index": "symbol"})
    stock_data.dropna(inplace=True)

    return stock_data

def compute_features_for_df(df):
    new_features = {}

    for i, row in df.iterrows():
        time_series = row['target']
        date_series = row['dates']
        features_timeseries,_,_ = compute_all_features_for_timeseries(time_series, date_series, True)
        new_features = merge_dicts(new_features, features_timeseries)

    return new_features

def fix_time_series_missing_data(df, time_series_col, date_col, allowed_missing=60):
    time_series = [eval(ts) for ts in df[time_series_col].values]
    dates = [eval(date, {"Timestamp": pd.Timestamp}) for date in df[date_col].values]

    # create a set of all the dates
    all_dates = set()

    for date in dates:
        all_dates.update(date)

    # for each time_serie check if it has all the dates, if not add the missing dates and impute the value
    time_series_to_remove = []
    missing_dates_amounts = []
    for i, time_serie in enumerate(time_series):
        missing_dates = all_dates - set(dates[i])
        missing_dates_amounts.append(len(missing_dates))

        if len(missing_dates) > allowed_missing:
            time_series_to_remove.append(df.index[i])
            continue

        for date in missing_dates:
            # check if the date is the first or the last
            if date < dates[i][0]:
                time_serie.insert(0, time_serie[0])
                dates[i].insert(0, date)
                continue
            if date > dates[i][-1]:
                time_serie.append(time_serie[-1])
                dates[i].append(date)
                continue

            # find the index of the previous date
            previous_date = max([d for d in dates[i] if d < date])
            previous_index = dates[i].index(previous_date)

            # find the index of the next date
            next_date = min([d for d in dates[i] if d > date])
            next_index = dates[i].index(next_date)

            # impute the missing date
            time_serie.insert(previous_index + 1, (time_series[i][previous_index] + time_series[i][next_index]) / 2)

            # add the missing date
            dates[i].insert(previous_index + 1, date)
    
    # update the dataframe, first add the time_series (since old ones are still there) and then remove the rows
    df[time_series_col] = time_series
    df[date_col] = dates
    df = df.drop(time_series_to_remove)

    # print the mean amount of missing dates
    print(np.mean(missing_dates_amounts))

    return df, time_series, dates


if __name__ == "__main__":
    stock_data = load_data_from_pickle('datasets/stock_data_for_emm.pkl')

    stock_data.drop('target', axis=1, inplace=True)

    time_series_data = load_data_from_csv('datasets/stocks_time_series_data.csv')
    time_series_data = time_series_data.dropna(subset=['time_series']) # drop rows with missing time_series

    time_series_data,_ ,_ = fix_time_series_missing_data(time_series_data, 'time_series', 'dates')

    stock_data = join_dfs(stock_data, time_series_data)

    stock_data_all_features = stock_data.assign(**compute_features_for_df(stock_data))

    stock_data_all_features = make_growth_target_df(stock_data_all_features)

    # stock_data_all_features.to_pickle('datasets/stock_data_all_features.pkl')
    stock_data_all_features.to_csv('datasets/stock_data_all_features.csv', index=False)

    