from unittest.mock import inplace

import pandas as pd
import numpy as np
import ast

from load_data import load_data_from_pickle, load_data_from_csv
from features_time_series_generator import *

def join_dfs(stock_data, time_series_data):
    result = stock_data.join(time_series_data, how='inner')
    result.rename(columns={'time_series': 'target'}, inplace=True)
    result['target'] = result['target'].apply(ast.literal_eval)
    
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

def compute_features(target_series):
    new_features = {}

    for time_series in target_series:
        features_timeseries,_,_ = compute_all_features_for_timeseries(time_series, True)
        new_features = merge_dicts(new_features, features_timeseries)

    return new_features

if __name__ == "__main__":
    stock_data = load_data_from_pickle('datasets/stock_data_for_emm.pkl')

    stock_data.drop('target', axis=1, inplace=True)

    time_series_data = load_data_from_csv('datasets/all_stocks_1_year_data.csv')
    time_series_data = time_series_data.dropna(subset=['time_series'])

    stock_data = join_dfs(stock_data, time_series_data)

    stock_data_all_features = stock_data.assign(**compute_features(stock_data['target']))

    stock_data_all_features.to_pickle('datasets/stock_data_all_features.pkl')
    # stock_data_all_features.to_csv('datasets/stock_data_all_features.csv', index=True)

    
