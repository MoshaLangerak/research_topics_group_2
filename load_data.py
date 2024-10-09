import pandas as pd
import pickle
import numpy as np

def load_data_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return pd.DataFrame(data)

def load_data_from_csv(file_path):
    return pd.read_csv(file_path, index_col=0)

