import pandas as pd
import pickle
import numpy as np

def load_data_from_pickle(file_path):
    """
    Load data from a pickle file.

    Parameters:
        file_path (str): The file path to the pickle file.

    Returns:
        pd.DataFrame: The data loaded from the pickle file.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return pd.DataFrame(data)

def load_data_from_csv(file_path):
    """
    Load data from a CSV file.

    Parameters:
        file_path (str): The file path to the CSV file.

    Returns:
        pd.DataFrame: The data loaded from the CSV file.
    """
    
    return pd.read_csv(file_path, index_col=0)

