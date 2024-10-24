import pandas as pd
from scipy.stats import linregress

def convert_to_percentage_growth(df):
    """
    Converts a DataFrame with value data to include percentage growth.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'window_index' and 'value' columns.

    Returns:
        pd.DataFrame: DataFrame containing 'window_index', 'value', and 'percentage_growth'.
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Compute the difference between consecutive windows
    df['delta'] = df['value'].diff()

    # Calculate the percentage growth
    df['percentage_growth'] = (df['delta'] / df['value'].shift(1)) * 100

    # Replace NaN for the first window_index with 0% using .loc
    df.loc[df.index[0], 'percentage_growth'] = 0.0

    # Round the percentage growth to two decimal places
    df['percentage_growth'] = df['percentage_growth'].round(5)

    return df[['window_index', 'value', 'percentage_growth', 'date']]

def compute_extra_statistics_og_ts(df):
    # Compute delta
    df['delta'] = df['value'].diff()

    # Define direction
    df['direction'] = df['delta'].apply(
        lambda x: 'increase' if x > 0 else ('decrease' if x < 0 else 'no_change')
    )

    # ---------------------------
    # Longest and Biggest Increase
    # ---------------------------

    # Mask for increasing windows
    df['is_increase'] = df['delta'] > 0

    # Identify streaks by cumulatively summing when the streak breaks
    df['increase_group'] = (df['is_increase'] != df['is_increase'].shift()).cumsum()

    # Filter only increase groups
    increase_streaks = df[df['is_increase']].groupby('increase_group')

    # Compute the length, sum, start_window_index, and end_window_index of each increase streak
    if increase_streaks.ngroups > 0:
        increase_stats = increase_streaks.agg(
            count=('delta', 'count'),
            sum=('delta', 'sum'),
            start_window_index=('window_index', 'min'),
            end_window_index=('window_index', 'max')
        )

        # Longest Continuous Increase
        longest_incr_idx = increase_stats['count'].idxmax()
        longest_continuous_increase = {
            "length": int(increase_stats.loc[longest_incr_idx, 'count']),
            "start_window_index": int(increase_stats.loc[longest_incr_idx, 'start_window_index']),
            "end_window_index": int(increase_stats.loc[longest_incr_idx, 'end_window_index'])
        }

        # Biggest Continuous Increase
        biggest_incr_idx = increase_stats['sum'].idxmax()
        biggest_continuous_increase = {
            "sum": round(increase_stats.loc[biggest_incr_idx, 'sum'], 5),
            "start_window_index": int(increase_stats.loc[biggest_incr_idx, 'start_window_index']),
            "end_window_index": int(increase_stats.loc[biggest_incr_idx, 'end_window_index'])
        }
    else:
        longest_continuous_increase = {
            "length": 0,
            "start_window_index": None,
            "end_window_index": None
        }
        biggest_continuous_increase = {
            "sum": 0.0,
            "start_window_index": None,
            "end_window_index": None
        }

    # ---------------------------
    # Longest and Biggest Decrease
    # ---------------------------

    # Mask for decreasing windows
    df['is_decrease'] = df['delta'] < 0

    # Identify streaks by cumulatively summing when the streak breaks
    df['decrease_group'] = (df['is_decrease'] != df['is_decrease'].shift()).cumsum()

    # Filter only decrease groups
    decrease_streaks = df[df['is_decrease']].groupby('decrease_group')

    # Compute the length, sum, start_window_index, and end_window_index of each decrease streak
    if decrease_streaks.ngroups > 0:
        decrease_stats = decrease_streaks.agg(
            count=('delta', 'count'),
            sum=('delta', 'sum'),
            start_window_index=('window_index', 'min'),
            end_window_index=('window_index', 'max')
        )

        # Longest Continuous Decrease
        longest_decr_idx = decrease_stats['count'].idxmax()
        longest_continuous_decrease = {
            "length": int(decrease_stats.loc[longest_decr_idx, 'count']),
            "start_window_index": int(decrease_stats.loc[longest_decr_idx, 'start_window_index']),
            "end_window_index": int(decrease_stats.loc[longest_decr_idx, 'end_window_index'])
        }

        # Biggest Continuous Decrease (use absolute value)
        biggest_decr_idx = decrease_stats['sum'].idxmin()  # Most negative sum
        biggest_continuous_decrease = {
            "sum": round(abs(decrease_stats.loc[biggest_decr_idx, 'sum']), 5) * -1,
            "start_window_index": int(decrease_stats.loc[biggest_decr_idx, 'start_window_index']),
            "end_window_index": int(decrease_stats.loc[biggest_decr_idx, 'end_window_index'])
        }
    else:
        longest_continuous_decrease = {
            "length": 0,
            "start_window_index": None,
            "end_window_index": None
        }
        biggest_continuous_decrease = {
            "sum": 0.0,
            "start_window_index": None,
            "end_window_index": None
        }

    # Compile the results into a dictionary
    statistics = {
        "longest_continuous_increase": longest_continuous_increase,
        "biggest_continuous_increase": biggest_continuous_increase,
        "longest_continuous_decrease": longest_continuous_decrease,
        "biggest_continuous_decrease": biggest_continuous_decrease
    }

    return statistics

def compute_simple_features_og_ts(df):
    features = {}

    # Average value
    features['mean'] = round(df['value'].mean(), 5)
    # Median value
    features['median'] = round(df['value'].median(), 5)
    # Volatility in time series
    features['std_dev'] = round(df['value'].std(), 5)
    # Autocorrelation 1-lag: The correlation of the time series with its own lagged version by one time step.
    features['autocorr_lag1'] = round(df['value'].autocorr(lag=1), 5)
    # Min value
    features["min"] = round(df['value'].min(), 5)
    # Max value
    features["max"] = round(df['value'].max(), 5)
    # Value range
    features["range"] = round(features["max"] - features["min"], 5)
    # window_max
    features['window_min'] = df['window_index'][df['value'].idxmin()]
    # window_min
    features['window_max'] = df['window_index'][df['value'].idxmax()]
    # Trend slope
    slope, intercept, r_value, p_value, std_err = linregress(df['window_index'], df['value'])
    features['trend_slope'] = round(slope, 5)

    return features, df, slope, intercept

def compute_simple_features_growth_perc_ts(df):
    features = {}

    # Average value
    features['mean'] = round(df['percentage_growth'].mean(), 5)
    # Median value
    features['median'] = round(df['percentage_growth'].median(), 5)
    # Volatility in time series
    features['std_dev'] = round(df['percentage_growth'].std(), 5)
    # Autocorrelation 1-lag: The correlation of the time series with its own lagged version by one time step.
    features['autocorr_lag1'] = round(df['value'].autocorr(lag=1), 5)
    # Min value
    features["min"] = round(df['percentage_growth'].min(), 5)
    # Max value
    features["max"] = round(df['percentage_growth'].max(), 5)
    # Value range
    features["range"] = round(features["max"] - features["min"], 5)
    # window_max
    features['window_min'] = df['window_index'][df['percentage_growth'].idxmin()]
    # window_min
    features['window_max'] = df['window_index'][df['percentage_growth'].idxmax()]
    # Trend slope
    slope, intercept, r_value, p_value, std_err = linregress(df['window_index'], df['percentage_growth'])
    features['trend_slope'] = round(slope, 5)

    # feature for best day of the week
    df['day_of_week'] = df['date'].dt.dayofweek
    features['best_day_of_week'] = df.groupby('day_of_week')['value'].mean().idxmax()

    # feature for best month of the year
    df['month'] = df['date'].dt.month
    features['best_month'] = df.groupby('month')['value'].mean().idxmax()

    # feature for best quarter of the year
    df['quarter'] = df['date'].dt.quarter
    features['best_quarter'] = df.groupby('quarter')['value'].mean().idxmax()

    # feature for best day of the month
    df['day_of_month'] = df['date'].dt.day
    features['best_day_of_month'] = df.groupby('day_of_month')['value'].mean().idxmax()

    # feature for best week of year
    df['week_of_month'] = df['date'].dt.isocalendar().week
    features['best_week_of_year'] = df.groupby('week_of_month')['value'].mean().idxmax()

    df.drop(columns=['day_of_week', 'month', 'quarter', 'day_of_month', 'week_of_month'], inplace=True)

    return features, df, slope, intercept

def compute_extra_statistics_growth_perc_ts(df):
    """
    Computes the biggest continuous increase and decrease in percentage growth.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'window_index' and 'percentage_growth'.

    Returns:
        dict: Dictionary containing statistics for the biggest continuous increase and decrease.
    """

    # Ensure required columns exist
    required_columns = {'window_index', 'percentage_growth'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"The DataFrame must contain the columns: {required_columns}")

    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()

    # ---------------------------
    # Biggest Continuous Increase Percentage
    # ---------------------------

    # Mask for windows with positive percentage growth
    df['is_increase'] = df['percentage_growth'] > 0

    # Identify increase streaks by cumulatively summing when the streak breaks
    df['increase_group'] = (df['is_increase'] != df['is_increase'].shift()).cumsum()

    # Filter only increase groups
    increase_streaks = df[df['is_increase']].groupby('increase_group')

    # Check if there are any increase streaks
    if increase_streaks.ngroups > 0:
        # Aggregate to find sum_percentage_growth, start_window_index, end_window_index for each streak
        increase_stats = increase_streaks.agg(
            sum_percentage_growth=('percentage_growth', 'sum'),
            start_window_index=('window_index', 'min'),
            end_window_index=('window_index', 'max')
        )

        # Find the streak with the maximum sum_percentage_growth
        biggest_incr_idx = increase_stats['sum_percentage_growth'].idxmax()
        biggest_continuous_increase_perc = {
            "sum_percentage_growth": round(increase_stats.loc[biggest_incr_idx, 'sum_percentage_growth'], 5),
            "start_window_index": int(increase_stats.loc[biggest_incr_idx, 'start_window_index']),
            "end_window_index": int(increase_stats.loc[biggest_incr_idx, 'end_window_index'])
        }
    else:
        biggest_continuous_increase_perc = {
            "sum_percentage_growth": 0.0,
            "start_window_index": None,
            "end_window_index": None
        }

    # ---------------------------
    # Biggest Continuous Decrease Percentage
    # ---------------------------

    # Mask for windows with negative percentage growth
    df['is_decrease'] = df['percentage_growth'] < 0

    # Identify decrease streaks by cumulatively summing when the streak breaks
    df['decrease_group'] = (df['is_decrease'] != df['is_decrease'].shift()).cumsum()

    # Filter only decrease groups
    decrease_streaks = df[df['is_decrease']].groupby('decrease_group')

    # Check if there are any decrease streaks
    if decrease_streaks.ngroups > 0:
        # Aggregate to find sum_percentage_growth, start_window_index, end_window_index for each streak
        decrease_stats = decrease_streaks.agg(
            sum_percentage_growth=('percentage_growth', 'sum'),
            start_window_index=('window_index', 'min'),
            end_window_index=('window_index', 'max')
        )

        # Find the streak with the minimum sum_percentage_growth (most negative)
        biggest_decr_idx = decrease_stats['sum_percentage_growth'].idxmin()
        biggest_continuous_decrease_perc = {
            "sum_percentage_growth": round(abs(decrease_stats.loc[biggest_decr_idx, 'sum_percentage_growth']), 5) * -1,
            "start_window_index": int(decrease_stats.loc[biggest_decr_idx, 'start_window_index']),
            "end_window_index": int(decrease_stats.loc[biggest_decr_idx, 'end_window_index'])
        }
    else:
        biggest_continuous_decrease_perc = {
            "sum_percentage_growth": 0.0,
            "start_window_index": None,
            "end_window_index": None
        }

    # Compile the results into a dictionary
    statistics = {
        "biggest_continuous_increase_perc": biggest_continuous_increase_perc,
        "biggest_continuous_decrease_perc": biggest_continuous_decrease_perc
    }

    return statistics

def compute_all_features_for_timeseries(time_series, date_series, unpack_advanced_features=True):
    """
    Computes features for a given time series and its percentage growth version.

    Parameters:
        time_series (list): A list of numerical values representing the time series.

    Returns:
        tuple:
            - dict: Contains features for both the original and percentage growth time series. All features are rounded to at most 5 decimals.
            - pd.DataFrame: The growth percentage DataFrame with 'window_index', 'value', and 'percentage_growth' columns.
    """
    df_ts = pd.DataFrame({'window_index': range(1, len(time_series) + 1), 'value': time_series, 'date': date_series})

    # Compute features on original time series
    basic_features, _, _, _ = compute_simple_features_og_ts(df_ts)
    advanced_features = compute_extra_statistics_og_ts(df_ts)

    if unpack_advanced_features:
        advanced_features = {
            'longest_continuous_increase': advanced_features['longest_continuous_increase']['length'],
            'biggest_continuous_increase': advanced_features['biggest_continuous_increase']['sum'],
            'longest_continuous_decrease': advanced_features['longest_continuous_decrease']['length'],
            'biggest_continuous_decrease': advanced_features['biggest_continuous_decrease']['sum'],
            'longest_continuous_increase_start': advanced_features['longest_continuous_increase']['start_window_index'],
            'biggest_continuous_increase_start': advanced_features['biggest_continuous_increase']['start_window_index'],
            'longest_continuous_decrease_start': advanced_features['longest_continuous_decrease']['start_window_index'],
            'biggest_continuous_decrease_start': advanced_features['biggest_continuous_decrease']['start_window_index']
        }

    # Convert original dataset to the growth percentage time series
    df_ts_growth_perc = convert_to_percentage_growth(df_ts)

    # Compute features on growth percentage time series
    basic_features_growth_perc, _, _, _ = compute_simple_features_growth_perc_ts(df_ts_growth_perc)
    advanced_features_growth_perc = compute_extra_statistics_growth_perc_ts(df_ts_growth_perc)

    if unpack_advanced_features:
        advanced_features_growth_perc = {
            'biggest_continuous_increase_perc': advanced_features_growth_perc['biggest_continuous_increase_perc']['sum_percentage_growth'],
            'biggest_continuous_decrease_perc': advanced_features_growth_perc['biggest_continuous_decrease_perc']['sum_percentage_growth'],
            'biggest_continuous_increase_perc_start': advanced_features_growth_perc['biggest_continuous_increase_perc']['start_window_index'],
            'biggest_continuous_decrease_perc_start': advanced_features_growth_perc['biggest_continuous_decrease_perc']['start_window_index']
        }

    # Compute features
    features_total_hierarchical = {}
    features_total_hierarchical["og_time_series"] = {**basic_features, **advanced_features}
    features_total_hierarchical["growth_perc_time_series"] = {**basic_features_growth_perc, **advanced_features_growth_perc}

    features_total_og_basic = {f'og_{key}': value for key, value in basic_features.items()}
    features_total_og_adv = {f'og_{key}': value for key, value in advanced_features.items()}
    features_total_growth_basic = {f'growth_{key}': value for key, value in basic_features_growth_perc.items()}
    features_total_growth_adv = {f'growth_{key}': value for key, value in advanced_features_growth_perc.items()}
    features_total = {**features_total_og_basic, **features_total_og_adv, **features_total_growth_basic, **features_total_growth_adv}

    return features_total, features_total_hierarchical, df_ts_growth_perc

if __name__ == '__main__':
    # Example usage
    time_series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    date_series = pd.date_range('2022-01-01', periods=len(time_series)).tolist()

    features_total, features_total_hierarchical, df_ts_growth_perc = compute_all_features_for_timeseries(time_series, date_series)
    print(features_total)
    print(features_total_hierarchical)
    print(df_ts_growth_perc)