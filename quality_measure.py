import numpy as np

def make_rolling_windows(growth_target, window_size, window_stepsize):
    """
    Creates rolling windows from the input data (growth_target) of the specified size and step size.

    Parameters:
    growth_target (list or array): The data from which to create rolling windows.
    window_size (int): The size of each rolling window.
    window_stepsize (int): The number of steps to move forward to create the next window.

    Returns:
    windows (list): A list of rolling windows from the input data.
    """
    windows = []
    # Loop through the data and create windows of the specified size
    for i in range(0, len(growth_target) - window_size + 1, window_stepsize):
        window = growth_target[i:i + window_size]
        windows.append(window)
    return windows


def quality_measure(targets_subgroup, targets_baseline,
                    window_size=7, window_overlap=0,
                    aggregate_func_window=np.mean, aggregate_func=np.max):
    """
    Calculates a quality score by comparing the subgroup targets to the baseline targets using rolling windows.

    Parameters:
    targets_subgroup (list of lists or arrays): A list of subgroup time series, where each element is a list representing a time series of a stock.
    targets_baseline (list or array): The baseline target data for comparison.
    window_size (int): The size of each rolling window. Default is 7.
    window_overlap (int): The overlap between windows (number of elements). Default is 0 (no overlap).
    aggregate_func_window (function): The function to apply to aggregate points in a window. Use Numpy functions . Default is np.mean.
    aggregate_func (function): The function to aggregate the z-scores across all windows. Use Numpy functions. Default is np.max.

    Returns:
    quality_score (float): A score representing the quality of the subgroup targets compared to the baseline.
    """

    # Calculate the effective window step size based on the overlap
    window_overlap = window_size - window_overlap

    # Create rolling windows for each subgroup and baseline using the specified window size and step size
    target_windows_subgroup = [make_rolling_windows(targets_subgroup_i, window_size, window_overlap)
                               for targets_subgroup_i in targets_subgroup]
    target_windows_baseline = make_rolling_windows(targets_baseline, window_size, window_overlap)

    all_z_scores = []  # List to store z-scores for each window comparison

    # Iterate through each baseline window and calculate the z-scores
    for i in range(len(target_windows_baseline)):
        # Compute the mean of the baseline window
        mean_baseline_t = np.mean(target_windows_baseline[i])

        # Compute the mean of the subgroup window (aggregated across all subgroups)
        mean_subgroup_t = np.mean([aggregate_func_window(target_windows_subgroup[j][i])
                                   for j in range(len(target_windows_subgroup))])

        # Calculate the absolute difference between baseline and subgroup means
        abs_diff_mean = abs(mean_subgroup_t - mean_baseline_t)

        # Calculate the standard deviation of the subgroup windows
        standard_deviation_subgroup = np.std([aggregate_func_window(target_windows_subgroup[j][i])
                                              for j in range(len(target_windows_subgroup))])

        # Calculate the standard error (accounts for number of subgroups)
        standard_error_subgroup = standard_deviation_subgroup / np.sqrt(len(target_windows_subgroup))

        # Compute the z-score for this window
        z_score_window = abs_diff_mean / standard_error_subgroup

        all_z_scores.append(z_score_window)

    # Apply the aggregate function to the list of z-scores to get the final quality score
    quality_score = aggregate_func(all_z_scores)

    return quality_score


