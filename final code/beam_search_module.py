from collections import deque
from queue import PriorityQueue
import pandas as pd
import numpy as np
import math

def gt(a, b):
    """Returns True if a is greater than b."""
    return a > b

def leeq(a, b):
    """Returns True if a is less than or equal to b."""
    return a <= b

def eq(a, b):
    """Returns True if a is equal to b."""
    return a == b

def neq(a, b):
    """Returns True if a is not equal to b."""
    return not eq(a, b)

def extract_subgroup(descriptors, data, col_index_dict):
    """Extracts a subgroup of data that matches all provided descriptors.

    Args:
        descriptors: A list of descriptors, each containing an attribute name, value, and operator.
        data: The dataset from which to extract the subgroup.
        col_index_dict: A dictionary mapping column names to their indices.

    Returns:
        A list of rows from the data that match all descriptors.
    """
    result = []
    for row in data:
        check = True
        for attribute in descriptors:
            att_name, descr_value, operator = attribute  # unpack 3 values from attribute
            att_index = col_index_dict[att_name]  # get the index for the attribute
            value = operator(row[att_index], descr_value)  # apply the operator

            if not value:  # if any descriptor does not match
                check = False
                break

        if check:  # if all descriptors match
            result.append(row)  # add the row to the result

    return result

def refin(seed, data, types, nr_bins, descr_indices, index_col_dict, col_index_dict):
    """Generates new descriptor sets by refining the seed descriptors.

    Args:
        seed: The initial set of descriptors to refine.
        data: The dataset used for extracting subgroups.
        types: The types of each column in the dataset.
        nr_bins: The number of bins to create for numeric attributes.
        descr_indices: The indices of potential descriptors.
        index_col_dict: A dictionary mapping index to column names.
        col_index_dict: A dictionary mapping column names to indices.

    Returns:
        A list of new descriptor sets created by refining the seed.
    """
    res = []
    used_descr = [col_index_dict[i[0]] for i in seed]  # get used descriptors
    not_used_indices = descr_indices[:]  # make a copy of descriptor indices
    # Filter out indices that are already used or are numeric types
    not_used_indices = [i for i in not_used_indices if i not in used_descr or types[i] == "numeric"]

    for i in not_used_indices:
        aux = list(seed)[:]  # copy the seed

        if types[i] == 'numeric':
            s = extract_subgroup(seed, data, col_index_dict)  # extract subgroup based on seed
            all_values = [float(entry[i]) for entry in s]  # get all values for the numeric attribute
            all_values = sorted(all_values)  # sort values
            n = len(all_values)
            # Create split points for binning
            split_points = [all_values[math.floor(j * (n/nr_bins))] for j in range(1, nr_bins)]
            for s in split_points:
                func1 = leeq
                func2 = gt

                # Create two new descriptors for each split point
                local0 = aux[:]
                local0.append((index_col_dict[i], s, func1))  # descriptor for less than or equal
                res.append(local0)

                local1 = aux[:]
                local1.append((index_col_dict[i], s, func2))  # descriptor for greater than
                res.append(local1)

        elif types[i] == 'binary':
            func = eq  # equality function for binary descriptors
            local0 = aux[:]
            local0.append((index_col_dict[i], 0, func))  # descriptor for value 0
            local1 = aux[:]
            local1.append((index_col_dict[i], 1, func))  # descriptor for value 1
            res.append(local0)
            res.append(local1)

        else:  # nominal attributes
            all_values = [entry[i] for entry in data]  # get all unique values
            for j in set(all_values):
                func1 = eq
                local0 = aux[:]
                local0.append((index_col_dict[i], j, func1))  # descriptor for equality
                res.append(local0)

                # descriptor for inequality, not used because gives bad results
                # func2 = neq
                # local1 = aux[:]
                # local1.append((index_col_dict[i], j, func2))
                # res.append(local1)

    return res

def put_item_in_queue(queue, quality, descriptor, size=0, window_index=0):
    """
    Adds an item to a priority queue based on its quality. If the queue is full,
    it replaces the item with the lowest quality if the new item's quality is higher.

    Args:
        queue: The priority queue to add the item to.
        quality: Quality score of the item (higher is better).
        descriptor: Description or identifier of the item.
        size: Size of the item group (optional, default is 0).
        window_index: Index of the window where quality score is max (optional, default is 0).
    """
    if queue.full():  # If queue is full, check for replacement
        min_quality, min_descriptor, min_size, min_window_index = queue.get()  # Remove lowest quality item
        if min_quality >= quality:  # If new item's quality is not higher, keep the old item
            queue.put((min_quality, min_descriptor, min_size, min_window_index))
        else:  # Replace with the new item if its quality is better
            queue.put((quality, descriptor, size, window_index))
    else:  # If queue has space, simply add the new item
        queue.put((quality, descriptor, size, window_index))


def categorize_columns_in_order(df, att_columns):
    """Categorizes columns of a DataFrame into numeric, binary, and nominal types.

    Args:
        df: The DataFrame containing the data.
        att_columns: The columns to categorize.

    Returns:
        A list of column types corresponding to the provided attribute columns.
    """
    column_types = []  # List to store the categories in order

    for col in att_columns:
        if pd.api.types.is_numeric_dtype(df[col]):  # Check if the column is numeric
            column_types.append('numeric')
        elif df[col].nunique() == 2:  # Check for binary columns
            column_types.append('binary')
        else:  # Otherwise, treat it as nominal
            column_types.append('nominal')

    return column_types

def make_rolling_windows(growth_target, window_size):
    """Creates rolling windows for the target data.

    Args:
        growth_target: The target data for which to create rolling windows.
        window_size: The size of each window.

    Returns:
        A new array of rolling windows.
    """
    return np.lib.stride_tricks.sliding_window_view(growth_target, window_shape=window_size)[::window_size]

def quality_measure(targets_subgroup, targets_baseline,
                    aggregate_func_window=np.mean, aggregate_func=np.mean, dataset_size = 78400):
    """Calculates a quality measure for a subgroup compared to a baseline.

    Args:
        targets_subgroup: list with timeseries where the timeseries are divided in windows (list with lists with lists)
        targets_baseline: list with baseline target values in windows (list with lists).
        aggregate_func_window: Function to aggregate over windows (default: mean).
        aggregate_func: Function to aggregate the final quality measure (default: mean).
        dataset_size: number of rows in the dataset

    Returns:
        A quality score representing the difference between the subgroup and baseline.
    """
    # Aggregate points in each window for each timeseries in the subgroup
    subgroup_aggregated_windows = aggregate_func_window(targets_subgroup, axis=2)

    # Calculate mean values for the baseline and subgroup for each window
    baseline_means = np.mean(targets_baseline, axis=1)
    subgroup_means = np.mean(subgroup_aggregated_windows, axis=0)

    # Calculate absolute differences between means for each window
    abs_diff_mean = np.abs(subgroup_means - baseline_means)

    # Calculate standard deviation of the subgroup for each window
    subgroup_std = np.std(subgroup_aggregated_windows, axis=0)
    safe_subgroup_std = np.where(subgroup_std == 0, 1, subgroup_std)

    # Calculate z-scores
    z_scores = abs_diff_mean / safe_subgroup_std#(subgroup_std / np.sqrt(0.5*len(targets_subgroup)))

    # Calculate the final quality score
    quality_score = aggregate_func(z_scores)

    max_index_quality = np.argmax(z_scores)

    # calculate subgroup proportion
    proportion_subgroup = len(targets_subgroup) / dataset_size

    # calculate sqrt entropy
    if proportion_subgroup != 0 and proportion_subgroup != 1:
        entropy = -1*((proportion_subgroup*np.log2(proportion_subgroup)) + ((1-proportion_subgroup)*np.log2(1-proportion_subgroup)))
        sqrt_entropy = np.sqrt(entropy)
    else:
        sqrt_entropy=0

    # mulitply quality score with the sqrt entropy
    entropy_quality_score = sqrt_entropy * quality_score

    return entropy_quality_score, max_index_quality

def get_all_descriptors(pq, index=1):
    """Retrieve all descriptors from a priority queue without altering its contents.

    Args:
        pq: A priority queue containing descriptor tuples.
        index: The index of item to retrieve

    Returns:
        A list with info out of the results pq
    """
    temp_items = []  # Temporary list to store all items from the queue
    info = []  # List to store info of results pq (descriptors, quality, etc.)

    # Retrieve all items from the queue
    while not pq.empty():
        item = pq.get()  # Get item from the queue
        temp_items.append(item)  # Store the item temporarily
        info.append(item[index])  # Extract and store the descriptor

    # Put all items back into the queue to maintain its original state
    for item in temp_items:
        pq.put(item)

    return info

def metrics_match(metric1, metric2):
    """
    Compares two metrics to determine if they match based on their values and functions.

    Args:
        metric1 (tuple): A tuple containing a value and a function (value1, func1).
        metric2 (tuple): A tuple containing a value and a function (value2, func2).

    Returns:
        bool: True if the metrics match (i.e., same function and same value), False otherwise.
    """
    value1, func1 = metric1
    value2, func2 = metric2

    # If functions differ, metrics don't match
    if func1 != func2:
        return False

    # If values differ, metrics don't match
    if value1 != value2:
        return False

    # If both the function and value match, return True
    return True

def descriptors_similar_paper(quality, descriptor1, pq, min_quality_improvement):
    """
    Determines if descriptor1 is similar to any descriptors in a priority queue based on matching metrics
    and a minimum quality improvement threshold.

    Args:
        quality (float): The quality score of descriptor1.
        descriptor1 (list): The list of metrics for the first descriptor.
        pq (PriorityQueue): Priority queue containing other descriptors.
        min_quality_improvement (float): Minimum improvement in quality required to consider similarity.

    Returns:
        bool: True if a similar descriptor is found, False otherwise.
    """
    # If descriptor1 has only one metric, return False early (no comparison needed)
    if len(descriptor1) == 1:
        return False

    # Total conditions to match (all but one metric must match)
    total_conditions = len(descriptor1) - 1

    # Get descriptor and quality lists from the priority queue
    descriptor_list = get_all_descriptors(pq, 1)  # Descriptors
    quality_list = get_all_descriptors(pq, 0)     # Qualities

    # Early exit if the closest quality difference is too large
    min_index = min(range(len(quality_list)), key=lambda i: abs(quality_list[i] - quality))
    min_difference = abs(quality_list[min_index] - quality)
    if min_difference > abs(quality_list[min_index] * min_quality_improvement):
        return False

    # Find indices where the quality difference is within the improvement threshold
    indices = [i for i, q in enumerate(quality_list) if abs(quality - q) <= min_quality_improvement]

    # Filter descriptor_list based on matching indices
    filtered_descriptors = [descriptor_list[i] for i in indices]

    # Compare descriptor1 against all filtered descriptors
    for descriptor2 in filtered_descriptors:
        match_count = 0

        # Copy descriptor2 to track used metrics
        remaining_descriptor2 = list(descriptor2)

        # Compare each metric in descriptor1 with those in descriptor2
        for metric1_name, value1, func1 in descriptor1:
            matched = False

            # Look for matching metrics in descriptor2
            for i, (metric2_name, value2, func2) in enumerate(remaining_descriptor2):
                if metric1_name == metric2_name:
                    if metrics_match((value1, func1), (value2, func2)):
                        matched = True
                        # Remove matched metric from remaining_descriptor2
                        remaining_descriptor2.pop(i)
                        break

            if matched:
                match_count += 1

            # If enough metrics match, consider descriptors similar
            if match_count >= total_conditions:
                return True

    # If no matching descriptor is found, return False
    return False

def dominance_pruning(pq, subgroup_size, col_index_dict, targets_baseline, data, target_ind, min_quality_improvement, agg_func, datasetsize):
    """
    Performs dominance pruning on a priority queue of descriptors by evaluating subgroups formed by
    removing individual metrics from each descriptor.

    Args:
        pq (PriorityQueue): The priority queue containing descriptors and their quality measures.
        subgroup_size (int): Minimum size required for a subgroup to be considered.
        col_index_dict (dict): Dictionary mapping column names to their indices in the data.
        targets_baseline (np.array): Baseline target values for quality comparison.
        data (list): The dataset from which subgroups are extracted.
        target_ind (int): Index of the target value in the dataset.
        min_quality_improvement (float): Minimum required quality improvement for acceptance.
        agg_func (function): Function used to aggregate quality measures.
        datasetsize (int): Size of the dataset for quality calculations.

    Returns:
        None: The function modifies the priority queue in place.
    """
    # Get lists of descriptors and their corresponding quality measures from the priority queue
    descriptor_list = get_all_descriptors(pq, 1)
    quality_list = get_all_descriptors(pq, 0)

    # Iterate through each descriptor and its quality
    for org_quality, org_descriptor in zip(quality_list, descriptor_list):
        # Skip descriptors with fewer than 3 metrics
        if len(org_descriptor) < 3:
            continue

        # Evaluate each metric in the descriptor
        for cc in range(len(org_descriptor)):
            # Create a temporary subgroup by removing one metric at a time
            temp_subgroup = org_descriptor[:cc] + org_descriptor[cc+1:]
            # Extract the subgroup from the data based on the current descriptor
            subgroup = extract_subgroup(temp_subgroup, data, col_index_dict)

            # Ensure the subgroup is large enough
            if len(subgroup) >= subgroup_size:
                # Extract target values for the current subgroup
                targets_subgroup = [j[target_ind] for j in subgroup]

                # Calculate the quality measure for the subgroup
                quality_result, window_index_quality = quality_measure(
                    targets_subgroup, targets_baseline,
                    aggregate_func=agg_func, dataset_size=datasetsize
                )

                # Check if the new descriptor is not similar to existing descriptors
                if not descriptors_similar_paper(quality_result, temp_subgroup, pq, min_quality_improvement):
                    # Add the new quality and descriptor to the priority queue
                    put_item_in_queue(pq, quality_result, tuple(temp_subgroup), len(subgroup), window_index_quality)


def beam_search_with_constraint(data, targets_baseline, column_names, beam_width, beam_depth, nr_bins, nr_saved, subgroup_size, target, types, window_size, min_quality_improvement, agg_func=np.mean):
    """
    Performs beam search with constraints to avoid adding similar descriptors.

    Args:
        data (list): The dataset to analyze.
        targets_baseline (list): The baseline target values for comparison.
        column_names (list): The names of the columns in the dataset.
        beam_width (int): The number of descriptors to keep at each depth level.
        beam_depth (int): The maximum depth of the beam search.
        nr_bins (int): The number of bins to create for numeric attributes.
        nr_saved (int): The number of best results to save.
        subgroup_size (int): The minimum size of a subgroup to consider.
        target (str): The target variable for which to evaluate subgroups.
        types (list): The types of each column in the dataset (e.g., numeric, binary, nominal).
        window_size (int): The size of the rolling window for target variable.
        min_quality_improvement (float): Minimum improvement needed to include similar descriptors.
        agg_func (function): Aggregation function for quality measures (default is np.mean).

    Returns:
        list: A list of the best descriptor sets found during the search, ensuring no similar descriptors.
    """
    # Create dictionaries for indexing columns by name and vice versa
    index_col_dict = {i: col for i, col in enumerate(column_names)}
    col_index_dict = {col: i for i, col in enumerate(column_names)}
    target_ind = column_names.index(target)  # Get index of the target column
    att_indices = list(range(len(column_names)))  # All attribute indices
    att_indices.remove(target_ind)  # Remove target index from attributes

    # Prepare data windows using rolling windows for the target variable
    data_windows = []
    for row in data:
        new_row = row[:]  # Copy the row
        new_row[target_ind] = make_rolling_windows(row[target_ind], window_size)  # Apply rolling window
        data_windows.append(new_row)  # Add the new row to data_windows

    # Update data and baseline targets to use rolling windows
    data = data_windows
    targets_baseline = make_rolling_windows(targets_baseline, window_size)

    # Initialize a deque for the beam search and a priority queue for results
    beam_queue = deque([()])  # Start with an empty seed
    results = PriorityQueue(nr_saved)  # Queue to hold the best results
    results.put((0, [(0, 0, 0)], 0, 0))  # Initialize with a dummy descriptor

    # Iterate through each depth of the beam search
    for depth in range(beam_depth):
        beam = PriorityQueue(beam_width)  # Initialize a new beam for this depth

        # While there are seeds in the beam queue
        while beam_queue:
            seed = beam_queue.popleft()  # Get the next seed descriptor
            descriptor_set = refin(seed, data, types, nr_bins, att_indices, index_col_dict, col_index_dict)  # Refine descriptors based on seed

            # Evaluate each descriptor set generated
            for descriptor in descriptor_set:
                subgroup = extract_subgroup(descriptor, data, col_index_dict)  # Extract subgroup for the current descriptor
                if len(subgroup) >= subgroup_size:  # Ensure subgroup is large enough
                    targets_subgroup = [i[target_ind] for i in subgroup]  # Extract target values for the subgroup
                    quality_result, window_index_quality = quality_measure(
                        targets_subgroup, targets_baseline,
                        aggregate_func=agg_func, dataset_size=len(data)
                    )  # Calculate quality measure

                    # Check if the new descriptor is not similar to existing descriptors
                    if not descriptors_similar_paper(quality_result, descriptor, results, min_quality_improvement):
                        put_item_in_queue(results, quality_result, tuple(descriptor), len(subgroup), window_index_quality)  # Add to results queue
                        put_item_in_queue(beam, quality_result, tuple(descriptor))  # Add to the current beam

        # After processing the beam, update the beam queue with new combinations
        while not beam.empty():
            new_combination = beam.get()[1]  # Get the highest quality descriptor from the beam
            beam_queue.append(new_combination)  # Add it to the next depth of the beam search

    # Apply dominance pruning to the results
    dominance_pruning(results, subgroup_size, col_index_dict, targets_baseline, data, target_ind, min_quality_improvement, agg_func, len(data))

    # Compile results into a list and reverse to have the best results first
    results_list = []
    while not results.empty():
        item = results.get()  # Get items from the results queue
        results_list.append(item)  # Add to the results list
    results_list.reverse()  # Reverse to prioritize best results

    return results_list  # Return the list of best descriptor sets



def filter_df_on_descriptors(df, descriptors):
    """Filters a DataFrame based on specified descriptors.

    Args:
        df: The DataFrame to filter.
        descriptors: A list of descriptors, each containing an attribute name, value, and operator.

    Returns:
        A filtered DataFrame where all descriptors hold true.
    """
    # Loop through each descriptor to apply filtering
    for desc in descriptors:
        # Apply the operator defined in the descriptor to filter the DataFrame
        df = df[df[desc[0]].apply(lambda x: desc[2](x, desc[1]))]  # Filter based on the descriptor
    return df  # Return the filtered DataFrame


