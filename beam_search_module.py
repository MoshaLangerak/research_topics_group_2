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

                # descriptor for not equality, not used because gives bad results
                # func2 = neq
                # local1 = aux[:]
                # local1.append((index_col_dict[i], j, func2))
                # res.append(local1)

    return res

def put_item_in_queue(queue, quality, descriptor, size=0):
    """Adds an item to a priority queue based on its quality.

    Args:
        queue: The priority queue to which the item will be added.
        quality: The quality measure of the item.
        descriptor: The descriptor associated with the item.
        size: The size of the subgroup represented by the descriptor.
    """
    if queue.full():  # if the queue is full
        min_quality, min_descriptor, min_size = queue.get()  # get the lowest quality item
        if min_quality >= quality:  # if the new item is not better, put the old one back
            queue.put((min_quality, min_descriptor, min_size))
        else:  # otherwise, add the new item
            queue.put((quality, descriptor, size))
    else:
        queue.put((quality, descriptor, size))  # add new item to the queue

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
                    aggregate_func_window=np.mean, aggregate_func=np.max):
    """Calculates a quality measure for a subgroup compared to a baseline.

    Args:
        targets_subgroup: list with timeseries where the timeseries are divided in windows (list with lists with lists)
        targets_baseline: list with baseline target values in windows (list with lists).
        aggregate_func_window: Function to aggregate over windows (default: mean).
        aggregate_func: Function to aggregate the final quality measure (default: max).

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

    # Calculate standard error of the subgroup for each window
    subgroup_std = np.std(subgroup_aggregated_windows, axis=0)
    standard_error_subgroup = subgroup_std / np.sqrt(len(targets_subgroup))

    # Calculate z-scores
    z_scores = np.divide(abs_diff_mean, standard_error_subgroup, where=standard_error_subgroup != 0)

    # Calculate the final quality score
    quality_score = aggregate_func(z_scores)

    return quality_score


def beam_search(data, targets_baseline, column_names, beam_width, beam_depth, nr_bins, nr_saved, subgroup_size, target, types, window_size, max_subgroup_size=100000):
    """Performs beam search to identify optimal descriptors for subgroups.

    Args:
        data: The dataset to analyze.
        targets_baseline: The baseline target values for comparison.
        column_names: The names of the columns in the dataset.
        beam_width: The number of descriptors to keep at each depth level.
        beam_depth: The maximum depth of the beam search.
        nr_bins: The number of bins to create for numeric attributes.
        nr_saved: The number of best results to save.
        subgroup_size: The minimum size of a subgroup to consider.
        target: The target variable for which to evaluate subgroups.
        types: The types of each column in the dataset (e.g., numeric, binary, nominal).
        window_size: The size of the rolling window.

    Returns:
        A list of the best descriptor sets found during the search.
    """
    # Create dictionaries for indexing columns by name and vice versa
    index_col_dict = {i: col for i, col in enumerate(column_names)}
    col_index_dict = {col: i for i, col in enumerate(column_names)}
    target_ind = column_names.index(target)  # Get index of the target column
    att_indices = list(range(len(column_names)))  # Create a list of all indices
    att_indices.remove(target_ind)  # Remove the target index from attribute indices

    # Prepare data windows with rolling windows for the target variable
    data_windows = []
    for row in data:
        new_row = row[:]  # Create a copy of the row
        new_row[target_ind] = make_rolling_windows(row[target_ind], window_size)  # Apply rolling window
        data_windows.append(new_row)  # Add the new row to data_windows

    # Update the data and baseline targets to use rolling windows
    data = data_windows
    targets_baseline = make_rolling_windows(targets_baseline, window_size)

    # Initialize a deque for the beam search and a priority queue for results
    beam_queue = deque([()])  # Start with an empty seed
    results = PriorityQueue(nr_saved)  # Queue to hold the best results

    # Iterate through each depth of the beam search
    for depth in range(beam_depth):
        beam = PriorityQueue(beam_width)  # Initialize a new beam for this depth

        # While there are seeds in the beam queue
        while bool(beam_queue):
            seed = beam_queue.popleft()  # Get the next seed descriptor
            descriptor_set = refin(seed, data, types, nr_bins, att_indices, index_col_dict, col_index_dict)  # Refine descriptors based on seed

            # Evaluate each descriptor set generated
            for descriptor in descriptor_set:
                subgroup = extract_subgroup(descriptor, data, col_index_dict)  # Extract subgroup for the current descriptor
                if len(subgroup) >= subgroup_size and len(subgroup)<max_subgroup_size:  # Ensure subgroup is large enough
                    targets_subgroup = [i[target_ind] for i in subgroup]  # Extract target values for the subgroup
                    quality_result = quality_measure(targets_subgroup, targets_baseline)  # Calculate quality measure
                    put_item_in_queue(results, quality_result, tuple(descriptor), len(subgroup))  # Add to results queue
                    put_item_in_queue(beam, quality_result, tuple(descriptor))  # Add to the current beam

        # After processing the beam, update the beam queue with new combinations
        while not beam.empty():
            new_combination = beam.get()  # Get the highest quality descriptor from the beam
            new_combination = new_combination[1]  # Extract the descriptor from the tuple
            beam_queue.append(new_combination)  # Add it to the next depth of the beam search

    # Compile results into a list and reverse to have the best results first
    results_list = []
    while not results.empty():
        item = results.get()  # Get items from the results queue
        results_list.append(item)  # Add to the results list
    results_list.reverse()  # Reverse the list to have best results first

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



###########################################################################################################
################# BEAM SEARCH WITH (SELF MADE) CONSTRAINT #################################################
### it is finished but not perfect and complicated, maybe finding something in the literature is better ###
###########################################################################################################



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


def are_descriptors_similar(descriptor1, pq):
    """Check if the given descriptor is similar to any descriptors in the priority queue.

    Args:
        descriptor1: The first descriptor to compare (a list of (metric, value, func) tuples).
        pq: A priority queue containing other descriptors.

    Returns:
        True if a similar descriptor is found in the queue, False otherwise.
    """
    tolerance = 0.25  # Tolerance range for numeric comparison
    desc1_dict = {metric: (value, func) for metric, value, func in descriptor1}  # Convert descriptor1 to a dictionary
    descriptor_list = get_all_descriptors(pq)  # Get all descriptors from the queue

    # Iterate through each descriptor in the list
    for descriptor2 in descriptor_list:
        desc2_dict = {metric: (value, func) for metric, value, func in descriptor2}  # Convert descriptor2 to a dictionary

        # If the length of descriptors doesn't match, skip
        if len(desc1_dict) != len(desc2_dict):
            continue

        all_metrics_match = True  # Flag to track if all metrics match

        # Check each metric in descriptor1
        for metric in desc1_dict:
            if metric in desc2_dict:
                value1, func1 = desc1_dict[metric]
                value2, func2 = desc2_dict[metric]

                # Check if the functions are the same
                if func1 != func2:
                    all_metrics_match = False
                    break

                # Check if numeric values are within tolerance range
                if (not isinstance(value1, str)) and abs(value1 - value2) > abs(tolerance * value1):
                    all_metrics_match = False
                    break

                # For strings, check if values are exactly the same
                if isinstance(value1, str) and value1 != value2:
                    all_metrics_match = False
                    break
            else:
                # If metric in descriptor1 is not in descriptor2, they are not similar
                all_metrics_match = False
                break

        # If all metrics match, return True (descriptors are similar)
        if all_metrics_match:
            return True

    # If no similar descriptors are found, return False
    return False

def beam_search_with_constraint(data, targets_baseline, column_names, beam_width, beam_depth, nr_bins, nr_saved, subgroup_size, target, types, window_size, max_subgroup_size=100000):
    """Performs beam search with a constraint to avoid adding similar descriptors.

    Args:
        data: The dataset to analyze.
        targets_baseline: The baseline target values for comparison.
        column_names: The names of the columns in the dataset.
        beam_width: The number of descriptors to keep at each depth level.
        beam_depth: The maximum depth of the beam search.
        nr_bins: The number of bins to create for numeric attributes.
        nr_saved: The number of best results to save.
        subgroup_size: The minimum size of a subgroup to consider.
        target: The target variable for which to evaluate subgroups.
        types: The types of each column in the dataset (e.g., numeric, binary, nominal).
        window_size: The size of the rolling window.

    Returns:
        A list of the best descriptor sets found during the search, ensuring no similar descriptors.
    """
    # Create dictionaries for indexing columns by name and vice versa
    index_col_dict = {i: col for i, col in enumerate(column_names)}
    col_index_dict = {col: i for i, col in enumerate(column_names)}
    target_ind = column_names.index(target)  # Get index of the target column
    att_indices = list(range(len(column_names)))  # Create a list of all indices
    att_indices.remove(target_ind)  # Remove the target index from attribute indices

    # Prepare data windows with rolling windows for the target variable
    data_windows = []
    for row in data:
        new_row = row[:]  # Create a copy of the row
        new_row[target_ind] = make_rolling_windows(row[target_ind], window_size)  # Apply rolling window
        data_windows.append(new_row)  # Add the new row to data_windows

    # Update the data and baseline targets to use rolling windows
    data = data_windows
    targets_baseline = make_rolling_windows(targets_baseline, window_size)

    # Initialize a deque for the beam search and a priority queue for results
    beam_queue = deque([()])  # Start with an empty seed
    results = PriorityQueue(nr_saved)  # Queue to hold the best results
    results.put((0, [(0, 0, 0)], 0))  # Add a dummy descriptor to initialize

    # Iterate through each depth of the beam search
    for depth in range(beam_depth):
        beam = PriorityQueue(beam_width)  # Initialize a new beam for this depth

        # While there are seeds in the beam queue
        while bool(beam_queue):
            seed = beam_queue.popleft()  # Get the next seed descriptor
            descriptor_set = refin(seed, data, types, nr_bins, att_indices, index_col_dict, col_index_dict)  # Refine descriptors based on seed

            # Evaluate each descriptor set generated
            for descriptor in descriptor_set:
                subgroup = extract_subgroup(descriptor, data, col_index_dict)  # Extract subgroup for the current descriptor
                if len(subgroup) >= subgroup_size and len(subgroup) < max_subgroup_size and not are_descriptors_similar(descriptor, results):  # Ensure subgroup is large enough and descriptor is not similar
                    targets_subgroup = [i[target_ind] for i in subgroup]  # Extract target values for the subgroup
                    quality_result = quality_measure(targets_subgroup, targets_baseline)  # Calculate quality measure
                    put_item_in_queue(results, quality_result, tuple(descriptor), len(subgroup))  # Add to results queue
                    put_item_in_queue(beam, quality_result, tuple(descriptor))  # Add to the current beam

        # After processing the beam, update the beam queue with new combinations
        while not beam.empty():
            new_combination = beam.get()  # Get the highest quality descriptor from the beam
            new_combination = new_combination[1]  # Extract the descriptor from the tuple
            beam_queue.append(new_combination)  # Add it to the next depth of the beam search

    # Compile results into a list and reverse to have the best results first
    results_list = []
    while not results.empty():
        item = results.get()  # Get items from the results queue
        results_list.append(item)  # Add to the results list
    results_list.reverse()  # Reverse the list to have best results first

    return results_list  # Return the list of best descriptor sets

###########################################################################################################
################# BEAM SEARCH WITH CONSTRAINT FROM PAPER ##################################################
###########################################################################################################

def beam_search_with_constraint_paper(data, targets_baseline, column_names, beam_width, beam_depth, nr_bins, nr_saved, subgroup_size, target, types, window_size, max_subgroup_size=100000):
    """Performs beam search with a constraint to avoid adding similar descriptors.

    Args:
        data: The dataset to analyze.
        targets_baseline: The baseline target values for comparison.
        column_names: The names of the columns in the dataset.
        beam_width: The number of descriptors to keep at each depth level.
        beam_depth: The maximum depth of the beam search.
        nr_bins: The number of bins to create for numeric attributes.
        nr_saved: The number of best results to save.
        subgroup_size: The minimum size of a subgroup to consider.
        target: The target variable for which to evaluate subgroups.
        types: The types of each column in the dataset (e.g., numeric, binary, nominal).
        window_size: The size of the rolling window.

    Returns:
        A list of the best descriptor sets found during the search, ensuring no similar descriptors.
    """
    # Create dictionaries for indexing columns by name and vice versa
    index_col_dict = {i: col for i, col in enumerate(column_names)}
    col_index_dict = {col: i for i, col in enumerate(column_names)}
    target_ind = column_names.index(target)  # Get index of the target column
    att_indices = list(range(len(column_names)))  # Create a list of all indices
    att_indices.remove(target_ind)  # Remove the target index from attribute indices

    # Prepare data windows with rolling windows for the target variable
    data_windows = []
    for row in data:
        new_row = row[:]  # Create a copy of the row
        new_row[target_ind] = make_rolling_windows(row[target_ind], window_size)  # Apply rolling window
        data_windows.append(new_row)  # Add the new row to data_windows

    # Update the data and baseline targets to use rolling windows
    data = data_windows
    targets_baseline = make_rolling_windows(targets_baseline, window_size)

    # Initialize a deque for the beam search and a priority queue for results
    beam_queue = deque([()])  # Start with an empty seed
    results = PriorityQueue(nr_saved)  # Queue to hold the best results
    results.put((0, [(0, 0, 0)], 0))  # Add a dummy descriptor to initialize

    # Iterate through each depth of the beam search
    for depth in range(beam_depth):
        beam = PriorityQueue(beam_width)  # Initialize a new beam for this depth

        # While there are seeds in the beam queue
        while bool(beam_queue):
            seed = beam_queue.popleft()  # Get the next seed descriptor
            descriptor_set = refin(seed, data, types, nr_bins, att_indices, index_col_dict, col_index_dict)  # Refine descriptors based on seed

            # Evaluate each descriptor set generated
            for descriptor in descriptor_set:
                subgroup = extract_subgroup(descriptor, data, col_index_dict)  # Extract subgroup for the current descriptor
                if len(subgroup) >= subgroup_size and len(subgroup) < max_subgroup_size:  # Ensure subgroup is large enough and descriptor is not similar
                    targets_subgroup = [i[target_ind] for i in subgroup]  # Extract target values for the subgroup
                    quality_result = quality_measure(targets_subgroup, targets_baseline)  # Calculate quality measure
                    if not descriptors_similar_paper(quality_result, descriptor, results): # check if there are already subgroups with similar descriptors
                        put_item_in_queue(results, quality_result, tuple(descriptor), len(subgroup))  # Add to results queue
                        put_item_in_queue(beam, quality_result, tuple(descriptor))  # Add to the current beam

        # After processing the beam, update the beam queue with new combinations
        while not beam.empty():
            new_combination = beam.get()  # Get the highest quality descriptor from the beam
            new_combination = new_combination[1]  # Extract the descriptor from the tuple
            beam_queue.append(new_combination)  # Add it to the next depth of the beam search

    # Compile results into a list and reverse to have the best results first
    results_list = []
    while not results.empty():
        item = results.get()  # Get items from the results queue
        results_list.append(item)  # Add to the results list
    results_list.reverse()  # Reverse the list to have best results first

    return results_list  # Return the list of best descriptor sets

def descriptors_similar_paper(quality, descriptor1, pq):

    if len(descriptor1) == 1:
        return False

    tolerance = 0.1
    desc1_dict = {metric: (value, func) for metric, value, func in descriptor1}
    descriptor_list = get_all_descriptors(pq, 1)
    quality_list = get_all_descriptors(pq, 0)

    # Early exit if quality difference exceeds threshold
    if min(abs(quality - q) for q in quality_list) > 5:
        return False

    # Compare against each descriptor in the queue
    for descriptor2 in descriptor_list:

        desc2_dict = {metric: (value, func) for metric, value, func in descriptor2}

        match_count = 0

        for metric, (value1, func1) in desc1_dict.items():
            if metric not in desc2_dict:
                continue  # If a metric is missing, no need to continue

            value2, func2 = desc2_dict[metric]

            # Function mismatch, skip this descriptor
            if func1 != func2:
                continue

            # Numeric comparison within tolerance
            if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                if abs(value1 - value2) > abs(tolerance * value1):
                    continue  # Values out of tolerance range

            # String comparison for non-numeric values
            elif isinstance(value1, str) and value1 != value2:
                continue  # String values don't match

            match_count += 1

            # Early exit when all metrics except 1 match
            if match_count >= len(descriptor1)-1:
                return True

    return False



