import json
import math
from collections import defaultdict

import pandas as pd
import seaborn as sns
import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.model_selection import train_test_split

from src.motivational.constants import QUEUE_LENGTH
from src.motivationalhetero.encoders import PLATFORM_TYPES


def preprocess_pods(pods_data):
    """
    Processes pods data to create a dictionary of pod counts per second for each task type (name),
    implementing forward fill for missing values.

    Args:
    pods_data: A list of dictionaries, where each dictionary has keys 'name', 'timestamp', and 'count'.

    Returns:
    A dictionary where keys are task names and values are lists of pod counts per second, with forward fill applied.
    """
    pods_by_name = {}
    for item in pods_data:

        name = item['name']
        timestamp = int(item['timestamp'])  # Ensure time is an integer
        count = item['count']

        if name not in pods_by_name:
            pods_by_name[name] = {}

        pods_by_name[name][timestamp] = count

    # Apply forward fill
    result = {}
    for name, time_data in pods_by_name.items():
        sorted_timestamps = sorted(time_data.keys())
        min_time = sorted_timestamps[0]
        max_time = sorted_timestamps[-1]

        filled_data = {}
        last_count = 0
        for i in range(min_time, max_time + 1):
            if i in time_data:
                last_count = time_data[i]
            filled_data[i] = last_count
        result[name] = filled_data

    return result


def calculate_metrics_combined(workload_data, pods_data, application_to_task_map, window_size=60, until=None):
    """
    Calculates average throughput and pod count over time windows by processing workload and pods simultaneously.

    Args:
    workload_data: List of dicts with keys 'type', 'dispatchedTime', 'elapsedTime', 'penalty'
    pods_data: List of dicts with keys 'name', 'timestamp', 'count'
    window_size: Size of the time window in seconds (default: 60)

    Returns:
    Dictionary with metrics per task type containing windows with avg throughput and pod count
    """
    # Initialize data structures
    metrics = {}
    window_requests = {}
    window_queue_lengths = {}
    window_penalties = {}
    window_pods = {}

    # Process workload data
    for item in workload_data:
        task_type = application_to_task_map[item['type']][0]
        # Use completion time for throughput calculation
        completion_time = int(item['dispatchedTime'] + item['elapsedTime'])
        window_start = (completion_time // window_size) * window_size

        # Initialize dictionaries for new task types and windows
        if task_type not in metrics:
            metrics[task_type] = {}
            window_requests[task_type] = {}
            window_queue_lengths[task_type] = {}
            window_penalties[task_type] = {}
            window_pods[task_type] = {}

        if window_start not in window_requests[task_type]:
            window_requests[task_type][window_start] = 0
            window_penalties[task_type][window_start] = 0
            window_queue_lengths[task_type][window_start] = []
            window_pods[task_type][window_start] = []

        # Count requests and penalties
        window_requests[task_type][window_start] += 1
        if item['penalty']:
            window_penalties[task_type][window_start] += 1

    # Process pods data
    for pod in pods_data:
        task_type = pod['name']
        timestamp = int(pod['timestamp'])
        window_start = (timestamp // window_size) * window_size

        if task_type in window_pods and window_start in window_pods[task_type]:
            window_pods[task_type][window_start].append(pod['count'])
            window_queue_lengths[task_type][window_start].append(pod['average_queue_length'])
        elif task_type in window_pods:
            window_pods[task_type][window_start] = [pod['count']]
            window_queue_lengths[task_type][window_start] = [pod['average_queue_length']]

    # Calculate final metrics
    for task_type in metrics:
        for window_start in window_requests[task_type]:
            total_requests = window_requests[task_type][window_start]
            total_penalties = window_penalties[task_type][window_start]

            # Skip windows with high penalty rate
            if total_requests > 0 and (total_penalties / total_requests) <= 0.2:
                pod_counts = window_pods[task_type].get(window_start, [0])
                q_counts = window_queue_lengths[task_type].get(window_start, [0])

                avg_queue_length_window = sum(q_counts) / len(q_counts) if q_counts else 0
                if avg_queue_length_window > QUEUE_LENGTH or not pod_counts:
                    continue
                if until is not None and window_start + window_size >= until:
                    break
                metrics[task_type][window_start] = {
                    'window_start': window_start,
                    'window_end': window_start + window_size,
                    'avg_throughput': total_requests / window_size,
                    'avg_queue_length': avg_queue_length_window,
                    'avg_pods': sum(pod_counts) / len(pod_counts) if pod_counts else 0,
                    'total_requests': total_requests,
                    'penalty_rate': total_penalties / total_requests if total_requests > 0 else 0
                }

    return metrics


def calculate_metrics_combined_by_platform_type(workload_data, pods_data, application_to_task_map, window_size=60,
                                                until=None):
    """
    Calculates average throughput and pod count over time windows by processing workload and pods simultaneously,
    grouped by both task type and device type.

    Args:
    workload_data: List of dicts with keys 'type', 'dispatchedTime', 'elapsedTime', 'penalty', 'platform_type'
    pods_data: List of dicts with keys 'name', 'timestamp', 'count', 'platform_type'
    window_size: Size of the time window in seconds (default: 60)

    Returns:
    Dictionary with metrics per task type and device type containing windows with avg throughput and pod count
    """
    # Initialize data structures
    metrics = {}
    window_requests = {}
    window_queue_lengths = {}
    window_penalties = {}
    window_pods = {}

    # Process workload data
    for item in workload_data:
        task_type = application_to_task_map[item['type']][0]
        platform_type = item['platform_type']
        completion_time = int(item['dispatchedTime'] + item['elapsedTime'])
        window_start = (completion_time // window_size) * window_size

        # Initialize dictionaries for new task types, device types, and windows
        if task_type not in metrics:
            metrics[task_type] = {}
            window_requests[task_type] = {}
            window_queue_lengths[task_type] = {}
            window_penalties[task_type] = {}
            window_pods[task_type] = {}

        if platform_type not in metrics[task_type]:
            metrics[task_type][platform_type] = {}
            window_requests[task_type][platform_type] = {}
            window_queue_lengths[task_type][platform_type] = {}
            window_penalties[task_type][platform_type] = {}
            window_pods[task_type][platform_type] = {}

        if window_start not in window_requests[task_type][platform_type]:
            window_requests[task_type][platform_type][window_start] = 0
            window_penalties[task_type][platform_type][window_start] = 0
            window_queue_lengths[task_type][platform_type][window_start] = []
            window_pods[task_type][platform_type][window_start] = []

        # Count requests and penalties
        window_requests[task_type][platform_type][window_start] += 1
        if item['penalty']:
            window_penalties[task_type][platform_type][window_start] += 1

    # Process pods data
    for pod in pods_data:
        task_type = pod['name']
        timestamp = int(pod['timestamp'])
        for platform_type in PLATFORM_TYPES:

            count = pod.get(platform_type)
            if count is None:
                continue
            window_start = (timestamp // window_size) * window_size

            if task_type in window_pods and platform_type in window_pods[task_type] and window_start in \
                    window_pods[task_type][platform_type]:
                window_pods[task_type][platform_type][window_start].append(count)
                window_queue_lengths[task_type][platform_type][window_start].append(pod['average_queue_length'])
            elif task_type in window_pods and platform_type in window_pods[task_type]:
                window_pods[task_type][platform_type][window_start] = [count]
                window_queue_lengths[task_type][platform_type][window_start] = [pod['average_queue_length']]

    # Calculate final metrics
    for task_type in metrics:
        for platform_type in metrics[task_type]:
            for window_start in window_requests[task_type][platform_type]:
                total_requests = window_requests[task_type][platform_type][window_start]
                total_penalties = window_penalties[task_type][platform_type][window_start]

                # Skip windows with high penalty rate
                penalty_proportion = total_penalties / total_requests
                if total_requests > 0 and penalty_proportion <= 0.4:
                    pod_counts = window_pods[task_type][platform_type].get(window_start, [0])
                    q_counts = window_queue_lengths[task_type][platform_type].get(window_start, [0])

                    avg_queue_length_window = sum(q_counts) / len(q_counts) if q_counts else 0
                    if avg_queue_length_window > QUEUE_LENGTH * 1.25 or not pod_counts:
                        continue
                    if until is not None and window_start + window_size >= until:
                        break
                    metrics[task_type][platform_type][window_start] = {
                        'window_start': window_start,
                        'window_end': window_start + window_size,
                        'avg_throughput': total_requests / window_size,
                        'avg_queue_length': avg_queue_length_window,
                        'avg_pods': sum(pod_counts) / len(pod_counts) if pod_counts else 0,
                        'total_requests': total_requests,
                        'penalty_rate': penalty_proportion if total_requests > 0 else 0
                    }

    return metrics


def preprocess_workload_task_results(workload_data):
    """
    Processes workload data to create a dictionary of request counts per second for each task type.

    Args:
    workload_data: A list of dictionaries, where each dictionary has keys 'taskType' and 'dispatchedTime'.

    Returns:
    A dictionary where keys are task types and values are lists of request counts per second.
    """
    workload_by_task = {}
    for item in workload_data:
        task_type = item['taskType']['name']
        dispatched_time = int(item['dispatchedTime'])  # Ensure time is an integer

        if task_type not in workload_by_task:
            workload_by_task[task_type] = {}

        if dispatched_time not in workload_by_task[task_type]:
            workload_by_task[task_type][dispatched_time] = 0

        workload_by_task[task_type][dispatched_time] += 1
    return workload_by_task

    # # Convert to list of request counts per second, filling missing seconds with 0
    # result = {}
    # for task_type, time_data in workload_by_task.items():
    #     min_time = 0
    #     max_time = max(time_data.keys())

    #     requests_per_second = [0] * (max_time - min_time + 1)
    #     for time, count in time_data.items():
    #         requests_per_second[time - min_time] = count

    #     last_count = requests_per_second[0]  # Initialize with 0, assuming no pods initially

    #     for i in range(max_time - min_time + 1):
    #         current_time = min_time + i
    #         # print(current_time)
    #         if current_time in time_data:
    #             last_count = time_data[current_time]  # Update last_count if a value exists
    #         requests_per_second[i] = last_count  # Assign the last_count (either new or previous)

    #     result[task_type] = requests_per_second


# def preprocess_workload_app_results(workload_data):
#     """
#     Processes workload data to create a dictionary of request counts per second for each task type.
#
#     Args:
#     workload_data: A list of dictionaries, where each dictionary has keys 'taskType' and 'dispatchedTime'.
#
#     Returns:
#     A dictionary where keys are task types and values are lists of request counts per second.
#     """
#     workload_by_task = {}
#     penalized_timestamps = {}
#     for item in workload_data:
#         task_type = item['type']
#         # if item['penalty']:
#         #     continue
#         dispatched_time = item['dispatchedTime'] + item['elapsedTime'] - item['communicationsTime'] / 2 - item['executionTime']  # Ensure time is an integer
#         dispatched_time = int(item['dispatchedTime'])
#         if penalized_timestamps.get(dispatched_time) == True:
#             continue
#         if task_type not in workload_by_task:
#             workload_by_task[task_type] = {}
#
#         if dispatched_time not in workload_by_task[task_type]:
#             workload_by_task[task_type][dispatched_time] = 0
#         if item['penalty']:
#             del workload_by_task[task_type][dispatched_time]
#             penalized_timestamps[dispatched_time] = True
#         else:
#             workload_by_task[task_type][dispatched_time] += 1
#     return workload_by_task

def preprocess_workload_app_results(workload_data):
    """
    Processes workload data to create a dictionary of request counts per second for each task type,
    penalizing timestamps where penalties exceed 10% of requests.

    Args:
    workload_data: A list of dictionaries containing task data with type, dispatch time, and penalty info.

    Returns:
    A dictionary where keys are task types and values are dictionaries of request counts per timestamp.
    """
    workload_by_task = {}
    timestamp_penalties = {}
    timestamp_totals = {}

    # First pass: Count total requests and penalties per timestamp
    for item in workload_data:
        dispatched_time = int(item['dispatchedTime'] + item['elapsedTime'])
        # dispatched_time = int(item['dispatchedTime'])
        if dispatched_time not in timestamp_penalties:
            timestamp_penalties[dispatched_time] = 0
            timestamp_totals[dispatched_time] = 0
        timestamp_totals[dispatched_time] += 1
        if item['penalty']:
            timestamp_penalties[dispatched_time] += 1

    # Second pass: Process requests, excluding penalized timestamps
    for item in workload_data:
        task_type = item['type']
        dispatched_time = int(item['dispatchedTime'] + item['elapsedTime'])
        # dispatched_time = int(item['dispatchedTime'])

        # Skip if timestamp has >10% penalties
        if (timestamp_penalties[dispatched_time] / timestamp_totals[dispatched_time]) == 0:
            continue

        if task_type not in workload_by_task:
            workload_by_task[task_type] = {}

        if dispatched_time not in workload_by_task[task_type]:
            workload_by_task[task_type][dispatched_time] = 0

        if not item['penalty']:
            workload_by_task[task_type][dispatched_time] += 1

    return workload_by_task

    # # Convert to list of request counts per second, filling missing seconds with 0
    # result = {}
    # for task_type, time_data in workload_by_task.items():
    #     min_time = 0
    #     max_time = max(time_data.keys())

    #     requests_per_second = [0] * (max_time - min_time + 1)
    #     for time, count in time_data.items():
    #         requests_per_second[time - min_time] = count

    #     last_count = requests_per_second[0]  # Initialize with 0, assuming no pods initially

    #     for i in range(max_time - min_time + 1):
    #         current_time = min_time + i
    #         # print(current_time)
    #         if current_time in time_data:
    #             last_count = time_data[current_time]  # Update last_count if a value exists
    #         requests_per_second[i] = last_count  # Assign the last_count (either new or previous)

    #     result[task_type] = requests_per_second


def create_inputs_outputs(workload_data, pods_data):
    """
    Creates input (workload) and output (pod count) pairs, aligning them by task type and timestamp.


    Args:
    workload_data: Dictionary of workload data (output of preprocess_workload).
    pods_data: Dictionary of pods data (output of preprocess_pods).


    Returns:
    A list of tuples, where each tuple contains (task_type, timestamp, workload_count, pod_count).
    """
    inputs_outputs = []
    for task_type, workload in workload_data.items():
        if task_type in pods_data:
            pods = pods_data[task_type]

            # Find common timestamps
            common_timestamps = sorted(set(workload.keys()) & set(pods.keys()))

            # Create input/output pairs for common timestamps
            for timestamp in common_timestamps:
                inputs_outputs.append((task_type, timestamp, workload[timestamp], pods[timestamp]))

    return inputs_outputs


def create_inputs_outputs_seperated_per_task(result):
    workload_data = preprocess_workload_task_results(result['stats']['taskResults'])
    pods_data = preprocess_pods(result['stats']['scaleEvents'])
    inputs = defaultdict(list)
    outputs = defaultdict(list)
    for task_type, workload in workload_data.items():
        if task_type in pods_data:
            pods = pods_data[task_type]

            # Find common timestamps
            common_timestamps = sorted(set(workload.keys()) & set(pods.keys()))

            # Create input/output pairs for common timestamps
            for timestamp in common_timestamps:
                inputs[task_type].append(workload[timestamp])
                outputs[task_type].append(pods[timestamp])

    return np.array(inputs), np.array(outputs)


def create_inputs_outputs_seperated_per_app(result):
    workload_data = preprocess_workload_task_results(result['stats']['applicationResults'])
    pods_data = preprocess_pods(result['stats']['scaleEvents'])
    inputs = defaultdict(list)
    outputs = defaultdict(list)
    for task_type, workload in workload_data.items():
        if task_type in pods_data:
            pods = pods_data[task_type]

            # Find common timestamps
            common_timestamps = sorted(set(workload.keys()) & set(pods.keys()))

            # Create input/output pairs for common timestamps
            for timestamp in common_timestamps:
                inputs[task_type].append(workload[timestamp])
                outputs[task_type].append(pods[timestamp])

    return np.array(inputs), np.array(outputs)


def preprocess_system_events(pods_data):
    """
    Processes pods data to create a dictionary of pod counts per second for each task type (name),
    implementing forward fill for missing values.

    Args:
    pods_data: A list of dictionaries, where each dictionary has keys 'name', 'timestamp', and 'count'.

    Returns:
    A dictionary where keys are task names and values are lists of pod counts per second, with forward fill applied.
    """
    pods_by_name = {}
    average_queue_length_by_name = {}
    for item in pods_data:

        name = item['name']
        timestamp = int(item['timestamp'])  # Ensure time is an integer
        count = item['count']
        average_queue_length = item['average_queue_length']

        if name not in pods_by_name:
            pods_by_name[name] = {}
            average_queue_length_by_name[name] = {}

        pods_by_name[name][timestamp] = count
        average_queue_length_by_name[name][timestamp] = average_queue_length

    return pods_by_name, average_queue_length_by_name


# def create_inputs_outputs_seperated_per_app_windowed(result, window_size, app_definitions):
#     """
#     Aggregates workload and pod data into time windows.
#
#
#     Args:
#     result: A dictionary containing 'stats' with 'applicationResults' and 'scaleEvents'.
#     window_size: The size of the time window (number of timestamps to aggregate).
#
#
#     Returns:
#     A tuple of dictionaries (inputs, outputs), where each dictionary has task types as keys and lists of aggregated
#     workload sums and maximum pod counts as values.  Returns NumPy arrays for compatibility with XGBoost.
#     """
#     workload_data = preprocess_workload_app_results(result['applicationResults'])
#     pods_data = preprocess_pods(result['scaleEvents'])
#     inputs = defaultdict(list)
#     outputs = defaultdict(list)
#
#
#     for app_type, workload in workload_data.items():
#         for task_type in app_definitions[app_type]:
#             if task_type in pods_data:
#                 pods = pods_data[task_type]
#
#
#                 # Find common timestamps and sort them
#                 common_timestamps = sorted(set(workload.keys()) & set(pods.keys()))
#
#
#                 # Aggregate into time windows
#                 for i in range(0, len(common_timestamps) - window_size + 1, window_size):  # Step by window_size
#                     window_timestamps = common_timestamps[i:i + window_size]  # Get timestamps for the current window
#
#
#                     # Sum workload for the window
#                     workload_sum = math.ceil(sum(workload[ts] for ts in window_timestamps) / len(window_timestamps))
#                     if workload_sum == 0:
#                         continue
#
#                     # Find maximum pod count for the window
#                     pod_max = max(pods[ts] for ts in window_timestamps)
#
#
#                     inputs[task_type].append(workload_sum)
#                     outputs[task_type].append(pod_max)
#
#
#     # Convert to NumPy arrays
#     for task_type in inputs.keys():
#         inputs[task_type] = np.array(inputs[task_type])
#         outputs[task_type] = np.array(outputs[task_type])
#
#
#     return inputs, outputs

def create_inputs_outputs_seperated_per_app_windowed(result, window_size, app_definitions):
    """
    Aggregates workload and pod data into time windows using metrics from calculate_metrics_combined.

    Args:
    result: A dictionary containing 'stats' with 'applicationResults' and 'scaleEvents'.
    window_size: The size of the time window in seconds.
    app_definitions: Dictionary mapping app types to task types.

    Returns:
    A tuple of dictionaries (inputs, outputs) with NumPy arrays of workload and pod metrics.
    """
    metrics = calculate_metrics_combined(result['applicationResults'], result['systemEvents'], app_definitions,
                                         window_size)
    inputs = defaultdict(list)
    outputs = defaultdict(list)

    for app_type, app_tasks in app_definitions.items():
        for task_type in app_tasks:
            if task_type in metrics:
                for window_data in metrics[task_type].values():
                    # Skip windows with zero throughput
                    if window_data['avg_throughput'] == 0:
                        continue

                    # Convert throughput to requests per window
                    workload_sum = window_data['avg_throughput']
                    pod_count = math.ceil(window_data['avg_pods'])

                    inputs[task_type].append(workload_sum)
                    outputs[task_type].append(pod_count)

    # Convert to NumPy arrays
    for task_type in inputs.keys():
        inputs[task_type] = np.array(inputs[task_type])
        outputs[task_type] = np.array(outputs[task_type])

    print(outputs)
    return inputs, outputs


def create_inputs_outputs_seperated_per_app_windowed(result, window_size, app_definitions, until=None):
    """
    Aggregates workload and pod data into time windows using metrics from calculate_metrics_combined.

    Args:
    result: A dictionary containing 'stats' with 'applicationResults' and 'scaleEvents'.
    window_size: The size of the time window in seconds.
    app_definitions: Dictionary mapping app types to task types.

    Returns:
    A tuple of dictionaries (inputs, outputs) with NumPy arrays of workload and pod metrics.
    """
    metrics = calculate_metrics_combined(result['applicationResults'], result['systemEvents'], app_definitions,
                                         window_size, until)
    inputs = defaultdict(list)
    outputs = defaultdict(list)

    for app_type, app_tasks in app_definitions.items():
        for task_type in app_tasks:
            if task_type in metrics:
                for window_data in metrics[task_type].values():
                    # Skip windows with zero throughput
                    if window_data['avg_throughput'] == 0:
                        continue

                    # Convert throughput to requests per window
                    workload_sum = window_data['avg_throughput']
                    pod_count = math.ceil(window_data['avg_pods'])

                    inputs[task_type].append(workload_sum)
                    outputs[task_type].append(pod_count)

    # Convert to NumPy arrays
    for task_type in inputs.keys():
        inputs[task_type] = np.array(inputs[task_type])
        outputs[task_type] = np.array(outputs[task_type])

    print(outputs)
    return inputs, outputs


from collections import defaultdict
import numpy as np
import math


def create_inputs_outputs_seperated_per_app_windowed_per_device_type(result, window_size, app_definitions, until=None):
    """
    Aggregates workload and pod data into time windows using metrics from calculate_metrics_combined,
    separated by application type, task type, and device type.

    Args:
    result: A dictionary containing 'stats' with 'applicationResults' and 'systemEvents'.
    window_size: The size of the time window in seconds.
    app_definitions: Dictionary mapping app types to task types.
    until: Optional end time for processing.

    Returns:
    A tuple of dictionaries (inputs, outputs) with NumPy arrays of workload and pod metrics,
    grouped by task type and device type.
    """
    metrics = calculate_metrics_combined_by_platform_type(result['applicationResults'], result['systemEvents'],
                                                          app_definitions,
                                                          window_size, until)
    inputs = defaultdict(lambda: defaultdict(list))
    outputs = defaultdict(lambda: defaultdict(list))

    for app_type, app_tasks in app_definitions.items():
        for task_type in app_tasks:
            if task_type in metrics:
                for device_type, device_metrics in metrics[task_type].items():
                    for window_data in device_metrics.values():
                        # Skip windows with zero throughput
                        if window_data['avg_throughput'] == 0:
                            continue

                        # Convert throughput to requests per window
                        workload_sum = window_data['avg_throughput']
                        pod_count = math.ceil(window_data['avg_pods'])

                        inputs[task_type][device_type].append(workload_sum)
                        outputs[task_type][device_type].append(pod_count)

    # Convert to NumPy arrays
    for task_type in inputs.keys():
        for device_type in inputs[task_type].keys():
            inputs[task_type][device_type] = np.array(inputs[task_type][device_type])
            outputs[task_type][device_type] = np.array(outputs[task_type][device_type])

    return inputs, outputs


def create_inputs_outputs_seperated_per_app_windowed_system_events(result, window_size, app_definitions):
    """
    Aggregates workload and pod data into time windows.


    Args:
    result: A dictionary containing 'stats' with 'applicationResults' and 'scaleEvents'.
    window_size: The size of the time window (number of timestamps to aggregate).


    Returns:
    A tuple of dictionaries (inputs, outputs), where each dictionary has task types as keys and lists of aggregated
    workload sums and maximum pod counts as values.  Returns NumPy arrays for compatibility with XGBoost.
    """
    workload_data = preprocess_workload_app_results(result['applicationResults'])
    pods_data, queue_data = preprocess_system_events(result['systemEvents'])
    inputs = defaultdict(list)
    outputs = defaultdict(list)

    for app_type, workload in workload_data.items():
        for task_type in app_definitions[app_type]:
            if task_type in pods_data:
                pods = pods_data[task_type]
                qs = queue_data[task_type]

                # Find common timestamps and sort them
                common_timestamps = sorted(set(workload.keys()) & set(pods.keys()))

                # Aggregate into time windows
                for i in range(0, len(common_timestamps) - window_size + 1, window_size):  # Step by window_size
                    window_timestamps = common_timestamps[i:i + window_size]  # Get timestamps for the current window

                    # Sum workload for the window
                    workload_sum = math.ceil(sum(workload[ts] for ts in window_timestamps))
                    if workload_sum == 0:
                        continue

                    # Find maximum queue length for the window
                    q_max = max(qs[ts] for ts in window_timestamps)
                    # Find maximum pod count for the window
                    pod_max = max(pods[ts] for ts in window_timestamps)

                    inputs[task_type].append([workload_sum, q_max])
                    outputs[task_type].append(pod_max)

    # Convert to NumPy arrays
    for task_type in inputs.keys():
        inputs[task_type] = np.array(inputs[task_type])
        outputs[task_type] = np.array(outputs[task_type])

    return inputs, outputs


def create_inputs_outputs_seperated_per_app_windowed_system_events_per_device_type(result, window_size,
                                                                                   app_definitions):
    """
    Aggregates workload and pod data into time windows.


    Args:
    result: A dictionary containing 'stats' with 'applicationResults' and 'scaleEvents'.
    window_size: The size of the time window (number of timestamps to aggregate).


    Returns:
    A tuple of dictionaries (inputs, outputs), where each dictionary has task types as keys and lists of aggregated
    workload sums and maximum pod counts as values.  Returns NumPy arrays for compatibility with XGBoost.
    """
    workload_data = preprocess_workload_app_results(result['applicationResults'])
    pods_data, queue_data = preprocess_system_events(result['systemEvents'])
    inputs = defaultdict(list)
    outputs = defaultdict(list)

    for app_type, workload in workload_data.items():
        for task_type in app_definitions[app_type]:
            if task_type in pods_data:
                pods = pods_data[task_type]
                qs = queue_data[task_type]

                # Find common timestamps and sort them
                common_timestamps = sorted(set(workload.keys()) & set(pods.keys()))

                # Aggregate into time windows
                for i in range(0, len(common_timestamps) - window_size + 1, window_size):  # Step by window_size
                    window_timestamps = common_timestamps[i:i + window_size]  # Get timestamps for the current window

                    # Sum workload for the window
                    workload_sum = math.ceil(sum(workload[ts] for ts in window_timestamps))
                    if workload_sum == 0:
                        continue

                    # Find maximum queue length for the window
                    q_max = max(qs[ts] for ts in window_timestamps)
                    # Find maximum pod count for the window
                    pod_max = max(pods[ts] for ts in window_timestamps)

                    inputs[task_type].append([workload_sum, q_max])
                    outputs[task_type].append(pod_max)

    # Convert to NumPy arrays
    for task_type in inputs.keys():
        inputs[task_type] = np.array(inputs[task_type])
        outputs[task_type] = np.array(outputs[task_type])

    return inputs, outputs


def create_train_test_split_per_windowed(inputs_outputs, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets, ensuring a split *within* each task type.


    Args:
    inputs_outputs: A list of tuples (task_type, timestamp, workload_count, pod_count).
    test_size: The proportion of data to use for testing.
    random_state: Random seed for reproducibility.


    Returns:
    A tuple of dictionaries: (train_data, test_data)
    Each dictionary has task types as keys and a list of (timestamp, workload_count, pod_count) tuples as values.
    """
    train_data = {}
    test_data = {}

    # Split data for each task type
    for task_type, data in inputs_outputs[0].items():

        workload_counts = data
        pod_counts = inputs_outputs[1][task_type]
        print(len(pod_counts))
        print(len(workload_counts))

        if len(pod_counts) == 0 or len(workload_counts) == 0:
            return {}, {}

        if len(pod_counts) == 1 and len(workload_counts) == 1:
            # Handle the single sample case manually
            X_train, y_train = workload_counts, pod_counts
            X_test, y_test = np.empty(shape=(0, 1)), np.empty(shape=(0,))
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                workload_counts, pod_counts, test_size=test_size, random_state=random_state, train_size=1
            )

        if len(X_train.shape) == 1:
            X_train = X_train.reshape(-1, 1)
            X_test = X_test.reshape(-1, 1)
        train_data[task_type] = list(zip(X_train, y_train))
        test_data[task_type] = list(zip(X_test, y_test))

    return train_data, test_data


def create_train_test_split_per_windowed_per_device_type(inputs_outputs, test_size=0.2, random_state=42, encoder=None):
    """
    Splits the data into training and testing sets, ensuring a split *within* each task type and device type.
    One-hot encodes the device type and includes it as part of the input features.

    Args:
    inputs_outputs: A tuple of dictionaries (inputs, outputs) where each dictionary has task types as keys
                   and a nested dictionary with device types as keys and NumPy arrays as values.
    test_size: The proportion of data to use for testing.
    random_state: Random seed for reproducibility.
    encoder: OneHotEncoder instance for encoding device types. If None, a new encoder will be created.

    Returns:
    A tuple of dictionaries: (train_data, test_data)
    Each dictionary has task types as keys and a list of (features, pod_count) tuples as values,
    where features include the workload count and one-hot encoded device type.
    """
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    import numpy as np

    train_data = {}
    test_data = {}

    # Create or use the provided encoder
    if encoder is None:
        # Get all unique device types across all task types
        all_device_types = []
        for task_type in inputs_outputs[0]:
            all_device_types.extend(list(inputs_outputs[0][task_type].keys()))
        all_device_types = list(set(all_device_types))

        # Create and fit the encoder
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit([[device_type] for device_type in all_device_types])

    # Split data for each task type
    for task_type in inputs_outputs[0]:
        train_data[task_type] = []
        test_data[task_type] = []
        first = True
        # Process each device type for this task type
        for device_type, workload_counts in inputs_outputs[0][task_type].items():
            # if not first and device_type != 'xavierGpu':
            #     continue
            # if first and device_type != 'xavierGpu':
            #     pod_counts = inputs_outputs[1][task_type][device_type][:2]
                # continue
            # else:
            pod_counts = inputs_outputs[1][task_type][device_type]
            first = False


            if len(pod_counts) == 0 or len(workload_counts) == 0:
                continue

            print(device_type)
            print(workload_counts, pod_counts)
            # One-hot encode the device type
            device_encoded = encoder.transform([[device_type]])

            if len(pod_counts) == 1 and len(workload_counts) == 1:
                # Handle the single sample case manually
                # Combine workload count with encoded device type
                X_train = np.column_stack((workload_counts, np.tile(device_encoded, (len(workload_counts), 1))))
                y_train = pod_counts
                X_test = np.empty(shape=(0, X_train.shape[1]))
                y_test = np.empty(shape=(0,))
            else:
                # Split the data
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    workload_counts, pod_counts, test_size=test_size, random_state=random_state
                )

                # Reshape if needed
                if len(X_train_raw.shape) == 1:
                    X_train_raw = X_train_raw.reshape(-1, 1)
                    X_test_raw = X_test_raw.reshape(-1, 1)

                # Combine workload count with encoded device type
                X_train = np.column_stack((X_train_raw, np.tile(device_encoded, (len(X_train_raw), 1))))
                X_test = np.column_stack((X_test_raw, np.tile(device_encoded, (len(X_test_raw), 1))))

            # Add to the appropriate data collection
            train_data[task_type].extend(list(zip(X_train, y_train)))
            test_data[task_type].extend(list(zip(X_test, y_test)))

    return train_data, test_data, encoder


def create_train_test_split_per_task(inputs_outputs, test_size=0.2, random_state=42):
    """
    Splits the data into training and testing sets, ensuring a split *within* each task type.


    Args:
    inputs_outputs: A list of tuples (task_type, timestamp, workload_count, pod_count).
    test_size: The proportion of data to use for testing.
    random_state: Random seed for reproducibility.


    Returns:
    A tuple of dictionaries: (train_data, test_data)
    Each dictionary has task types as keys and a list of (timestamp, workload_count, pod_count) tuples as values.
    """
    train_data = {}
    test_data = {}

    # Group data by task type
    grouped_data = {}
    for (task_type, timestamp), (workload_count, pod_count) in zip(inputs_outputs[0], inputs_outputs[1]):
        if task_type not in grouped_data:
            grouped_data[task_type] = []
        grouped_data[task_type].append((timestamp, workload_count, pod_count))

    # Split data for each task type
    for task_type, data in grouped_data.items():
        timestamps = [item[0] for item in data]
        workload_counts = [item[1] for item in data]
        pod_counts = [item[2] for item in data]

        X_train, X_test, y_train, y_test = train_test_split(
            workload_counts, pod_counts, test_size=test_size, random_state=random_state
        )

        train_data[task_type] = list(zip(X_train, y_train))
        test_data[task_type] = list(zip(X_test, y_test))

    return train_data, test_data


def train_xgboost_per_task(train_data, params=None):
    """Trains an XGBoost model for each task type.


    Args:
    train_data: Dictionary where keys are task types and values are lists of (timestamp, workload_count, pod_count) tuples.
    params: XGBoost parameters (optional). If None, default parameters are used.


    Returns:
    A dictionary where keys are task types and values are trained XGBoost models.
    """
    models = {}
    if params is None:
        params = {'objective': 'reg:squarederror', 'n_estimators': 200, 'seed': 435}  # Default parameters

    for task_type, data in train_data.items():
        # Extract features (timestamp, workload_count) and target (pod_count)
        X_train = [item[0] for item in data]  # Input
        y_train = [item[1] for item in data]  # Pod count

        # Convert lists to NumPy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        models[task_type] = model

    return models



def train_gpr_per_task(train_data, params=None):
    """Trains an XGBoost model for each task type.


    Args:
    train_data: Dictionary where keys are task types and values are lists of (timestamp, workload_count, pod_count) tuples.
    params: XGBoost parameters (optional). If None, default parameters are used.


    Returns:
    A dictionary where keys are task types and values are trained XGBoost models.
    """
    models = {}
    X_scalers = {}
    y_scalers = {}
    for task_type, data in train_data.items():
        # Extract features (timestamp, workload_count) and target (pod_count)
        X_train = [item[0] for item in data]  # Input
        y_train = [item[1] for item in data]  # Pod count

        # Convert lists to NumPy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        from sklearn.preprocessing import StandardScaler

        # Create scalers
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()

        # Transform training data
        X_train_scaled = X_train.copy()
        
        visualize(X_train, y_train)
        # Scale only the first feature (column)
        # Extract the first column, reshape it to 2D array, scale it, and flatten back
        X_train_scaled[:, 0] = X_scaler.fit_transform(X_train[:, 0].reshape(-1, 1)).flatten()

        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

        X_scalers[task_type] = X_scaler
        y_scalers[task_type] = y_scaler

        # Define kernel (RBF kernel is good for smooth functions)
        # kernel = ConstantKernel() * RBF()
        from sklearn.gaussian_process.kernels import Matern
        kernel = ConstantKernel() * Matern(length_scale=1.0, nu=2)

        # Create and train the GP model
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, alpha=1e-1 )

        model.fit(X_train_scaled, y_train_scaled)
        models[task_type] = model

    return models, X_scalers, y_scalers


def visualize(X_train, y_train):
    X = X_train[:, 0].reshape(-1, 1)
    df = pd.DataFrame({'x': [x[0] for x in X], 'y': y_train})
    sns.scatterplot(x='x',y='y',data=df)
    plt.show()
    plt.close()


def evaluate_xgboost_per_task(models, test_data):
    """Evaluates the trained XGBoost models on the test data.


    Args:
    models: Dictionary where keys are task types and values are trained XGBoost models.
    test_data: Dictionary where keys are task types and values are lists of (timestamp, workload_count, pod_count) tuples.


    Returns:
    A dictionary where keys are task types and values are the Mean Squared Error (MSE) on the test data.
    """
    mse_scores = {}

    for task_type, data in test_data.items():
        if task_type in models:  # Make sure we have a model for this task type
            # Extract features (workload_count) and target (pod_count)
            X_test = [item[0] for item in data]  # Only workload count
            y_test = [item[1] for item in data]  # Pod count

            # Convert lists to NumPy arrays
            X_test = np.array(X_test)
            y_test = np.array(y_test)

            y_pred = models[task_type].predict(X_test)
            mse = root_mean_squared_error(y_test, y_pred)
            mse_scores[task_type] = mse
        else:
            mse_scores[task_type] = None  # Or some other indicator that there's no model

    return mse_scores


def evaluate_gpr_per_task(models, test_data, X_scalers, y_scalers):
    """Evaluates the trained GPR models on the test data and analyzes uncertainty correlation with errors.

    Args:
    models: Dictionary where keys are task types and values are trained GPR models.
    test_data: Dictionary where keys are task types and values are lists of (timestamp, workload_count, pod_count) tuples.
    X_scalers: Dictionary of feature scalers per task type
    y_scalers: Dictionary of target scalers per task type

    Returns:
    A dictionary with evaluation metrics including MSE and uncertainty correlation.
    """
    results = {}

    for task_type, data in test_data.items():
        if task_type in models:  # Make sure we have a model for this task type
            # Extract features (workload_count) and target (pod_count)
            X_test = np.array([item[0] for item in data])  # Only workload count
            y_test = np.array([item[1] for item in data])  # Pod count

            # Transform test data
            X_test_scaled = X_test.copy()
            X_test_scaled[:, 0] = X_scalers[task_type].fit_transform(X_test[:, 0].reshape(-1, 1)).flatten()

            # Predict with uncertainty
            mean_scaled, std_scaled = models[task_type].predict(X_test_scaled, return_std=True)

            # Transform predictions back to original scale
            mean_prediction = y_scalers[task_type].inverse_transform(mean_scaled.reshape(-1, 1)).ravel()
            std_prediction = std_scaled * y_scalers[task_type].scale_

            # Calculate absolute errors
            abs_errors = np.abs(y_test - mean_prediction)

            # Calculate RMSE
            rmse = root_mean_squared_error(y_test, mean_prediction)

            # Calculate correlation between uncertainty (std_prediction) and absolute errors
            uncertainty_error_correlation = np.corrcoef(std_prediction, abs_errors)[0, 1]

            # Calculate additional metrics for uncertainty evaluation
            # 1. Calibration: percentage of true values within 2 std deviations (should be ~95%)
            within_2std = np.mean((y_test >= mean_prediction - 2*std_prediction) &
                                  (y_test <= mean_prediction + 2*std_prediction))

            # 2. Average uncertainty
            avg_uncertainty = np.mean(std_prediction)

            # 3. Uncertainty vs. distance from training data
            # This assumes X_train is available - you might need to store it during training
            # or pass it as an argument

            # Store all metrics
            results[task_type] = {
                'rmse': rmse,
                'uncertainty_error_correlation': uncertainty_error_correlation,
                'calibration_2std': within_2std,
                'avg_uncertainty': avg_uncertainty,
                'std_prediction': std_prediction,  # Store for further analysis
                'abs_errors': abs_errors,  # Store for further analysis
                'mean_prediction': mean_prediction,
                'y_test': y_test,
                'X_test': X_test
            }

            # Print summary
            print(f"Task: {task_type}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  Uncertainty-Error Correlation: {uncertainty_error_correlation:.4f}")
            print(f"  Calibration (% within 2σ): {within_2std*100:.2f}%")
            print(f"  Average Uncertainty: {avg_uncertainty:.4f}")
            print(f"  Uncertainty Range: {np.min(std_prediction):.4f} - {np.max(std_prediction):.4f}")
            plot_uncertainty_error_correlation(results, task_type)
        else:
            results[task_type] = None  # Or some other indicator that there's no model


    return results

# Add a function to visualize the uncertainty-error relationship
def plot_uncertainty_error_correlation(evaluation_results, task_type):
    """
    Visualizes the relationship between prediction uncertainty and actual errors.

    text
    Args:
    evaluation_results: Results dictionary from evaluate_gpr_per_task
    task_type: The specific task to visualize
    """
    if task_type not in evaluation_results or evaluation_results[task_type] is None:
        print(f"No evaluation results for task type: {task_type}")
        return

    results = evaluation_results[task_type]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Scatter plot of uncertainty vs absolute error
    ax1.scatter(results['std_prediction'], results['abs_errors'], alpha=0.6)
    ax1.set_xlabel('Predicted Uncertainty (σ)')
    ax1.set_ylabel('Absolute Error')
    ax1.set_title(f'Uncertainty vs Error (r = {results["uncertainty_error_correlation"]:.4f})')

    # Add a trend line
    z = np.polyfit(results['std_prediction'], results['abs_errors'], 1)
    p = np.poly1d(z)
    ax1.plot(sorted(results['std_prediction']), p(sorted(results['std_prediction'])),
             "r--", linewidth=2)

    # Plot 2: Predictions with uncertainty bands
    # Sort by X for better visualization - extract only first element from each X instance
    X_first_element = np.array([x[0] for x in results['X_test']])  # Extract first element from each X
    sort_idx = np.argsort(X_first_element)

    # Apply sorting to all arrays
    X_first_sorted = X_first_element[sort_idx]
    y_sorted = results['y_test'][sort_idx]
    mean_sorted = results['mean_prediction'][sort_idx]
    std_sorted = results['std_prediction'][sort_idx]

    ax2.scatter(X_first_sorted, y_sorted, alpha=0.6, label='Actual')
    ax2.plot(X_first_sorted, mean_sorted, 'r-', label='Prediction')
    ax2.fill_between(X_first_sorted,
                     mean_sorted - 2*std_sorted,
                     mean_sorted + 2*std_sorted,
                     alpha=0.2, color='r', label='95% confidence')
    ax2.set_xlabel('Workload Count (First Feature)')
    ax2.set_ylabel('Pod Count')
    ax2.set_title(f'Predictions with Uncertainty Bands (Calibration: {results["calibration_2std"]*100:.2f}%)')
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Additional plot: Histogram of standardized errors
    plt.figure(figsize=(8, 6))
    standardized_errors = (results['y_test'] - results['mean_prediction']) / results['std_prediction']
    plt.hist(standardized_errors, bins=20, alpha=0.7, density=True)

    # Add normal distribution curve for comparison
    x = np.linspace(-4, 4, 100)
    plt.plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2)

    plt.xlabel('Standardized Error ((y_true - y_pred) / σ)')
    plt.ylabel('Density')
    plt.title('Standardized Error Distribution\n(Should follow standard normal if well-calibrated)')
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.close()

def main():
    with open('./result/20250219-123131-522218.json', 'r') as f:
        result = json.load(f)
        preprocess_pods(result['scaleEvents'])
        preprocess_workload_app_results(result['applicationResults'])
        app_definitions = {}
        for task in result['taskResults']:
            app_definitions[task['applicationType']['name']] = list(task['applicationType']['dag'].keys())
        print(create_inputs_outputs_seperated_per_app_windowed(result, 5, app_definitions))


if __name__ == '__main__':
    main()
