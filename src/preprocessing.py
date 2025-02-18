from collections import defaultdict

import numpy as np
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


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

def preprocess_workload(workload_data):
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

def create_inputs_outputs_seperated(result):
    this fails key error
    workload_data = preprocess_workload(result['taskResults'])
    pods_data = preprocess_pods(result['scaleEvents'])
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
    for task_type, timestamp, workload_count, pod_count in inputs_outputs:
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
        params = {'objective': 'reg:squarederror', 'n_estimators': 100, 'seed': 42}  # Default parameters


    for task_type, data in train_data.items():
        # Extract features (timestamp, workload_count) and target (pod_count)
        X_train = [[item[0]] for item in data]  # Timestamp and workload
        y_train = [item[1] for item in data]  # Pod count


        # Convert lists to NumPy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)


        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        models[task_type] = model


    return models

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
            X_test = [[item[0]] for item in data]  # Only workload count
            y_test = [item[1] for item in data]  # Pod count


            # Convert lists to NumPy arrays
            X_test = np.array(X_test)
            y_test = np.array(y_test)


            y_pred = models[task_type].predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mse_scores[task_type] = mse
        else:
            mse_scores[task_type] = None  # Or some other indicator that there's no model


    return mse_scores
