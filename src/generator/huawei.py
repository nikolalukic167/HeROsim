import datetime
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.generator.traces import compress_request_pattern_with_sampling, convert_datetime_timestamps, plot_pattern


def get_days_of_month_from_minutes(minutes_passed_tuple):
    # Convert each minute number to a day number (0-based)
    days = []
    for minutes_passed in minutes_passed_tuple:
        # Calculate how many days have passed (integer division)
        days_passed = minutes_passed // (24 * 60)
        days.append(days_passed)

    # Return unique days in order
    return sorted(list(set(days)))


def preprocess_day_file(file_path, fn):
    df = pd.read_csv(file_path)
    df = df.drop(['day'], axis=1)

    # Convert 'time' column to datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    # Split column names into function ID, user, and pool
    column_info = [col.split('---') for col in df.columns if '---' in col]
    new_columns = new_columns = [f"{func}" for func, _, _ in column_info]

    # Assign the new MultiIndex to the DataFrame columns
    df.columns = new_columns

    # Group by function ID and user, summing over pools
    grouped = df.groupby(df.columns, axis=1).sum().fillna(0)

    return grouped[[fn]]


def prepare_weekly_data(preprocessed_data):
    # Combine all days into a single DataFrame
    combined_df = pd.concat(preprocessed_data.values())

    # Ensure the data starts on a Monday
    start_date = combined_df.index.min()
    # while start_date.dayofweek != 0:  # 0 represents Monday
    #     start_date += pd.Timedelta(days=1)

    # Truncate data to start from the first Monday
    combined_df = combined_df.loc[start_date:]

    # Resample to minute frequency and fill missing values
    combined_df = combined_df.resample('T').asfreq().fillna(0)

    # Calculate the number of complete weeks
    minutes_per_week = 7 * 24 * 60
    total_minutes = len(combined_df)
    complete_weeks = total_minutes // minutes_per_week

    # Reshape the data into weekly chunks for each function-user group
    weekly_data = {}
    for column in combined_df.columns:
        series = combined_df[column].values[:complete_weeks * minutes_per_week]
        # weekly_data[column] = np.mean(series.reshape(complete_weeks, minutes_per_week), axis=0).reshape(1,-1)
        weekly_data[column] = series.reshape(complete_weeks, minutes_per_week)

    return weekly_data


def read_and_preprocess_files(directory, from_to, fn):
    preprocessed_data = {}
    for day_id in range(31):  # Assuming 7 days of data
        file_name = f'day_{day_id:02d}.csv'
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            preprocessed_data[day_id] = preprocess_day_file(file_path, fn)
        else:
            print(f"File not found: {file_name}")
    return preprocessed_data


def normalize_fn_patterns(df, fn):
    copy_df = df.copy()
    mean = np.mean(copy_df[fn])
    std = np.std(copy_df[fn])
    # Avoid division by zero
    if std == 0:
        copy_df[fn] = copy_df[fn] - mean
    else:
        copy_df[fn] = (copy_df[fn] - mean) / std
    return copy_df


# def create_minute_arrivals(series):
#     # Generate arrivals for each minute
#     all_arrivals = []
#     minutes = len(series)
#     base_time = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
#
#     for minute, minute_requests in enumerate(series):
#
#         # Generate arrivals for this minute
#         for i in range(int(minute_requests)):
#             # Allow some overlap with adjacent minutes for smoother transitions
#             minute_offset = random.gauss(0, 0.25)  # Small random offset, mostly within ±0.5 minute
#             actual_minute = max(0, min(minutes - 0.001, minute + minute_offset))
#
#             # Random position within the minute
#             position_in_minute = random.random()
#
#             # Calculate total seconds
#             seconds = actual_minute * 60 + position_in_minute * 60
#
#             # Add to arrivals list
#             all_arrivals.append(seconds)
#     all_arrivals.sort()
#
#     return [base_time + datetime.timedelta(seconds=t) for t in all_arrivals]

# For even better performance with very large series, consider parallel processing:
def process_chunk(chunk_data):
    minute_offset, chunk = chunk_data
    minutes = len(chunk)
    all_arrivals = []

    for minute, minute_requests in enumerate(chunk):
        requests = int(minute_requests)
        if requests == 0:
            continue

        for _ in range(requests):
            offset = random.gauss(0, 0.25)
            actual_minute = max(0, min(minutes - 0.001, minute + offset))
            position = random.random()
            seconds = (minute_offset + actual_minute) * 60 + position * 60
            all_arrivals.append(seconds)

    return all_arrivals


def create_minute_arrivals_parallel(series, chunk_size=1000):
    # For extremely large series, process in parallel
    base_time = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Split series into chunks
    chunks = []
    for i in range(0, len(series), chunk_size):
        chunks.append((i, series[i:i + chunk_size]))

    # Process chunks in parallel
    all_arrivals = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = executor.map(process_chunk, chunks)
        for result in results:
            all_arrivals.extend(result)

    all_arrivals.sort()
    return all_arrivals


import numpy as np
import datetime


def create_minute_arrivals_optimized(series):
    """
    Further optimized function to generate arrivals using fully vectorized operations.

    Parameters:
        series (np.array): Array of request counts per minute.

    Returns:
        list: List of datetime objects representing arrival times.
    """
    # Base time for the day
    base_time = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Convert series to numpy array if it's not already
    series = np.asarray(series)

    # Get indices of non-zero elements to avoid unnecessary calculations
    non_zero_indices = np.nonzero(series)[0]
    non_zero_counts = series[non_zero_indices]

    # Generate minute indices for all requests at once
    minute_indices = np.repeat(non_zero_indices, non_zero_counts.astype(int))

    # Generate all random values in one batch
    minute_offsets = np.random.normal(0, 0.25, size=len(minute_indices))
    positions_in_minute = np.random.random(size=len(minute_indices))

    # Calculate actual minutes with bounds checking
    actual_minutes = np.clip(minute_indices + minute_offsets, 0, len(series) - 0.001)

    # Calculate seconds for all arrivals at once
    seconds = actual_minutes * 60 + positions_in_minute * 60

    # Sort the seconds
    seconds.sort()

    return seconds


def read_huawei_dfs(days, time_window, fn, region):
    start = pd.to_datetime(time_window[0], unit='m')
    end = pd.to_datetime(time_window[1], unit='m')
    home = Path.home()
    directory = str(home / f'resources/datasets/huawei_public_cloud/{region}/requests')
    data = read_and_preprocess_files(directory, days, fn)
    combined_df = pd.concat(data.values()).fillna(0)
    mean = combined_df[fn].mean()
    std = combined_df[fn].std()
    combined_df = normalize_fn_patterns(combined_df, fn)
    combined_df = combined_df.loc[start:end]
    return combined_df, mean, std


def fetch_huawei_arrival_times(fn, region, time_window, simulation_duration, new_average_rps):
    days = get_days_of_month_from_minutes(time_window)
    df, mean, std = read_huawei_dfs(days, time_window, fn, region)
    new_std_rps = std * (new_average_rps / mean)
    df[fn] = (df[fn] * new_std_rps) + new_average_rps
    print("Starting create_minute_arrivals_parallel")
    minute_arrivals = create_minute_arrivals_optimized(df[fn])
    # plot_pattern(minute_arrivals, '60S', None)
    print("Compressing events")
    events = compress_request_pattern_with_sampling(minute_arrivals, original_duration=time_window[1] - time_window[0],
                                                    target_duration=simulation_duration)
    # plot_pattern(events, '60S', None)
    return events, new_std_rps


def freq():
    # Example: Replace this with your actual pandas Series (requests per minute)
    # Assume the data spans 4 weeks (10080 samples per week, total 40320 samples)
    date_range = pd.date_range(start="2025-02-03", periods=40320, freq="T")
    requests_per_minute = np.random.randint(50, 200, size=len(date_range))
    time_series = pd.Series(data=requests_per_minute, index=date_range)
    # time_series = df[fn]

    # Sampling rate: 1 sample per minute
    sampling_rate = 1 / 60  # Frequency in Hz (samples per second)

    def calculate_frequency_amplitude(time_series, sampling_rate):
        # Perform FFT
        fft_result = np.fft.fft(time_series.values)

        # Calculate amplitude (magnitude of FFT result)
        amplitude = np.abs(fft_result) / len(time_series)

        # Calculate frequencies
        frequencies = np.fft.fftfreq(len(time_series), d=1 / sampling_rate)

        # Only keep positive frequencies
        positive_frequencies = frequencies[:len(frequencies) // 2]
        positive_amplitudes = amplitude[:len(amplitude) // 2]

        return positive_frequencies, positive_amplitudes

    # Split the data into weekly segments
    weekly_data = [time_series[i:i + 10080] for i in range(0, len(time_series), 10080)]

    # Analyze each week and plot frequency vs amplitude
    for i, week_series in enumerate(weekly_data):
        freqs, amps = calculate_frequency_amplitude(week_series, sampling_rate)

        plt.figure(figsize=(12, 6))
        plt.plot(freqs, amps)
        plt.title(f"Frequency vs Amplitude for Week {i + 1}")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

    from sklearn.metrics.pairwise import cosine_similarity

    # Calculate amplitudes for all weeks
    amplitudes_all_weeks = [calculate_frequency_amplitude(week_series, sampling_rate)[1] for week_series in weekly_data]

    # Compute cosine similarity between weeks
    cosine_similarities = cosine_similarity(amplitudes_all_weeks[:4])

    print("Cosine Similarity Matrix:")
    print(cosine_similarities)

    from scipy.spatial.distance import euclidean

    # Compute pairwise Euclidean distances between weeks
    distances = []
    for i in range(4):
        for j in range(i + 1, 4):
            dist = euclidean(amplitudes_all_weeks[i], amplitudes_all_weeks[j])
            distances.append((f"Week {i + 1} vs Week {j + 1}", dist))

    print("Euclidean Distances Between Weeks:")
    for pair in distances:
        print(pair)


def main():
    fn = '49'
    region = 'R1'
    # first week
    for i in range(4):
        time_window = (10080 * i, 10080 * (i + 1))
        simulation_duration = 20
        new_average_rps = 7500
        arrivals, new_std_rps = fetch_huawei_arrival_times(fn, region, time_window, simulation_duration, new_average_rps)
        arrival_file = f'{region}-{fn}-{time_window[0]}-{time_window[1]}-{new_average_rps}-{int(new_std_rps)}-{simulation_duration}'
        plot_pattern(arrivals, 60, f'data/nofs-ids/arrivals/{arrival_file}')

        arrivals_timestamps = convert_datetime_timestamps(arrivals)
        with open(
                f'data/nofs-ids/arrivals/{arrival_file}.json',
                'w') as fd:
            json.dump(arrivals_timestamps, fd)


if __name__ == '__main__':
    main()
