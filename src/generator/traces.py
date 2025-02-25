"""
Copyright 2024 b<>com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import datetime
import random

from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.placement.model import (
    ApplicationType,
    QoSType,
    SimulationData,
    TimeSeries,
    WorkloadEvent,
)
import seaborn as sns


def poisson_process(lambd: int, duration_time: int) -> List[float]:
    arrivals: List[float] = []
    current_time = 0.0

    while current_time < duration_time:
        inter_arrival_time = random.expovariate(lambd)
        current_time += inter_arrival_time

        if current_time < duration_time:
            arrivals.append(current_time)

    return arrivals


def exponential_arrivals(mean_rate: float, duration: int) -> List[float]:
    arrivals = []
    current_time = 0.0
    while current_time < duration:
        inter_arrival = random.expovariate(1.0 / mean_rate)
        current_time += inter_arrival
        if current_time < duration:
            arrivals.append(current_time)
    return arrivals


def gamma_distribution(alpha: float, beta: float, size: int) -> List[float]:
    samples = []
    for _ in range(size):
        # Using sum of exponentials approximation
        sample = sum(random.expovariate(beta) for _ in range(int(alpha)))
        samples.append(sample)
    # Convert inter-arrival times to absolute arrival times
    arrival_times = [0]  # Start at time 0
    for delta in samples:
        arrival_times.append(arrival_times[-1] + delta)
    arrival_times = arrival_times[1:]  # Remove the initial 0
    return arrival_times


def bimodal_normal(mu1: float, sigma1: float, mu2: float, sigma2: float,
                   weight: float, size: int) -> List[float]:
    samples = []
    for _ in range(size):
        if random.random() < weight:
            samples.append(random.gauss(mu1, sigma1))
        else:
            samples.append(random.gauss(mu2, sigma2))
    return samples


def beta_distribution(alpha: float, beta: float, size: int) -> List[float]:
    samples = []
    for _ in range(size):
        x = random.random()
        y = random.random()
        if x + y <= 1:
            samples.append(x / (x + y))
    return samples[:size]


def uniform_arrivals(min_rate: float, max_rate: float, duration: int) -> List[float]:
    arrivals = []
    current_time = 0.0
    while current_time < duration:
        rate = random.uniform(min_rate, max_rate)
        inter_arrival = 1.0 / rate
        current_time += inter_arrival
        if current_time < duration:
            arrivals.append(current_time)
    return arrivals


def generate_time_series(
        data: SimulationData, rps: int, duration_time: int
) -> TimeSeries:
    # Generate Poisson process arrivals
    arrivals = poisson_process(rps, duration_time)

    events: List[WorkloadEvent] = []
    qos_levels_per_app = {}
    for application_type in data.application_types.keys():
        qos_type_count: int = len(data.qos_types)
        qos_type_index: int = random.randint(0, qos_type_count - 1)
        qos_type_name: str = list(data.qos_types)[qos_type_index]
        qos_type: QoSType = data.qos_types[qos_type_name]
        qos_levels_per_app[str(application_type)] = qos_type

    for timestamp in arrivals:
        application_type_count: int = len(data.application_types)
        application_type_index: int = random.randint(0, application_type_count - 1)
        application_type_name: str = list(data.application_types)[
            application_type_index
        ]
        application_type: ApplicationType = data.application_types[
            application_type_name
        ]

        workload_event: WorkloadEvent = {
            "timestamp": timestamp,
            "application": application_type,
            "qos": qos_levels_per_app[application_type_name],
        }

        events.append(workload_event)

    time_series = TimeSeries(rps=rps, duration=duration_time, events=events)

    return time_series


def generate_diurnal_pattern(hours=24, requests_per_hour=100):
    all_arrivals = []
    base_time = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    for hour in range(hours):
        # Model time-of-day variation (busier during work hours)
        if 9 <= hour <= 17:  # Business hours
            alpha = 1.0
            beta = 3.0  # Higher rate during business hours
        else:
            alpha = 1.0
            beta = 1.0  # Lower rate outside business hours

        # Generate inter-arrival times for this hour
        inter_arrivals = gamma_distribution(alpha, beta, requests_per_hour)

        # Convert to absolute times within this hour
        hour_start = hour * 3600  # seconds
        for i, delta in enumerate(inter_arrivals):
            position = hour_start + (i * 3600 / requests_per_hour) + delta
            all_arrivals.append(position)

    return [base_time + datetime.timedelta(seconds=t) for t in all_arrivals]

def compress_request_pattern_with_sampling(arrival_times, original_duration=60, target_duration=1):
    """
    Compress a request pattern from original_duration to target_duration minutes
    while maintaining the average requests per second.

    Parameters:
    - arrival_times: List of datetime objects representing arrival times
    - original_duration: Original duration in minutes (default: 60 for an hour)
    - target_duration: Target duration in minutes (default: 1 minute)

    Returns:
    - List of compressed datetime objects with preserved average RPS
    """
    # Convert to timestamps if they're datetime objects
    if isinstance(arrival_times[0], datetime.datetime):
        timestamps = [(t - arrival_times[0]).total_seconds() for t in arrival_times]
    else:
        timestamps = arrival_times.copy()

    # Get the original time range
    start_time = min(timestamps)
    end_time = max(timestamps)
    original_range = end_time - start_time

    # Calculate compression ratio for time
    time_compression_ratio = (target_duration * 60) / original_range

    # Calculate sampling ratio to maintain average RPS
    # We need to sample a fraction of events equal to the time compression ratio
    sampling_ratio = time_compression_ratio

    # Randomly sample events to maintain original RPS
    import random
    num_samples = max(1, int(len(timestamps) * sampling_ratio))
    sampled_timestamps = sorted(random.sample(timestamps, num_samples))

    # Scale each timestamp
    compressed_timestamps = []
    base_time = datetime.datetime.now()

    for ts in sampled_timestamps:
        # Scale the timestamp relative to start_time
        relative_time = ts - start_time
        compressed_time = relative_time * time_compression_ratio
        compressed_timestamps.append(base_time + datetime.timedelta(seconds=compressed_time))

    return compressed_timestamps

def main():
    average_rolling_window_duration = 60
    duration = 1200

    # arrival_times = uniform_arrivals(10, 60, 1200)

    # arrival_times = poisson_process(30, duration)

    arrival_times = gamma_distribution(1, 50, 50000)

    # Assuming you have your arrival_times as timestamps in seconds
    # Convert timestamps to datetime objects
    arrival_times_datetime = [datetime.datetime.fromtimestamp(ts) for ts in arrival_times]

    arrival_times_datetime = generate_diurnal_pattern(requests_per_hour=10 * 60 * 60)

    # Convert to pandas Series
    arrival_times_series = pd.Series(arrival_times_datetime)
    events = compress_request_pattern_with_sampling(arrival_times_series, original_duration=60 * 24, target_duration=20)

    # Count events per second by grouping
    events_per_second = pd.Series(events)
    events_per_second = events_per_second.groupby(events_per_second.dt.floor('1S')).size()

    # Calculate rolling average
    rolling_avg = events_per_second.rolling(window=average_rolling_window_duration, center=False).mean()

    # Plot both the events per second and the rolling average
    plt.figure(figsize=(12, 6))
    sns.set(style="whitegrid")

    # Plot original data
    sns.lineplot(x=events_per_second.index, y=events_per_second.values, label='Events per Second')

    # Plot rolling average
    sns.lineplot(x=rolling_avg.index, y=rolling_avg.values, label='60-Second Rolling Average', color='red')

    plt.title('Events Per Second Over Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Events')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
