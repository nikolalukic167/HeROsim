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
from typing import List, Dict

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.placement.model import (
    ApplicationType,
    QoSType,
    SimulationData,
    TimeSeries,
    WorkloadEvent,
)


def poisson_process(lambd: int, duration_time: int) -> List[float]:
    arrivals: List[float] = []
    current_time = 0.0

    while current_time < duration_time:
        inter_arrival_time = random.expovariate(lambd)
        current_time += inter_arrival_time

        if current_time < duration_time:
            arrivals.append(current_time)

    return arrivals


def ramp_up_poisson_process(lambd: int, duration_time: int, ramp_ups, ramp_up_time) -> List[float]:
    arrivals: List[float] = []
    current_time = 0.0
    ramp_up_rps = lambd / ramp_ups

    for i in range(ramp_ups):
        end_ramp_up = current_time + ramp_up_time
        while current_time < end_ramp_up:
            inter_arrival_time = random.expovariate(ramp_up_rps * (i + 1))
            current_time += inter_arrival_time

            if current_time < end_ramp_up:
                arrivals.append(current_time)

    end_time = duration_time + (ramp_ups * ramp_up_time)
    while current_time < end_time:
        inter_arrival_time = random.expovariate(lambd)
        current_time += inter_arrival_time

        if current_time < end_time:
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
        data: SimulationData,
        rps: int,
        duration_time: int,
        pattern: str,
        apps: [],
        pdf_path: str,
        peaks: List,
        config: Dict = None,
        app_weights: Dict[str, float] | None = None,
) -> TimeSeries:
    # Load client count from config if provided, otherwise use default
    if config and 'nodes' in config and 'client_nodes' in config['nodes']:
        n_clients = config['nodes']['client_nodes']['count']
    else:
        n_clients = 20  # Default fallback
    
    client_ids = []
    # Generate Poisson process arrivals
    if peaks is not None:
        arrivals = generate_diurnal_pattern(requests_per_hour=60 * 60 * rps, peaks=peaks)
        arrivals = compress_request_pattern_with_sampling(arrivals, 24 * 60, target_duration=round(duration_time / 60))
        plot_pattern(arrivals, 60, pdf_path)
        arrivals = convert_datetime_timestamps(arrivals)
    elif pattern == 'daily_delayed':
        arrivals = generate_diurnal_pattern_delayed(requests_per_hour=60 * 60 * rps)
        plot_pattern(arrivals, 60, f'{pdf_path}-24h')

        arrivals = compress_request_pattern_with_sampling(arrivals, 24 * 60, target_duration=round(duration_time / 60))
        plot_pattern(arrivals, 60, pdf_path)
        arrivals = convert_datetime_timestamps(arrivals)
    elif pattern == 'poisson-increasing':
        ramp_ups = 4
        ramp_up_time = 30
        arrivals = ramp_up_poisson_process(rps, duration_time, ramp_ups, ramp_up_time)
        arrival_times_datetime = [datetime.datetime.fromtimestamp(ts) for ts in arrivals]
        plot_pattern(arrival_times_datetime, 60, pdf_path)

    else:
        # todo: nikola map arrival time to clients
        # todo: have two specified distributions for the client nodes:
        # one for the client nodes and one for all clients
        # that way we can model congestion on specific clients and not just the average
        arrivals_with_clients = []
        for n in range(n_clients):
            client_arrivals = poisson_process(rps + random.randint(-5, 5), duration_time)
            offset = 0
            if n != 0:
                offset = random.uniform(0, 20)
            arrivals_with_clients.extend((ts + offset, n) for ts in client_arrivals)

        # Sort by arrival times
        arrivals_with_clients.sort()
        arrivals = [item[0] for item in arrivals_with_clients]
        client_ids = [item[1] for item in arrivals_with_clients]

        arrival_times_datetime = [datetime.datetime.fromtimestamp(ts) for ts in arrivals]
        plot_pattern(arrival_times_datetime, 60, pdf_path)

    events: List[WorkloadEvent] = []
    qos_levels_per_app = {}
    if apps is not None:
        if isinstance(apps, str):
            app_types = [apps]
        else:
            app_types = list(apps)
    else:
        app_types = list(data.application_types.keys())

    for application_type in app_types:
        qos_type_count: int = len(data.qos_types)
        qos_type_index: int = random.randint(0, qos_type_count - 1)
        qos_type_name: str = list(data.qos_types)[qos_type_index]
        qos_type: QoSType = data.qos_types[qos_type_name]
        qos_levels_per_app[str(application_type)] = qos_type

    weights = None
    if app_weights:
        weights = [max(0.0, float(app_weights.get(name, 0.0))) for name in app_types]
        if sum(weights) <= 0:
            weights = None

    for i, timestamp in enumerate(arrivals):
        if weights is None:
            application_type_name = random.choice(app_types)
        else:
            application_type_name = random.choices(app_types, weights=weights, k=1)[0]
        application_type = data.application_types[application_type_name]
        if len(client_ids) > 0:
            node_name = f"client_node{client_ids[i] % n_clients}"
        else:
            node_name = f"client_node{random.randint(0, n_clients - 1)}"
        workload_event: WorkloadEvent = {
            "timestamp": timestamp,
            "application": application_type,
            "qos": qos_levels_per_app[application_type_name],
            "node_name": node_name
        }

        events.append(workload_event)

    time_series = TimeSeries(rps=rps, duration=duration_time, events=events)

    return time_series


def generate_diurnal_paattern(hours=24, requests_per_hour=100):
    all_arrivals = []
    base_time = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    for hour in range(hours):
        # Model time-of-day variation (busier during work hours)
        if 12 <= hour <= 17:  # Business hours
            alpha = 1.0
            beta = 3.0  # Higher rate during business hours (3x more requests)
        elif 6 <= hour <= 11:
            alpha = 1.0
            beta = 1.5  # Higher rate during business hours (3x more requests)
        elif 18 <= hour <= 22:
            alpha = 1.0
            beta = 2  # Higher rate during business hours (3x more requests)
        else:
            alpha = 1.0
            beta = 1.0  # Lower rate outside business hours

        # Calculate requests for this hour based on time of day
        hour_requests = requests_per_hour * (beta / 2)  # Scale by beta/2 to create the pattern

        # Generate inter-arrival times for this hour
        inter_arrivals = gamma_distribution(alpha, beta, int(hour_requests))

        # Convert to absolute times within this hour
        hour_start = hour * 3600  # seconds
        for i, delta in enumerate(inter_arrivals):
            # Use modulo to wrap events within the hour
            position = hour_start + ((i * 3600 / hour_requests) + delta) % 3600
            all_arrivals.append(position)

    return [base_time + datetime.timedelta(seconds=t) for t in all_arrivals]


def generate_diurnal_pattern(hours=24, requests_per_hour=100, peaks=None):
    """
    Generate a diurnal pattern with custom peaks and smooth transitions.

    Parameters:
    - hours: Total duration to generate in hours
    - requests_per_hour: Average number of requests per hour
    - peaks: List of tuples (hour, intensity, width) where:
             - hour: Hour of the day for the peak (0-23)
             - intensity: Relative intensity of the peak (1.0 = baseline)
             - width: Width of the peak in hours (controls smoothness)

    Returns:
    - List of datetime objects representing arrival times
    """
    import numpy as np
    import random
    import datetime

    # Default peaks if none provided (business hours peak)
    if peaks is None:
        peaks = [(13, 3.0, 8)]  # Peak at 1 PM, 3x intensity, 8 hours wide

    base_time = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Define a smooth rate multiplier function based on custom peaks
    def rate_multiplier(hour):
        # Convert hour to continuous time (0-24)
        continuous_hour = hour % 24

        # Start with baseline intensity
        intensity = 1.0

        # Add contribution from each peak (using Gaussian-like function)
        for peak_hour, peak_intensity, peak_width in peaks:
            # Calculate distance from peak (accounting for day wrap-around)
            dist = min(abs(continuous_hour - peak_hour),
                       24 - abs(continuous_hour - peak_hour))

            # Apply Gaussian-like falloff from peak
            sigma = peak_width / 2.355  # Convert width to standard deviation
            peak_contribution = peak_intensity * np.exp(-(dist ** 2) / (2 * sigma ** 2))

            # Add peak contribution to total intensity
            intensity = max(intensity, peak_contribution)

        return intensity

    # Calculate rate multipliers for all hours
    multipliers = [rate_multiplier(h) for h in range(hours)]

    # Normalize to ensure we get the requested average
    total_multiplier = sum(multipliers)
    normalized_multipliers = [m * hours / total_multiplier for m in multipliers]

    # Generate arrivals for each hour
    all_arrivals = []

    for hour in range(hours):
        # Calculate requests for this hour
        hour_requests = int(requests_per_hour * normalized_multipliers[hour])

        # Generate arrivals for this hour
        for i in range(hour_requests):
            # Allow some overlap with adjacent hours for smoother transitions
            hour_offset = random.gauss(0, 0.25)  # Small random offset, mostly within ±0.5 hour
            actual_hour = max(0, min(hours - 0.001, hour + hour_offset))

            # Random position within the hour
            position_in_hour = random.random()

            # Calculate total seconds
            seconds = actual_hour * 3600 + position_in_hour * 3600

            # Add to arrivals list
            all_arrivals.append(seconds)

    # Sort arrivals and convert to datetime
    all_arrivals.sort()
    return [base_time + datetime.timedelta(seconds=t) for t in all_arrivals]


def generate_diurnal_pattern_minutes(minutes=1440, requests_per_minute=100 / 60, peaks=None):
    """
    Generate a diurnal pattern with custom peaks and smooth transitions.

    Parameters:
    - minutes: Total duration to generate in minutes
    - requests_per_minute: Average number of requests per minute
    - peaks: List of tuples (minute, intensity, width) where:
             - minute: Minute of the day for the peak (0-1439)
             - intensity: Relative intensity of the peak (1.0 = baseline)
             - width: Width of the peak in minutes (controls smoothness)

    Returns:
    - List of datetime objects representing arrival times
    """
    import numpy as np
    import random
    import datetime

    # Default peaks if none provided (business hours peak)
    if peaks is None:
        peaks = [(780, 3.0, 480)]  # Peak at 1 PM (780 minutes), 3x intensity, 480 minutes (8 hours) wide

    base_time = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Define a smooth rate multiplier function based on custom peaks
    def rate_multiplier(minute):
        # Convert minute to continuous time (0-1439)
        continuous_minute = minute % minutes

        # Start with baseline intensity
        intensity = 1.0

        # Calculate peak contributions
        for peak_minute, peak_intensity, peak_width in peaks:
            dist = min(abs(continuous_minute - peak_minute),
                       minutes - abs(continuous_minute - peak_minute))

            # Apply Gaussian-like falloff from peak
            sigma = peak_width / 2.355  # Convert width to standard deviation
            peak_contribution = peak_intensity * np.exp(-(dist ** 2) / (2 * sigma ** 2))

            # Add peak contribution to total intensity
            intensity = max(intensity, peak_contribution)

        return intensity

    # Calculate rate multipliers for all minutes
    multipliers = [rate_multiplier(m) for m in range(minutes)]

    # Normalize to ensure we get the requested average
    total_multiplier = sum(multipliers)
    normalized_multipliers = [m * minutes / total_multiplier for m in multipliers]

    # Generate arrivals for each minute
    all_arrivals = []

    for minute in range(minutes):
        # Calculate requests for this minute
        minute_requests = int(requests_per_minute * normalized_multipliers[minute])

        # Generate arrivals for this minute
        for i in range(minute_requests):
            # Allow some overlap with adjacent minutes for smoother transitions
            minute_offset = random.gauss(0, 0.25)  # Small random offset, mostly within ±0.5 minute
            actual_minute = max(0, min(minutes - 0.001, minute + minute_offset))

            # Random position within the minute
            position_in_minute = random.random()

            # Calculate total seconds
            seconds = actual_minute * 60 + position_in_minute * 60

            # Add to arrivals list
            all_arrivals.append(seconds)

    # Sort arrivals and convert to datetime
    all_arrivals.sort()
    return [base_time + datetime.timedelta(seconds=t) for t in all_arrivals]


def generate_diurnal_pattern_delayed(hours=24, requests_per_hour=100):
    all_arrivals = []
    base_time = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    for hour in range(hours):
        # Model time-of-day variation (busier during work hours)
        if 13 <= hour <= 21:  # Business hours
            alpha = 1.0
            beta = 3.0  # Higher rate during business hours
        else:
            alpha = 1.0
            beta = 1.0  # Lower rate outside business hours

        # Generate inter-arrival times for this hour
        print(f'{requests_per_hour}, {hour}')
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
    sampled_timestamps = sorted(random.sample(list(timestamps), num_samples))

    # Scale each timestamp
    compressed_timestamps = []
    base_time = datetime.datetime.now()

    for ts in sampled_timestamps:
        # Scale the timestamp relative to start_time
        relative_time = ts - start_time
        compressed_time = relative_time * time_compression_ratio
        compressed_timestamps.append(base_time + datetime.timedelta(seconds=compressed_time))

    return compressed_timestamps


def convert_datetime_timestamps(arrivals):
    # Assuming compressed_timestamps is your list of datetime objects
    arrival_times_seconds = []

    for dt in arrivals:
        # Convert datetime to seconds since epoch
        seconds_since_epoch = dt.timestamp()
        arrival_times_seconds.append(seconds_since_epoch)

    # Get seconds relative to the first event
    first_event_time = arrival_times_seconds[0]
    arrival_times_seconds = [t - first_event_time for t in arrival_times_seconds]

    return arrival_times_seconds


def main():
    average_rolling_window_duration = 60
    duration = 1200

    # arrival_times = uniform_arrivals(10, 60, 1200)

    # arrival_times = poisson_process(30, duration)

    arrival_times = gamma_distribution(1, 50, 50000)

    # Assuming you have your arrival_times as timestamps in seconds
    # Convert timestamps to datetime objects
    arrival_times_datetime = [datetime.datetime.fromtimestamp(ts) for ts in arrival_times]

    # Morning and evening rush hours
    rush_hour_peaks = [
        (8, 2.5, 7),  # Morning rush: 8 AM, 2.5x intensity, 3 hours wide
        (17, 4.0, 3),  # Evening rush: 5 PM, 3x intensity, 4 hours wide
    ]

    arrival_times_datetime = generate_diurnal_pattern(requests_per_hour=40 * 60 * 60, peaks=rush_hour_peaks)
    plot_pattern(arrival_times_datetime, average_rolling_window_duration, None)
    plt.close()
    events = compress_request_pattern_with_sampling(arrival_times_datetime, original_duration=60 * 24,
                                                    target_duration=20)
    plot_pattern(events, average_rolling_window_duration, None)


def plot_pattern(arrival_times_datetime, average_rolling_window_duration, pdf_path):
    # Convert to pandas Series
    arrival_times_series = pd.Series(arrival_times_datetime)

    # Count events per second by grouping
    events_per_second = pd.Series(arrival_times_series)
    events_per_second = events_per_second.groupby(events_per_second.dt.floor('1s')).size()
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
    if pdf_path is not None:
        plt.savefig(f'{pdf_path}.pdf')
    else:
        plt.show()


if __name__ == '__main__':
    main()
