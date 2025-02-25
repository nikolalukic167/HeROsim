import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from src.generator.traces import gamma_distribution, compress_request_pattern_with_sampling


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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# Example usage:
# Compress to 10 minutes total duration while maintaining 5s→1s window compression
# compressed_arrivals = compress_request_pattern(
#     arrival_times=original_arrivals,
#     target_duration_minutes=10,
#     original_window=5,
#     target_window=1
# )


# Generate an hour of data
hour_arrivals = generate_diurnal_pattern(hours=24, requests_per_hour=30 * 60 * 60)

# Compress to 1 minute
# minute_arrivals = compress_request_pattern(hour_arrivals)
minute_arrivals = compress_request_pattern_with_sampling(
    arrival_times=hour_arrivals, original_duration=24 * 60, target_duration=10
)


# Visualize both patterns
def plot_comparison(hour_data, minute_data):
    # Process hour data
    hour_series = pd.Series(hour_data)
    hour_counts = hour_series.groupby(hour_series.dt.floor('1S')).size()

    # Process minute data
    minute_series = pd.Series(minute_data)
    minute_counts = minute_series.groupby(minute_series.dt.floor('1S')).size()

    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot hour data
    sns.lineplot(x=hour_counts.index, y=hour_counts.values, ax=ax1)
    ax1.set_title('Original Pattern (1 hour)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Requests')

    # Plot minute data
    sns.lineplot(x=minute_counts.index, y=minute_counts.values, ax=ax2)
    ax2.set_title('Compressed Pattern (1 minute)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Requests')

    plt.tight_layout()
    plt.show()


plot_comparison(hour_arrivals, minute_arrivals)
