import json
from collections import defaultdict
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def increase_events(data, factor):
    # Group events by application and timestamp
    app_time_events = defaultdict(lambda: defaultdict(list))
    new_data = []

    # First, organize existing events
    for event in data:
        timestamp = event['timestamp']
        app_name = event['application']['name']
        app_time_events[app_name][int(timestamp)].append(event)

    # Process each application separately
    for app_name, time_events in app_time_events.items():
        timestamps = sorted(time_events.keys())
        min_time = int(min(timestamps))
        max_time = int(max(timestamps))

        # For each second in the range
        for timestamp in range(min_time, max_time + 1):
            existing_count = len(time_events[timestamp])

            # Calculate target count for this second
            target_mean = existing_count * factor if existing_count > 0 else 0

            if target_mean > existing_count:
                # Generate additional events using Poisson distribution
                additional_count = np.random.poisson(target_mean - existing_count)

                # Add new events
                for _ in range(additional_count):
                    # Clone a template event if one exists for this timestamp
                    if existing_count > 0:
                        template_event = deepcopy(time_events[timestamp][0])
                    else:
                        # Create new event using template from another timestamp
                        template_event = deepcopy(next(iter(time_events[timestamps[0]]))[0])
                        template_event['timestamp'] = timestamp

                    time_events[timestamp].append(template_event)

    # Flatten the structure back to a list
    for app_events in app_time_events.values():
        for time_events in app_events.values():
            new_data.extend(time_events)

    return new_data

def increase_events_of_app(data, factor, app_name_increase: str):
    # Group events by application and timestamp
    app_time_events = defaultdict(lambda: defaultdict(list))
    new_data = []

    # First, organize existing events
    for event in data:
        timestamp = event['timestamp']
        app_name = event['application']['name']
        if app_name != app_name_increase:
            continue
        app_time_events[app_name][int(timestamp)].append(event)

    # Process each application separately
    for app_name, time_events in app_time_events.items():

        timestamps = sorted(time_events.keys())
        min_time = int(min(timestamps))
        max_time = int(max(timestamps))

        # For each second in the range
        for timestamp in range(min_time, max_time + 1):
            existing_count = len(time_events[timestamp])

            # Calculate target count for this second
            target_mean = existing_count * factor if existing_count > 0 else 0

            if target_mean > existing_count:
                # Generate additional events using Poisson distribution
                additional_count = np.random.poisson(target_mean - existing_count)

                # Add new events
                for _ in range(additional_count):
                    # Clone a template event if one exists for this timestamp
                    if existing_count > 0:
                        template_event = deepcopy(time_events[timestamp][0])
                    else:
                        # Create new event using template from another timestamp
                        template_event = deepcopy(next(iter(time_events[timestamps[0]]))[0])
                        template_event['timestamp'] = timestamp

                    time_events[timestamp].append(template_event)

    # Flatten the structure back to a list
    for app_events in app_time_events.values():
        for time_events in app_events.values():
            new_data.extend(time_events)

    return new_data



def analyze_applications(data):
    # Create dictionaries to store counts and timestamps per application
    app_time_counts = defaultdict(lambda: defaultdict(int))
    app_timestamps = defaultdict(list)

    # Collect timestamps per application
    for entry in data:
        timestamp = int(entry['timestamp'])
        app_name = entry['application']['name']
        app_timestamps[app_name].append(timestamp)
        app_time_counts[app_name][timestamp] += 1

    # Calculate averages per application
    averages = {}
    for app_name in app_time_counts:
        min_time = min(app_timestamps[app_name])
        max_time = max(app_timestamps[app_name])
        total_count = sum(app_time_counts[app_name].values())
        time_range = max_time - min_time + 1
        average = total_count / time_range
        averages[app_name] = {
            'average': average,
            'min_timestamp': min_time,
            'max_timestamp': max_time,
            'time_range': time_range,
            'total_count': total_count
        }

    return averages


def plot_comparison(original_data, increased_data):
    # Create dictionaries to store counts per timestamp
    original_counts = defaultdict(lambda: defaultdict(int))
    increased_counts = defaultdict(lambda: defaultdict(int))

    # Count events per timestamp for both datasets
    for event in original_data:
        timestamp = int(event['timestamp'])
        app_name = event['application']['name']
        original_counts[app_name][timestamp] += 1

    for event in increased_data:
        timestamp = int(event['timestamp'])
        app_name = event['application']['name']
        increased_counts[app_name][timestamp] += 1

    # Get unique applications
    apps = set(original_counts.keys())

    # Calculate number of subplots needed
    n_apps = len(apps)
    n_cols = min(2, n_apps)
    n_rows = (n_apps + n_cols - 1) // n_cols

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_apps == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot each application
    for idx, app_name in enumerate(sorted(apps)):
        ax = axes[idx]

        # Get time range for this application
        timestamps_orig = sorted(original_counts[app_name].keys())
        timestamps_incr = sorted(increased_counts[app_name].keys())
        min_time = int(min(min(timestamps_orig), min(timestamps_incr)))
        max_time = int(max(max(timestamps_orig), max(timestamps_incr)))

        # Create continuous time series
        time_range = range(min_time, max_time + 1)
        orig_values = [original_counts[app_name][t] for t in time_range]
        incr_values = [increased_counts[app_name][t] for t in time_range]

        # Normalize timestamps for better readability
        time_range_norm = [t - min_time for t in time_range]

        # Plot
        ax.plot(time_range_norm, orig_values, label='Original', color='blue', alpha=0.6)
        ax.plot(time_range_norm, incr_values, label='Increased', color='red', alpha=0.6)

        ax.set_title(f'Application: {app_name}')
        ax.set_xlabel('Seconds from start')
        ax.set_ylabel('Events per second')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Force y-axis to use integers
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Remove any unused subplots
    for idx in range(n_apps, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    return fig


# Example usage
if __name__ == "__main__":
    file_path = "data/nofs-ids/traces/workload-60-600.json"  # Replace with your JSON file path
    increase_factor = 1  # Desired increase factor

    try:
        # Read original data
        with open(file_path, 'r') as file:
            original_data = json.load(file)

        # Analyze original data
        print("\nOriginal statistics:")
        original_stats = analyze_applications(original_data['events'])
        for app, stats in original_stats.items():
            print(f"\nApplication: {app}")
            print(f"Average RPS: {stats['average']:.2f}")
            print(f"Total count: {stats['total_count']}")

        # Increase events
        increased_data = increase_events(original_data['events'], increase_factor)

        # Analyze increased data
        print("\nIncreased statistics:")
        increased_stats = analyze_applications(increased_data)
        for app, stats in increased_stats.items():
            print(f"\nApplication: {app}")
            print(f"Average RPS: {stats['average']:.2f}")
            print(f"Total count: {stats['total_count']}")

        # Create and save the plot
        fig = plot_comparison(original_data['events'], increased_data)
        plt.show()
        # plt.savefig('events_comparison.png', dpi=300, bbox_inches='tight')
        # plt.close()
        print('here')

        # Optionally save the new data
        with open('increased_events.json', 'w') as f:
            json.dump(increased_data, f, indent=2)

        print("\nPlot saved as 'events_comparison.png'")

    except FileNotFoundError:
        print("Error: JSON file not found")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
