import json
from collections import defaultdict

def analyze_applications(json_file_path):
    # Read JSON file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Create dictionaries to store counts and timestamps per application
    app_time_counts = defaultdict(lambda: defaultdict(int))
    app_timestamps = defaultdict(list)

    # Collect timestamps per application
    for entry in data['events']:
        timestamp = int(entry['timestamp'])
        app_name = entry['application']['name']
        app_timestamps[app_name].append(timestamp)
        app_time_counts[app_name][timestamp] += 1

    # Calculate averages per application using their specific time ranges
    averages = {}
    for app_name in app_time_counts:
        min_time = min(app_timestamps[app_name])
        max_time = max(app_timestamps[app_name])
        total_count = sum(app_time_counts[app_name].values())
        # Add 1 to time range to include both start and end seconds
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

# Example usage
if __name__ == "__main__":
    file_path = "data/ids/traces/workload-83-10.json"  # Replace with your JSON file path
    try:
        results = analyze_applications(file_path)
        print("\nStatistics for each application:")
        for app, stats in results.items():
            print(f"\nApplication: {app}")
            print(f"Average occurrences per second: {stats['average']:.2f}")
            print(f"Time range: {stats['time_range']} seconds")
            print(f"First occurrence: {stats['min_timestamp']}")
            print(f"Last occurrence: {stats['max_timestamp']}")
            print(f"Total count: {stats['total_count']}")
    except FileNotFoundError:
        print("Error: JSON file not found")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format")
