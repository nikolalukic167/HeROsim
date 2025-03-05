from typing import List

from src.placement.infrastructure import Task
from src.placement.model import TimeSeries


def count_tasks_in_windows(self, tasks: List[Task], task_type: str, lookback: int, window_size: int) -> List[int]:
    """
    Count tasks of given type in time windows up to lookback seconds.

    Args:
        tasks: List of Task objects
        task_type: Type of task to count
        lookback: How many seconds to look back from now
        window_size: Size of each window in seconds

    Returns:
        List of task counts per window, from oldest to newest
    """
    current_time = self.env.now
    start_time = current_time - lookback

    # Calculate number of windows
    n_windows = lookback // window_size
    if lookback % window_size != 0:
        n_windows += 1

    # Initialize counts for each window
    window_counts = [0] * n_windows

    # Filter tasks by type, lookback period, and dispatched status
    relevant_tasks = [
        task for task in tasks
        if hasattr(task, 'dispatched_time')  # Check if task has been dispatched
           and task.dispatched_time >= start_time
           and task.type['name'] == task_type
    ]

    # Count tasks per window
    for task in relevant_tasks:
        # Calculate which window this task belongs to
        window_index = (task.dispatched_time - start_time) // window_size
        if 0 <= window_index < n_windows:
            window_counts[int(window_index)] += 1

    return window_counts

def count_events_in_windows_ts(current_time: int, time_series: TimeSeries, task_type: str, lookforward: int, window_size: int) -> List[int]:
    """
    Count matching events in future time windows.

    Args:
        current_time: Current time
        time_series: TimeSeries object containing future events
        task_type: Type of task to match in event's dag
        lookforward: How many seconds to look forward from now
        window_size: Size of each window in seconds

    Returns:
        List of event counts per window, from now to lookforward
    """
    end_time = current_time + lookforward

    # Calculate number of windows
    n_windows = lookforward // window_size
    if lookforward % window_size != 0:
        n_windows += 1

    # Initialize counts for each window
    window_counts = [0] * n_windows

    # Filter relevant events
    relevant_events = [
        event for event in time_series.events
        if current_time <= event['timestamp'] <= end_time
           and task_type in event['application']['dag']
    ]
    if len(relevant_events) == 0:
        return None

    # Count events per window
    for event in relevant_events:
        window_index = (event['timestamp'] - current_time) // window_size
        if 0 <= window_index < n_windows:
            window_counts[int(window_index)] += 1

    return window_counts
