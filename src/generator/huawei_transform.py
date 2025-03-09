import json
import random
from typing import List

from src.generator.traces import plot_pattern
from src.parser.parser import parse_simulation_data
from src.placement.model import TimeSeries, WorkloadEvent, ApplicationType, SimulationData, QoSType, \
    DataclassJSONEncoder


def main():
    fn = '49'
    region = 'R1'
    # first week
    for i in range(4):
        print(f"Prepare {i}")
        time_window = (10080 * i, 10080 * (i + 1))
        simulation_duration = 20
        new_average_rps = 7500
        new_std_rps = "5456"
        app = 'nofs-dnn1'
        data_directory = 'data/nofs-ids'
        data: SimulationData = parse_simulation_data(data_directory)
        arrival_file = f'{region}-{fn}-{time_window[0]}-{time_window[1]}-{new_average_rps}-{new_std_rps}-{simulation_duration}'
        with open(
                f'data/nofs-ids/arrivals/{arrival_file}.json',
                'r') as fd:
            arrivals = json.load(fd)

            map_to_events(app, arrival_file, arrivals, data, data_directory, new_average_rps, simulation_duration)


def map_to_events(app, arrival_file, arrivals, data, data_directory, new_average_rps, simulation_duration):
    events: List[WorkloadEvent] = []
    qos_levels_per_app = {}
    if app is not None:
        app_types = [app]
    else:
        app_types = data.application_types.keys()
    for application_type in app_types:
        qos_type_count: int = len(data.qos_types)
        qos_type_index: int = random.randint(0, qos_type_count - 1)
        qos_type_name: str = list(data.qos_types)[qos_type_index]
        qos_type: QoSType = data.qos_types[qos_type_name]
        qos_levels_per_app[str(application_type)] = qos_type
    for timestamp in arrivals:
        application_type_count: int = len(data.application_types)
        if app is not None:

            application_type_name: str = app
            application_type: ApplicationType = data.application_types[
                application_type_name
            ]
        else:
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
    save_as = f'{arrival_file}-{app}'
    time_series = TimeSeries(rps=new_average_rps, duration=simulation_duration * 60, events=events)
    out_file = f'{data_directory}/traces/{save_as}.json'
    with open(out_file, 'w') as fd:
        json.dump(time_series, fd, indent=2, cls=DataclassJSONEncoder)
    return out_file

if __name__ == '__main__':
    main()
