import json
import pathlib

from src.executeinitial import flatten_workloads


def main():
    to_flatten = 'data/nofs-ids/peak_configs_2.json'
    to_flatten_path = pathlib.Path(to_flatten)
    base_dir = pathlib.Path('./data/nofs-ids/traces')
    with open(to_flatten) as f:
        peak_config = json.load(f)

        for idx, config in enumerate(peak_config):
            workloads = {}
            for app, workload_config in config.items():
                with open(base_dir / f'workload-{workload_config["rps"]}-750-{app}-peak_pattern.json', 'r') as fd:
                    workload = json.load(fd)
                    workloads[app] = workload['events']

            workload = flatten_workloads(workloads)
            workload['peak_config'] = peak_config
            with open(base_dir / f'workload-flatten-{to_flatten_path.stem}-{idx}.json', 'w') as fd:
                json.dump(workload, fd)


if __name__ == '__main__':
    main()
