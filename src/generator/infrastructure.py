import argparse
import json
from pathlib import Path

from src.executeinitial import prepare_simulation_config
from src.placement.model import dir_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "☁️ Tasks Scheduling on Heterogeneous Resources for Serverless Cloud"
            " Computing"
        )
    )
    parser.add_argument(
        "-d",
        "--data-directory",
        help="Simulation data JSON files directory",
        type=dir_path,
        required=True,
    )

    args = parser.parse_args()

    data_dir = Path(args.data_directory)
    with open(data_dir / 'motivational-samples.json', 'r') as fd:
        obj = json.load(fd)
        for idx, sample in enumerate(obj['samples']):
            infra_config = prepare_simulation_config(sample, obj['mapping'], obj)
            infra_config['device_props'] = {
                "device_prop_rpi": sample[2],
                "device_prop_xavier": sample[3],
                "device_prop_pyngFpga": sample[4]
            }
            with open(data_dir / f'motivational-infrastructures/{idx}.json', 'w') as fd:
                json.dump(infra_config, fd, indent=4)


if __name__ == '__main__':
    main()
