import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.executeinitial import setup_logging
from src.preprocessing import evaluate_xgboost_per_task, train_xgboost_per_task, \
    create_inputs_outputs_seperated_per_app_windowed, create_train_test_split_per_windowed, \
    create_inputs_outputs_seperated_per_app_windowed_system_events
from src.train import save_models


def read_all_stats(dir, infra, rpss, logger, include_queue_length: bool):
    all_train_data = defaultdict(list)
    all_test_data = defaultdict(list)

    base_path = os.path.join(dir, f"infra-{infra}", "results-reactive")
    base_directory = Path(base_path)
    logger.info(f'Start preparing training and test data, including_queue_length: {include_queue_length}')
    for rps in rpss:
        for exp_dir in base_directory.iterdir():
                for rep_exp_dir in exp_dir.iterdir():
                    file = rep_exp_dir / f"{rps}.json"
                    if not file.exists():
                        continue
                    print(file)

                    with open(file, "r") as infile:
                        stats = json.load(infile)
                        app_definitions = {}
                        for task in stats['taskResults']:
                            app_definitions[task['applicationType']['name']] = list(task['applicationType']['dag'].keys())
                        if not include_queue_length:
                            train_data_sample, test_data_sample = create_train_test_split_per_windowed(
                                create_inputs_outputs_seperated_per_app_windowed(stats, 5, app_definitions))
                        else:
                            train_data_sample, test_data_sample = create_train_test_split_per_windowed(
                                create_inputs_outputs_seperated_per_app_windowed_system_events(stats, 5, app_definitions))
                        if len(train_data_sample) == 0 or len(test_data_sample) == 0:
                            logger.warning(f'Data for infra {infra} and rps {rps} is empty')
                        for fn, data in train_data_sample.items():
                            all_train_data[fn].extend(data)
                        for fn, data in test_data_sample.items():
                            all_test_data[fn].extend(data)
    #
    # for fn, data in all_train_data.items():
    #     all_train_data[fn] = np.array(data).reshape(-1, 2)
    # for fn, data in all_test_data.items():
    #     all_test_data[fn] = np.array(data).reshape(-1, 2)

    logger.info('Finished preparing training and test data')
    return all_train_data, all_test_data



def main():

    dir = sys.argv[1]
    logger = setup_logging(Path(dir))
    infra = sys.argv[2]
    rpss = sys.argv[3].split('-')

    # train xgboost using those results


    all_train_data, all_test_data = read_all_stats(dir, infra, rpss, logger,include_queue_length=False)

    logger.info('Start training')
    models = train_xgboost_per_task(all_train_data)
    logger.info('Finished training')

    print(evaluate_xgboost_per_task(models, all_test_data))

    model_output_dir = os.path.join(dir, f"infra-{infra}", "models", sys.argv[3])
    os.makedirs(model_output_dir, exist_ok=True)
    print(save_models(models, Path(model_output_dir)))


if __name__ == '__main__':
    main()
