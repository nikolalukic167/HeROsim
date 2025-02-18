import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
from xgboost import Booster

# Assuming increase_events is imported from your previous script
from src.preprocessing import create_train_test_split_per_task, preprocess_workload, create_inputs_outputs, \
    preprocess_pods, train_xgboost_per_task, evaluate_xgboost_per_task


def train_model(output_dir, samples):
    all_train_data = defaultdict(list)
    all_test_data = defaultdict(list)
    for i, sample in enumerate(samples[:1]):
        with open(output_dir / f"simulation_{i + 1}.json", 'r') as fd:
            obj = json.load(fd)['stats']
            train_data_sample, test_data_sample = create_train_test_split_per_task(
                create_inputs_outputs(preprocess_workload(obj['taskResults']), preprocess_pods(obj['scaleEvents'])))
            for fn, data in train_data_sample.items():
                all_train_data[fn].append(data)
            for fn, data in test_data_sample.items():
                all_test_data[fn].append(data)
    for fn, data in all_train_data.items():
        all_train_data[fn] = np.array(data).reshape(-1, 2)
    for fn, data in all_test_data.items():
        all_test_data[fn] = np.array(data).reshape(-1, 2)
    models = train_xgboost_per_task(all_train_data)
    return models, evaluate_xgboost_per_task(models, all_test_data)

def load_models(model_locations: Dict[str, str]):
    models = {}
    for fn, model_location in model_locations.items():
        loaded_model = Booster()
        loaded_model.load_model(model_location)
        models[fn] = loaded_model
    return models

def save_models(models, output_dir):
    model_paths = {}
    for fn, model in models.items():
        model_path = output_dir / f"{fn}.json"
        model.save_model(model_path)
        model_paths[fn] = model_path
    return model_paths

if __name__ == '__main__':
    base_dir = Path("simulation_data")
    sim_input_path = Path("data/ids")  # Base path for simulation input files
    samples_file = base_dir / "lhs_samples.npy"
    mapping_file = base_dir / "lhs_samples_mapping.pkl"
    config_file = base_dir / "infrastructure_config.json"
    workload_base_file = "data/ids/traces/workload-83-100.json"
    output_dir = base_dir / "results"
    samples = np.load(samples_file)

    print(train_model(output_dir, samples)[0])
