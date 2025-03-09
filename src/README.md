Simulation-driven synthetic dataset generation to optimize proactive serverless autoscaling
============================================================================================

1. Choose scenario folder from `data` (i.e., `data/ids`)
2. Needs to have: traces (e.g., `traces/workload-83-100.json`), `application-types.json`, `platform-types.json`, `qos-types.json`, `storage-types.json`, `task-types.json`.
3. Create a space JSON file (e.g., `simulation_data/infrastructure_config.json`)
4. Generate all combinations based on space JSON (`src/generateall.py <space-json> <output-folder>`)
   * Saves the combinations as `combinations.npy` and `combinations_mapping.pkl`.
   * The `combinations.npy` is a list of all possible configurations the space spans. The mapping file is needed to map the individual entries of each configuration to the respective value (i.e,. `0 -> network_bandwidth`, etc.)
5. Sample from all combinations using Latin Hypercube Sampling (LHS) (`src/sample.py <n> <seed> <sample.npy> <sample_mapping.pkl> <output-folder>`)
    * Saves the samples as `lhs_samples.npy` and `lhs_samples_mapping.pkl`.
    * `n` is the number of samples.
6. Execute the simulation-driven synthetic dataset generation.



# Space JSON

Space JSON file contains the space of environments the scenarios can be in.
Allows to specify dynamic generation of the infrastructure and workload patterns.

Following keys need to be set:
1. `pci` - describes the possible devices in the cluster. Each entry must have a name (key of object), and `min`, `max`, `step` to define space (i.e., proportion the device occurs) and then `specs` which contains all platforms, memory and storage the device should be generated with.
2. `wsc` - describes the workload space per application. Required keys of each object: `average`, `min`, `max` and `step`.
3. `csc` - describes the possible cluster sizes. Required keys of each object: `average`, `min`, `max`, `step`.
4. `nwc` - describes the possible network speeds. Required keys of each object: `average`, `min`, `max`, `step`.


# Simulation-driven Synthetic Dataset Generation

The dataset generation mainly consists of three steps:

1. Execute the reactive policy on the samples, train models per task and then evaluate the penalties for each sample. 
2. Samples with a penalty higher than `pen-threshold` will be further optimized. Specifically, for each sample we start a Bayesian Optimization that will explore the nearby area of the sample. The nearby area is specified through `area-threshold` in percentage (< 1). The bayesian optimization uses this evaluation function:
   1. Simulate scenario (given based on parameters chosen by bayesian optimization) using reactive policy and generate new dataset.
   2. Fine-tune models using new dataset. 
   3. Simulate scenario using proactive policy using fine-tuned models.
   4. If penalty is below threshold, abort optimization, otherwise continue.
3. The script also keeps track of those intermin synthetic datasets that have improved model performance and saves them after.

More notes on execution:
1. Each simulation is run by starting a dedicated docker container that is removed after the simulation.
2. Passing data between container and the optimization script is done through files.

The execution is split into two scripts.
The first script generates the initial dataset based on the sampled parameters, the second one performs the optimization for each sample where the proactive policy has underperformed.

### Execution 

#### 1. Initial dataset:

* Run: `python -m src.executeinitial <base-dir> <sim-input-dir> <base-workload-file> <n-parallel>`
  * `base-dir`: The folder in which the sample files and space JSON are residing.
    * Required files in `base-dir`: `lhs_samples.npy`, `lhs_samples_mapping.pkl`, `space.json`
    * The `results`folder will contain a JSON file for each run that contains all info about the run (i.e., sample used to run it and the results), also the trained XGBoost models per task.
  * `sim-input-dir`: The folder in which other relevant JSON files reside which the simulation needs.
    * Required files in `sim-input-dir`: `application-types.json`, `platform-types.json`, `qos-types.json`, `storage-types.json`, `task-types.json`
  * `base-workoad-file`: contains the arrival of tasks and is used throughout all scenario runs as base workload.
  * `n-parallel`: specifies the number of containers that can be spawned concurrently.
* After executing this script, the `initial_results` folder should be filled with simulation JSON files and trained models.
* Each simulation will be started in its own container and input and output are transferred via the filesystem.
* The output of each simulation is a single JSON file that includes all results and the input sample (i.e., enough to train model and to start optimization from).

#### 2. Optimize under-performing samples

* Run: `python -m src.executeoptimization <initial-results-folder> <n-parallel> <under-performing-percentile> <target-penalty>`
  * `initial-results-folder`: A folder that contains the samples and the results of initial generation.
    * Required files: `lhs_samples.npy`
    * Required folder: `initial_results` - must contain all simulation results and initial models as JSON files.
  * `n-parallel`: Specifies the number of optimizations that can be run concurrently.
  * `under-performing-percentile`: Specifies which percentile (upwards) should be optimized based on result's penalty
  * `target-penalty`: Which penalty proportion the fine-tuned model should have to stop optimization for a sample.
* After executing this script, the `optimization_results/%Y%m%d_%H%M%S` folder will contain all results, including interim results.

Short Version
=============

1. Create space.json
2. Generate all samples
3. Sample
4. Execute Initial
5. Execute Optimization

# Post Optimization

1. fine tune base models with new data: src/optimizer.py -> saves models in optimized folder
2. run baseline workloads with new models: src/motivational/proactiveparalleldiffworkloadsfinetunedmodels.py (model dir -> fine tuned folder)
