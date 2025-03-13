Motivational Experiments
========================

1. execute reactive with 30, 60, 90 and infra 0 (`python -m src.motivational.reactive <output-dir> 0 30-60-90`)
   * Results are saved `<output-dir>/infra-0/results/`
2. train xgboost using those results (`python -m src.motivational.train <output-dir> 0 30-60-90`)
   * Models are saved under `<output-dir>/models/30-60-90/`
4. then use proactive to execute 30, 45, 60, 80, 90 and show penalties (`python -m src.motivational.proactive <output-dir> 0 0 30-60-90 45`)
5. repeat 1,2,3,4 with infra 1 (`python -m src.motivational.reactive 1`)
6. use model trained on infra 0 and use for infra 1 (`python -m src.motivational.proactive 0 1`)
7. use model trained on infra 1 and use for infra 0 (`python -m src.motivational.proactive 1 0`)

Functions:

medium similarity: 817, 2119, 1412
high similarity: 1233, 1437, 351
low similarity: 49, 1465, 1358
