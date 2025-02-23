source .venv/bin/activate

# infra 0
python -m src.motivational.reactive motivational_results 0 30-60-90 5
python -m src.motivational.train motivational_results 0 30-60-90
python -m src.motivational.proactive motivational_results 0 0 30-60-90 30-45-60-80-90
python -m src.motivational.proactiveparallel motivational_results 0 0 60-90 30-45-60-80-90 3 4

# infra 1
python -m src.motivational.reactive motivational_results 1 30-60-90
python -m src.motivational.train motivational_results 1 30-60-90
python -m src.motivational.proactive motivational_results 1 1 30-60-90 30-45-60-80-90

# infra 0 model on infra 1
python -m src.motivational.proactive motivational_results 0 1 30-60-90 30-45-60-80-90

# infra 1 model on infra 0
python -m src.motivational.proactive motivational_results 1 0 30-60-90 30-45-60-80-90

