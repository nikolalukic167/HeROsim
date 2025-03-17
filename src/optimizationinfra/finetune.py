import os
import sys
from pathlib import Path

from src.optimizer import finetune_initial_models
from src.train import save_models


def main():
    base_dir = Path(sys.argv[1])
    opt_path = base_dir / "optimization_results" / sys.argv[2]
    path_fine_tuned_models = opt_path / "fine_tuned_models"

    models_path = base_dir / "initial_results"

    fine_tuned_models = finetune_initial_models(models_path=models_path, opt_path=opt_path)
    os.makedirs(path_fine_tuned_models, exist_ok=True)
    save_models(fine_tuned_models, path_fine_tuned_models)


if __name__ == '__main__':
    main()