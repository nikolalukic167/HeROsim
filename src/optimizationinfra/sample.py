import sys

from src.sample import load_data, latin_hypercube_sampling, save_samples


def main():
    input_dir = sys.argv[1]
    input_prefix = f'{input_dir}/combinations'
    output_prefix = f'{input_dir}/lhs_samples'

    n_samples = int(sys.argv[2])
    seed = 42

    combinations, mapping = load_data(input_prefix)

    selected_samples = latin_hypercube_sampling(
        combinations,
        n_samples=n_samples,
        seed=seed
    )

    save_samples(selected_samples, mapping, output_prefix)


if __name__ == '__main__':
    main()
