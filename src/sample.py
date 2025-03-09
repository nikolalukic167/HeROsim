import numpy as np
from scipy.stats import qmc
import json
import pickle
from pathlib import Path

def load_data(prefix="combinations"):
    """Load combinations and mapping."""
    # Load combinations
    combinations = np.load(f"{prefix}.npy")

    # Load mapping
    with open(f"{prefix}_mapping.pkl", 'rb') as f:
        mapping = pickle.load(f)

    return combinations, mapping

def normalize_array(arr):
    """Normalize array to [0,1] range for each dimension."""
    min_vals = np.min(arr, axis=0)
    max_vals = np.max(arr, axis=0)
    # Avoid division by zero
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1
    return (arr - min_vals) / ranges, min_vals, max_vals

def find_nearest_combination(sample_point, combinations, normalized_combinations):
    """Find the existing combination closest to the sampled point."""
    distances = np.linalg.norm(normalized_combinations - sample_point, axis=1)
    nearest_idx = np.argmin(distances)
    return combinations[nearest_idx]

def latin_hypercube_sampling(combinations, n_samples, seed=42):
    """
    Perform Latin Hypercube Sampling on the combinations.
    Returns both the sampled points and the nearest actual combinations.
    """
    n_dimensions = combinations.shape[1]

    # Normalize combinations to [0,1] range
    normalized_combinations, min_vals, max_vals = normalize_array(combinations)

    # Initialize LHS sampler
    sampler = qmc.LatinHypercube(d=n_dimensions, seed=seed)

    # Generate samples
    samples = sampler.random(n=n_samples)

    # Find nearest actual combinations for each sample
    selected_combinations = np.array([
        find_nearest_combination(sample, combinations, normalized_combinations)
        for sample in samples
    ])

    return selected_combinations

def save_samples(samples, mapping, output_prefix="lhs_samples"):
    """Save samples with mapping."""
    # Save samples
    np.save(f"{output_prefix}.npy", samples)

    # Save mapping
    with open(f"{output_prefix}_mapping.pkl", 'wb') as f:
        pickle.dump(mapping, f)

def print_sample_statistics(sample, mapping):
    """Print detailed statistics about a sample."""
    print("\nSample values:")
    for idx, value in enumerate(sample):
        print(f"{mapping[idx]}: {value}")

def main():
    # Configuration
    input_prefix = "simulation_data/combinations_simple"
    output_prefix = "simulation_data/lhs_samples_simple"
    n_samples = 8
    seed = 42

    try:
        # Load data
        print(f"Loading combinations from {input_prefix}.npy")
        combinations, mapping = load_data(input_prefix)
        print(f"Loaded {len(combinations)} combinations")

        # Perform LHS
        print(f"Performing Latin Hypercube Sampling to select {n_samples} samples...")
        selected_samples = latin_hypercube_sampling(
            combinations,
            n_samples=n_samples,
            seed=seed
        )

        # Save results
        save_samples(selected_samples, mapping, output_prefix)
        print(f"Saved {n_samples} samples to {output_prefix}.npy")

        # Print statistics
        print("\nSample statistics:")
        print(f"Original combinations: {len(combinations)}")
        print(f"Selected samples: {len(selected_samples)}")
        print("\nFirst few samples:")
        for i, sample in enumerate(selected_samples[:3]):
            print(f"\nSample {i+1}:")
            print_sample_statistics(sample, mapping)

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
