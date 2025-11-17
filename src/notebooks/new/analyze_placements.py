#!/usr/bin/env python3
"""
Analyze placement summaries to find duplicates and visualize data.
"""

import os
import random
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

# Root directory
ROOT_DIR = Path("/root/projects/my-herosim/simulation_data/artifacts/run10_all/gnn_datasets")

def load_placement_summaries(dataset_dir: Path) -> List[Tuple[str, Tuple[Tuple[int, int], ...], float]]:
    """Load all placement summaries from a dataset directory."""
    placements_csv_dir = dataset_dir / "placements_csv"
    if not placements_csv_dir.exists():
        return []
    
    summaries = []
    for csv_file in placements_csv_dir.glob("placement_summary_*.csv"):
        try:
            rows = []
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        task = int(row.get('task', -1)) if row.get('task') else -1
                        node = int(row.get('node', -1)) if row.get('node') else -1
                        platform = int(row.get('platform', -1)) if row.get('platform') else -1
                        rtt_val = float(row.get('rtt', 0)) if row.get('rtt') else 0.0
                        
                        if node >= 0 and platform >= 0 and rtt_val > 0:
                            rows.append((task, node, platform, rtt_val))
                    except (ValueError, KeyError):
                        continue
            
            if not rows:
                continue
            
            # Sort by task to match hash table format
            rows.sort(key=lambda x: (x[0] if x[0] >= 0 else 999, x[1], x[2]))
            
            # Build combo tuple: ((node1, platform1), (node2, platform2), ...)
            combo = tuple((node, platform) for _, node, platform, _ in rows)
            rtt = rows[0][3]  # Use first RTT value
            
            summaries.append((csv_file.name, combo, rtt))
        except Exception as e:
            print(f"  Error loading {csv_file.name}: {e}")
    
    return summaries

def find_duplicates(summaries: List[Tuple[str, Tuple[Tuple[int, int], ...], float]]) -> Dict[Tuple[Tuple[int, int], ...], List[Tuple[str, float]]]:
    """Find duplicate placement summaries (same combo tuple)."""
    combo_to_files = defaultdict(list)
    
    for filename, combo, rtt in summaries:
        combo_to_files[combo].append((filename, rtt))
    
    # Return only duplicates (more than one file with same combo)
    return {combo: files for combo, files in combo_to_files.items() if len(files) > 1}

def main():
    print("="*80)
    print("PLACEMENT SUMMARY ANALYSIS")
    print("="*80)
    print()
    
    # Get all dataset directories
    dataset_dirs = sorted([d for d in ROOT_DIR.iterdir() if d.is_dir() and d.name.startswith("ds_")])
    
    if len(dataset_dirs) == 0:
        print(f"ERROR: No dataset directories found in {ROOT_DIR}")
        return
    
    print(f"Found {len(dataset_dirs)} dataset directories")
    print()
    
    # Pick 10 random dataset directories
    selected_dirs = random.sample(dataset_dirs, min(10, len(dataset_dirs)))
    print(f"Analyzing {len(selected_dirs)} random dataset directories:")
    for d in selected_dirs:
        print(f"  - {d.name}")
    print()
    
    # Analyze each dataset
    all_summaries = []
    total_duplicates = 0
    
    for dataset_dir in selected_dirs:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_dir.name}")
        print(f"{'='*80}")
        
        summaries = load_placement_summaries(dataset_dir)
        all_summaries.extend([(dataset_dir.name, *s) for s in summaries])
        
        print(f"  Found {len(summaries)} placement summaries")
        
        if summaries:
            # Show first few summaries
            print(f"\n  First 3 placement summaries:")
            for i, (filename, combo, rtt) in enumerate(summaries[:3]):
                print(f"    [{i+1}] {filename}")
                print(f"        Combo: {combo}")
                print(f"        RTT: {rtt:.4f}")
            
            # Find duplicates within this dataset
            duplicates = find_duplicates(summaries)
            if duplicates:
                print(f"\n  ⚠️  Found {len(duplicates)} duplicate placements in this dataset:")
                for combo, files in list(duplicates.items())[:3]:
                    print(f"    Combo: {combo}")
                    print(f"      Found in {len(files)} files:")
                    for filename, rtt in files:
                        print(f"        - {filename} (RTT: {rtt:.4f})")
                total_duplicates += len(duplicates)
            else:
                print(f"  ✓ No duplicates found in this dataset")
    
    # Find duplicates across all selected datasets
    print(f"\n{'='*80}")
    print("CROSS-DATASET DUPLICATE ANALYSIS")
    print(f"{'='*80}")
    
    # Group by combo tuple (ignoring dataset_id)
    combo_to_datasets = defaultdict(list)
    for dataset_id, filename, combo, rtt in all_summaries:
        combo_to_datasets[combo].append((dataset_id, filename, rtt))
    
    cross_duplicates = {combo: files for combo, files in combo_to_datasets.items() if len(files) > 1}
    
    if cross_duplicates:
        print(f"\n⚠️  Found {len(cross_duplicates)} placement combos that appear in multiple datasets:")
        for i, (combo, files) in enumerate(list(cross_duplicates.items())[:5]):
            print(f"\n  [{i+1}] Combo: {combo}")
            print(f"      Appears in {len(files)} files across {len(set(d for d, _, _ in files))} datasets:")
            for dataset_id, filename, rtt in files:
                print(f"        - {dataset_id}/{filename} (RTT: {rtt:.4f})")
    else:
        print(f"\n✓ No cross-dataset duplicates found")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total placement summaries analyzed: {len(all_summaries)}")
    print(f"Unique placement combos: {len(combo_to_datasets)}")
    print(f"Duplicates within datasets: {total_duplicates}")
    print(f"Cross-dataset duplicates: {len(cross_duplicates)}")
    
    # Show why we might have multiple successful regret calculations
    print(f"\n{'='*80}")
    print("EXPLANATION: Why Multiple Successful Regret Calculations?")
    print(f"{'='*80}")
    print("""
The hash table is built from placement_summary CSV files. Each CSV file contains:
- A placement combo: ((node1, platform1), (node2, platform2), ...)
- An RTT value

When building the hash table:
- If multiple CSV files have the SAME placement combo, only ONE entry is kept (the one with lowest RTT)
- So each (dataset_id, combo) key appears only once in the hash table

Why you see multiple successful regret calculations:
- You're processing MULTIPLE graphs in the validation set (e.g., 355 graphs)
- Each graph that has a predicted placement matching the hash table gets counted ONCE
- So if 9 graphs have successful matches, you count 9 successful calculations

This is CORRECT behavior! Each graph is evaluated independently, and if its predicted
placement exists in the hash table, it contributes one successful regret calculation.

The number of successful calculations = number of graphs with predicted placements that
match entries in the hash table.
    """)

if __name__ == "__main__":
    main()

