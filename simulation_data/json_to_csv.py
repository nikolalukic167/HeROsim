#!/usr/bin/env python3
import json
import csv
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

def convert_json_to_csv(json_path, csv_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        placement_plan = data.get('placement_plan', {})
        rtt = data.get('rtt', '')
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['task', 'node', 'platform', 'rtt'])
            
            for task, values in sorted(placement_plan.items(), key=lambda x: int(x[0])):
                if len(values) >= 2:
                    writer.writerow([task, values[0], values[1], rtt])
        
        return True
    except Exception as e:
        print(f"Error converting {json_path}: {e}")
        return False

def _convert_task(args):
    json_path, csv_path = args
    return convert_json_to_csv(Path(json_path), Path(csv_path))

def process_ds_dir(ds_dir):
    placements_dir = ds_dir / 'placements'
    if not placements_dir.exists():
        return 0
    
    csv_dir = ds_dir / 'placements_csv'
    csv_dir.mkdir(exist_ok=True)
    
    json_files = list(placements_dir.glob('placement_summary_*.json'))
    tasks = [(str(json_file), str(csv_dir / (json_file.stem + '.csv'))) for json_file in json_files]
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(_convert_task, tasks))
    
    return sum(results)

def main():
    import sys
    run_name = sys.argv[1] if len(sys.argv) > 1 else 'run10_all'
    base_dir = Path(f'/root/projects/my-herosim/simulation_data/artifacts/{run_name}/gnn_datasets')
    ds_dirs = sorted(base_dir.glob('ds_*'))
    
    print(f"Found {len(ds_dirs)} ds directories")
    
    total_files = 0
    for ds_dir in ds_dirs:
        count = process_ds_dir(ds_dir)
        total_files += count
        print(f"Processed {ds_dir.name}: {count} files")
    
    print(f"Total: {total_files} JSON files converted to CSV")

if __name__ == '__main__':
    main()

