
"""
Post processing for merging
"""

import os
import pickle
import random

# Directory where your SLURM jobs saved the .pkl files
input_dir = 'results'
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

job_ids = [0, 1, 2, 3]

for job_id in job_ids:
    merged_scenarios = []
    job_files = [f for f in os.listdir(input_dir) if f"job{job_id}_" in f and f.endswith('.pkl')]
    print(f"\nProcessing Job {job_id} files:")
    for f in job_files:
        print(f"  - {f}")
    
    for pkl_file in job_files:
        pkl_path = os.path.join(input_dir, pkl_file)
        with open(pkl_path, 'rb') as file:
            scenarios = pickle.load(file)
            merged_scenarios.extend(scenarios)
    
    random.shuffle(merged_scenarios)
    
    output_file = os.path.join(output_dir, f'PVAttacks_8500_job{job_id}_merged.pkl')
    with open(output_file, 'wb') as outfile:
        pickle.dump(merged_scenarios, outfile)
    
    print(f"Saved merged job {job_id} to {output_file}")