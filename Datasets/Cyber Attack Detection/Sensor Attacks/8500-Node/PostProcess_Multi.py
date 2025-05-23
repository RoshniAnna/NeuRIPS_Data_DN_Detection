import os
import gzip
import pickle
import random
from glob import glob

# Define the results directory
results_dir = "results"
job_id = 0 # Similarly can do for 1, 2, and 3

# Find all part files for job0
pattern = f"*SensorAttacks_8500_job{job_id}_part*.pkl.gz"
job_files = sorted(glob(os.path.join(results_dir, pattern)))

# Merge scenarios
merged_scenarios = []
print(f"Merging files for job {job_id}:", flush =True)
for file in job_files:
    print(f"  - {file}", flush=True)
    with gzip.open(file, 'rb') as f:
        data = pickle.load(f)
        merged_scenarios.extend(data)

# Shuffle
random.shuffle(merged_scenarios)

# Save to final compressed file
output_file = os.path.join(results_dir, f"SensorAttacks_8500_job{job_id}_merged.pkl.gz")
with gzip.open(output_file, 'wb') as f:
    pickle.dump(merged_scenarios, f)

print(f"\n Merged and shuffled {len(merged_scenarios)} scenarios into: {output_file}", flush = True)
