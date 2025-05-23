import os
import gzip
import pickle
import random
import gc
from glob import glob

# Define the results directory
results_dir = "results"
job_id = 0 # Similarly do for 1, 2 and 3


# Find all part files for this job
pattern = f"*EVCSAttacks_8500_job{job_id}_part*.pkl.gz"
job_files = sorted(glob(os.path.join(results_dir, pattern)))

# Output file (gzip-wrapped pickle stream)
output_file = os.path.join(results_dir, f"EVCSAttacks_8500_job{job_id}_merged.pkl.gz")

print(f"\nStreaming merge for job {job_id} into {output_file}", flush=True)
total_count = 0

with gzip.open(output_file, 'wb') as out_f:
    for file in job_files:
        print(f"  - Processing {file}", flush=True)
        with gzip.open(file, 'rb') as f:
            data = pickle.load(f)
            for scenario in data:
                pickle.dump(scenario, out_f)
                print(f"Scenario:{total_count}", flush = True)
                total_count += 1

print(f"\nStreamed {total_count} scenarios into: {output_file}", flush=True)