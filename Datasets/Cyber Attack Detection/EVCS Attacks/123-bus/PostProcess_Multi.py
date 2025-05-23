import os
import gzip
import pickle
import random
import gc
from glob import glob
 
# Define the results directory
results_dir = "results"
total_jobs = 4  # Adjust if more jobs
 
for job_id in range(total_jobs):
    # Find all part files for this job
    pattern = f"*EVCSAttacks_123_job{job_id}_part*.pkl.gz"
    job_files = sorted(glob(os.path.join(results_dir, pattern)))
 
 
    # Merge scenarios
    merged_scenarios = []
    print(f"\nMerging files for job {job_id}:", flush=True)
    for file in job_files:
        print(f"  - {file}", flush=True)
        with gzip.open(file, 'rb') as f:
            data = pickle.load(f)
            merged_scenarios.extend(data)
 
    # Shuffle merged scenarios
    random.shuffle(merged_scenarios)
 
    # Save to final compressed file
    output_file = os.path.join(results_dir, f"EVCSAttacks_123_job{job_id}_merged.pkl.gz")
    with gzip.open(output_file, 'wb') as f:
        pickle.dump(merged_scenarios, f)
 
    print(f"\n Merged and shuffled {len(merged_scenarios)} scenarios into: {output_file}", flush=True)
    # Clear memory and force garbage collection
    del merged_scenarios
    gc.collect()