
"""
Post processing for parallel
"""

import os
import pickle
import random

# Directory where your SLURM jobs saved the .pkl files
input_dir = 'results'
output_file = 'merged_shuffled_PVAttacks_8500.pkl'

# Collect all .pkl files from the directory
pkl_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.pkl')]

# Initialize merged list
merged_scenarios = []

# Load and combine all scenarios
for pkl_file in pkl_files:
    print(f"Loading {pkl_file}")
    with open(pkl_file, 'rb') as file:
        scenarios = pickle.load(file)
        merged_scenarios.extend(scenarios)

print(f"Total scenarios before shuffling: {len(merged_scenarios)}")

# Shuffle the combined list
random.shuffle(merged_scenarios)

# Save the merged + shuffled list
with open(os.path.join(input_dir, output_file), 'wb') as outfile:
    pickle.dump(merged_scenarios, outfile)

print(f"Saved merged and shuffled scenarios to {output_file}")
