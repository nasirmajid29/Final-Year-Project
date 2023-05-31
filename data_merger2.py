import torch
import numpy as np
import os

data_files = [
    "/vol/bitbucket/nm219/data/take_off_weighing_scales_100eps_uncoloured_1.pt",
    "/vol/bitbucket/nm219/data/take_off_weighing_scales_100eps_uncoloured_2.pt"
]

chunk_size = 5000  # adjust this to change the chunk size

full_data = []
for data_file in data_files:
    if os.path.getsize(data_file) > 0:
        print(f"Loading {data_file}")
        data = torch.load(data_file)
        print(f"Concatenating {len(data)} elements...")
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            full_data.append(chunk)
            print(f"  Processed {i + len(chunk)} / {len(data)} elements...")
    else:
        print(f"Skipping empty file: {data_file}")

full_data = np.concatenate(full_data)

print(f"Saving concatenated data ({len(full_data)} elements)...")
torch.save(full_data, "/vol/bitbucket/nm219/data/take_off_weighing_scales_100eps_uncoloured.pt", pickle_protocol=4)
print("Done!")
