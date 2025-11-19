import os
from tqdm import tqdm
from numpy import load, save
from Dataset_generator_script import room_indices as ri

os.makedirs("Signes_data", exist_ok=True)
data_dir = "VAST_filter_archive_730"
full_data = os.listdir(data_dir)
for data in tqdm(full_data):
    i = int(data.split("_")[1])
    if i in ri:
        data_point = load(f"{data_dir}/{data}", allow_pickle=True).item()
        save(f"Signes_data/{data}", data_point, allow_pickle=True)