import os
import torch
from sklearn.model_selection import train_test_split
from Dataset_class import CustomDataset
import Dataset_generator_script as dgs

def load_test_train_data(test_size=0.25, random_seed=42):
    data_dir = "ACC_filter_archive"
    full_data = os.listdir(data_dir)

    # Perform train/test split with fixed random seed
    train_files, test_files = train_test_split(
        full_data, test_size=test_size, random_state=random_seed, shuffle=True
    )

    # Create dataset instances
    train_dataset = CustomDataset(data_dir, train_files)
    test_dataset = CustomDataset(data_dir, test_files)

    return test_dataset, train_dataset
