import os
from sklearn.model_selection import train_test_split
from Dataset_class import CustomDataset
from torch.utils.data import DataLoader


def load_test_train_data(test_size=0.25, random_seed=42):
    data_dir = "Data Archive"
    full_data = os.listdir(data_dir)

    # Perform train/test split with fixed random seed
    train_files, test_files = train_test_split(
        full_data, test_size=test_size, random_state=random_seed, shuffle=True
    )

    # Create dataset instances
    temp_var_train = CustomDataset(data_dir, train_files)
    temp_var_test = CustomDataset(data_dir, test_files)

    train_loader = DataLoader(temp_var_train, batch_size=len(temp_var_train), shuffle=False)
    test_loader = DataLoader(temp_var_test, batch_size=len(temp_var_test), shuffle=False)
    data_train = [batch for batch in train_loader][0]
    data_test = [batch for batch in test_loader][0]

    return data_test, data_train
