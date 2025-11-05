from Dataset_class import CustomDataset, L, J
from torch.utils.data import DataLoader
from Dataset_generator_script import room_indices as ri
import torch
import os

#---Load data and split into test and traning data---
data_dir="Signes_data"
full_data = os.listdir(data_dir)
data_points = []
train_points = []
test_points = []
for data in full_data:
    i = int(data.split("_")[1])
    if (i in ri) and (i not in ri[::4]):
        train_points.append(data)
        data_points.append(data)
    else:
        test_points.append(data)
        data_points.append(data)
        
data_train=CustomDataset(data_dir,train_points)
data_train_loader=DataLoader(data_train,batch_size=len(data_train), shuffle=True)
data_test=CustomDataset(data_dir,test_points)
data_test_loader=DataLoader(data_test,batch_size=len(data_test), shuffle=True)

temp_var_train=[batch for batch in data_train_loader][0]
temp_var_test=[batch for batch in data_test_loader][0]

X_train=temp_var_train[0]
X_test=temp_var_test[0]

filters_train=temp_var_train[1]
filters_test=temp_var_test[1]

bright_zone_mics_index_train=temp_var_train[2]
bright_zone_mics_index_test=temp_var_test[2]

dark_zone_mics_index_train=temp_var_train[3]
dark_zone_mics_index_test=temp_var_test[3]

n_srcs_train=temp_var_train[4]
n_srcs_test=temp_var_test[4]

RIRs_train=temp_var_train[5]
RIRs_test=temp_var_test[5]

#---Perform random selection between filters for the entire test set---

def random_selection(X_test,dictionary,seed_value):
    torch.manual_seed(seed_value)
    N_dic=dictionary.shape[0]
    N_test=X_test.shape[0]
    random_indices=torch.randint(low=0,high=N_dic,size=(N_test,))
    selected_filters=dictionary[random_indices]
    return X_test,selected_filters, random_indices

X_test,selected_filters, random_indices=random_selection(X_test,filters_train,42)

torch.save({
    'selected_filters': selected_filters,
    'random_indices': random_indices,
    'X_test':X_test
}, "random_selection_data.pt")

"""
Load data with
data = torch.load("random_selection_data.pt")
selected_filters = data['selected_filters']
random_indices = data['random_indices']

"""