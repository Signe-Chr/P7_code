import torch
from Test_train_split import load_test_train_data


##---Perform random selection between filters for the entire test set---
data_test, data_train, data_val = load_test_train_data()
filters_test, filters_train = data_test[1], data_train[1]

def random_selection(filters_test, filters_train, seed_value):
    torch.manual_seed(seed_value)
    N_dic=filters_train.shape[0]
    N_test=len(filters_test)
    random_indices=torch.randint(low=0,high=N_dic,size=(N_test,))
    selected_filters=filters_train[random_indices]
    return data_test, selected_filters, random_indices

X_test, selected_filters, random_indices = random_selection(filters_test, filters_train, 42)

torch.save(selected_filters, "Saved Filters/random_filters.pt")
print("Saved filters")

"""
torch.save({
    'selected_filters': selected_filters,
    'random_indices': random_indices,
    'X_test': X_test
}, "Saved Filters/random_selection_filters.pt")
print("Saved filters")

Load data with
data = torch.load("Saved Filters/random_selection_filters.pt")
selected_filters = data['selected_filters']
random_indices = data['random_indices']

"""