import os
import torch
import Cross_validation_models as cvm
from Test_train_split import load_test_train_data

# GLOBAL VARIABLES
save_dir = "Saved Filters"
os.makedirs(save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_names = ("regression", "classification", "interpolation")

#---Load data and split into test and traning data---
data_test, data_train, data_val = load_test_train_data()
X_test = data_test[0]
X_train = data_train[0]
X_val = data_val[0]
filters_test = data_test[1]
filters_train = data_train[1]
filters_val = data_val[1]

def load_model(model_name):
    input_size = 9
    if model_name == "regression":
        output_size = filters_train.shape[1]
        model = cvm.FilterNet_regression(input_size, output_size).to(device)
        model.load_state_dict(torch.load("MLP_regression.pth", map_location=device))
    elif model_name == "classification":
        output_size = filters_train.shape[0] 
        model = cvm.FilterNet_classification(input_size, output_size).to(device)
        model.load_state_dict(torch.load("MLP_classification.pth", map_location=device))
    elif model_name == "interpolation":
        output_size = filters_train.shape[0]
        model = cvm.FilterNet_interpolation(input_size, output_size).to(device)
        model.load_state_dict(torch.load("MLP_interpolation_without_weights.pth", map_location=device))
    model.eval()
    return model

def validate_filters(model_name, X_test=X_test, Y_test=filters_test, Y_train=filters_train):
    model = load_model(model_name)
    count = 0
    difference_per_filter = 0
    difference_per_entry = 0


    if model_name == "regression": # Test regression
        for filter, configuration in zip(Y_test, X_test):
            with torch.no_grad():
                output = model(configuration.unsqueeze(0).float())
                difference = output-filter
                difference_per_filter += (difference).sum()
                difference_per_entry += (difference).mean()
                print(difference_per_entry)

        print(f"The model has an average difference per entry of {difference_per_entry/len(Y_test[0]):.8f}.")        
        print(f"The model has an average difference per filter of {difference_per_filter/len(X_test):.4f}.")

    if model_name == "classification": # Test classification
        for filter, configuration in zip(Y_test, X_test):
            with torch.no_grad():
                output = model(configuration.unsqueeze(0).float())
                output = _, prediction = torch.max(output, 1)
                output = Y_train[prediction]
                if torch.all(filter == output):
                    count += 1
        print(f"{count}/{len(X_test)} correct.")
        print(f"The model has an accuracy of {count/len(X_test)*100:.2f}%.")

    elif model_name == "interpolation": # Test interpolation
        for filter, configuration in zip(Y_test, X_test):
            with torch.no_grad():
                output = model(configuration.unsqueeze(0).float())
                output = torch.matmul(output.float(), Y_train.float())
                if torch.all(filter == output):
                    count += 1
        print(f"{count}/{len(X_test)} correct.")
        print(f"The model has an accuracy of {count/len(X_test)*100:.2f}%.")

# vælg model her
chosen_model = "regression"
validate_filters(chosen_model, X_test=X_train, Y_test=filters_train)
