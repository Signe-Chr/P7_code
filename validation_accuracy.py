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

def load_model(a):
    model_name = model_names[a]  # vælg model her
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

def validate_filters(a, X_test=X_test, Y_test = filters_test, Y_train = filters_train):
    model = load_model(a) # Choose model here (0-2)
    count = 0
    for filter, configuration in zip(Y_test, X_test):
        with torch.no_grad():
            output = model(configuration.unsqueeze(0).float())
            if a == 1:
                output = _, prediction = torch.max(output, 1)
                output = Y_train[prediction]
            elif a == 2:
                output = torch.matmul(output.float(), Y_train.float())
            if torch.all(filter == output):
                count += 1
    print(f"The model has an accuracy of {count/len(X_test):.2f}%.")

# vælg model her (0-2)
#validate_filters(0)  # regression
validate_filters(1, X_test=X_train, Y_test=filters_train)  # classification
#validate_filters(2)  # interpolation