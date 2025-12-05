import os
import torch
import Cross_validation_models as cvm
from Test_train_split import load_test_train_data

# GLOBAL VARIABLES
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#---Load data and split into test and traning data---
data_test, data_train, data_val = load_test_train_data()
X_test = data_test[0]
X_train = data_train[0]
X_val = data_val[0]
filters_test = data_test[1]
filters_train = data_train[1]
filters_val = data_val[1]

def load_model():
    input_size = 9
    output_size = filters_train.shape[0] 
    model = cvm.FilterNet_classification(input_size, output_size).to(device)
    model.load_state_dict(torch.load("MLP_classification.pth", map_location=device))
    model.eval()
    return model

def validate_classification(X_test=X_test, Y_test = filters_test, Y_train = filters_train):
    model = load_model() # Choose model here (0-2)
    count = 0
    for i, configuration in enumerate(X_test):
        with torch.no_grad():
            output = model(configuration.unsqueeze(0).float())
            _, prediction = torch.max(output, 1)
            count += (prediction == i).item()
    print(f"{count}/{len(X_test)} correct.")
    print(f"The model has an accuracy of {count/len(X_test)*100:.2f}%.")

validate_classification(X_test=X_train, Y_test=filters_train)