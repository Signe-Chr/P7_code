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
        model.load_state_dict(torch.load("MLP_regression_50.pth", map_location=device))
    elif model_name == "classification":
        output_size = filters_train.shape[0] 
        model = cvm.FilterNet_classification(input_size, output_size).to(device)
        model.load_state_dict(torch.load("MLP_classification_550.pth", map_location=device))
    elif model_name == "interpolation":
        output_size = filters_train.shape[0]
        model = cvm.FilterNet_interpolation(input_size, output_size).to(device)
        model.load_state_dict(torch.load("MLP_interpolation.pth", map_location=device))
    model.eval()
    return model

def validate_filters(model_name, X_test=X_test, Y_test=filters_test, X_train=X_train, Y_train=filters_train):
    model = load_model(model_name)

    if model_name == "regression": # Test regression
        """
        print("Testing on test data")
        difference_per_filter = 0
        difference_per_entry = 0
        rmse_per_filter = []
        for filter, configuration in zip(Y_test, X_test):
            with torch.no_grad():
                output = model(configuration.unsqueeze(0).float())
                difference = torch.abs(filter-output)
                difference_per_filter += difference.sum()
                difference_per_entry += difference.mean()
                difference = (filter - output) ** 2
                #mse_per_filter += difference.sum()
                mse = torch.mean((filter - output.squeeze(0))**2)
                rmse = torch.sqrt(mse)
                rmse_per_filter.append(rmse.item())
        avg_rmse = sum(rmse_per_filter) / len(rmse_per_filter)
        print(f"Average difference per entry: {difference_per_entry/len(X_test):.8f}.")        
        print(f"Average difference per filter: {difference_per_filter/len(X_test):.4f}.")
        print(f"Average RMSE per filter: {avg_rmse:.6f}")
        print("")
        """
            
        rmse_per_filter = []

        with torch.no_grad():
            for filter_true, configuration in zip(Y_train, X_train):
                output = model(configuration.unsqueeze(0).float())
                mse = torch.mean((filter_true - output.squeeze(0))**2)
                rmse = torch.sqrt(mse)
                rmse_per_filter.append(rmse.item())

        # Convert to tensor
        rmse_per_filter = torch.tensor(rmse_per_filter)

        # Average RMSE across all filters
        avg_rmse = rmse_per_filter.mean().item()

        # Normalized RMSE (relative to mean absolute filter magnitude)
        mean_filter_magnitude = torch.mean(torch.abs(Y_train))
        nrmse = avg_rmse / mean_filter_magnitude

        print(f"Average RMSE per filter: {avg_rmse:.6f}")
        print(f"Normalized RMSE (relative to mean filter magnitude): {nrmse*100:.2f}%")

    if model_name == "classification": # Test classification
        """print("Testing on test data")
        count = 0
        for filter, configuration in zip(Y_test, X_test):
            with torch.no_grad():
                output = model(configuration.unsqueeze(0).float())
                _, prediction = torch.max(output, 1)
                output = Y_train[prediction]
                if torch.allclose(filter, output, atol=1e-6):
                    count += 1
        print(f"{count}/{len(X_test)} correct.")
        print(f"The model has an accuracy of {count/len(X_test)*100:.2f}%.")
        print("")"""
        count = 0
        print("Testing on training data")
        for filter, configuration in zip(Y_train, X_train):
            with torch.no_grad():
                output = model(configuration.unsqueeze(0).float())
                _, prediction = torch.max(output, 1)
                output = Y_train[prediction]
                if torch.allclose(filter, output, atol=1e-6):
                    count += 1
        print(f"{count}/{len(X_train)} correct.")
        print(f"The model has an accuracy of {count/len(X_train)*100:.2f}%.")

    elif model_name == "interpolation": # Test interpolation
        """print("Testing on test data")
        error = 0
        for filter, configuration in zip(Y_test, X_test):
            with torch.no_grad():
                output = model(configuration.unsqueeze(0).float())
                print(output)
                output = torch.matmul(output.float(), Y_train.float())
                error += torch.mean((output - filter)**2)
        print(f"The model has an average error per filter of {error/len(X_test):.2f}.")
        print("")"""
        error = 0
        print("Testing on training data")
        for filter, configuration in zip(Y_train, X_train):
            with torch.no_grad():
                output = model(configuration.unsqueeze(0).float())
                output = torch.matmul(output.float(), Y_train.float())
                error += torch.mean((output - filter)**2)
        print(f"The model has an average error per filter of {error/len(X_train):.2f}.")

        # for printing weights
        for filter, configuration in zip(Y_train, X_train):
            with torch.no_grad():
                output = model(configuration.unsqueeze(0).float())
                #print(output)
                #print(output.max())
                #print(torch.where(output==output.max()))

        
        rmse_per_filter = []
        with torch.no_grad():
            for filter, configuration in zip(Y_train, X_train):
                output = model(configuration.unsqueeze(0).float())
                output = torch.matmul(output.float(), Y_train.float())
                output = output.squeeze(0)
                mse = torch.mean((filter.float() - output.float())**2)
                rmse = torch.sqrt(mse)
                rmse_per_filter.append(rmse.item())

        rmse_per_filter = torch.tensor(rmse_per_filter)
        avg_rmse = rmse_per_filter.mean().item()
        mean_filter_magnitude = torch.mean(torch.abs(Y_train.float())).item()
        nrmse_dataset = avg_rmse / mean_filter_magnitude

        print(f"Average RMSE per filter: {avg_rmse:.6f}")
        print(f"Normalized RMSE (relative to dataset mean abs magnitude): {nrmse_dataset*100:.2f}%")


# vælg model her
chosen_model = "regression"
#validate_filters(chosen_model)


def print_target_stats(Y_train, Y_test):
    print("Train target stats:")
    print(" min", torch.min(Y_train).item(),
          " max", torch.max(Y_train).item(),
          " mean(abs)", torch.mean(torch.abs(Y_train)).item(),
          " std", torch.std(Y_train).item())

    print("\nTest target stats:")
    print(" min", torch.min(Y_test).item(),
          " max", torch.max(Y_test).item(),
          " mean(abs)", torch.mean(torch.abs(Y_test)).item(),
          " std", torch.std(Y_test).item())
    
#print_target_stats(filters_train, filters_test)
a = torch.mean(torch.abs(X_train), dim=0)
b =torch.std(X_train, dim=0)
#print(a,b)
validate_filters(chosen_model)