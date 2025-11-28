import torch.nn as nn
from Test_train_split import load_test_train_data
from torch.utils.data import DataLoader
from MLP_classification import train_epoch
import Cross_validation_models as cvm
import torch.optim as optim
import torch
import numpy as np
from torchsummary import summary


neurons = [128, 256, 512]

layers = [1,2,3]

num_folds=5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_test,data_train=load_test_train_data()


input_size = len(data_train[0][0])
output_size = 50

models=cvm.cv_models(input_size,output_size)
#for model in models:
 #   summary(model, input_size=(input_size,))
"""
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

dark_zone_mics_index=[0,1,2,3,4,5,6,7,8,9,10,11]
bright_zone_mics_index=[12]
"""




loss_1_layer_test=[]
loss_2_layers_test=[]
loss_3_layers_test=[]
loss_1_layer_train=[]
loss_2_layers_train=[]
loss_3_layers_train=[]
for model_ in models:
    train_error_list=[]
    test_error_list=[]
    for neuron in neurons:
        fold_train_error = []
        fold_test_error = []

        for folds in range(num_folds):
            data_test,data_train=load_test_train_data(random_seed=folds)
            data_test=data_test
            data_train=data_train

            X_train= data_train[0][0:50]
            X_test=data_test[0][0:50]
            X_train = X_train.to(torch.float32)
            X_test  = X_test.to(torch.float32)

            filters_train= data_train[1][0:50]
            filters_test=data_test[1][0:50]
            unique_filters_train, filter_indices_train = np.unique(filters_train, axis=0, return_inverse=True)
            filters_train = torch.from_numpy(filter_indices_train).long()  
            unique_filters_test, filter_indices_test = np.unique(filters_test, axis=0, return_inverse=True)
            filters_test = torch.from_numpy(filter_indices_test).long()  
            print(np.shape(filters_train))
            model=model_.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr = 1e-3)
            for epoch in range(1, 41):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, filters_train)
                loss.backward()
                optimizer.step()

            print("\n Training complete!")
            
            model.eval()
            with torch.no_grad():
                train_pred_logits = model(X_train)
                test_pred_logits = model(X_test)
                
                pred_classes_train = train_pred_logits.argmax(dim=1)
                pred_classes_test = test_pred_logits.argmax(dim=1)

                true_filters_train = filters_train
                true_filters_test = filters_test
                
                pred_filters_train = filters_train[pred_classes_train.numpy()]
                pred_filters_test = filters_test[pred_classes_test.numpy()]
                
                train_err=np.mean((true_filters_train-pred_filters_train) ** 2).item()
                test_err=np.mean((true_filters_test-pred_filters_test) ** 2).item()

                fold_train_acc.append(train_acc)
                fold_test_acc.append(test_acc)
            print(f"Neuron: {neuron}, Fold {folds+1} -- Train MSE: {train_acc:.4f}, Test MSE: {test_acc:.4f}")
        
        print(f"Neuron: {neuron} -- Average Train Acc: {np.mean(fold_train_acc):.4f}, "
          f"Average Test Acc: {np.mean(fold_test_acc):.4f}\n")
        train_acc_list.append(np.mean(fold_train_acc))
        test_acc_list.append(np.mean(fold_train_acc))
    for neuronss,train,test in zip(neurons,train_acc_list,test_acc_list):
        print(f"Average over 5 folds: (Neurons, Train MSE, Test MSE): ({neuronss}, {train:.20f}, {test:.20f})")