import torch.nn as nn
from Test_train_split import load_test_train_data
import Cross_validation_models as cvm
import torch.optim as optim
import torch
import numpy as np
from Loss_functions import Cosine_similarity, MSEP, AC_loss, MSE,compute_H_matrix
import matplotlib.pyplot as plt

p=1
neurons = [128, 256, 512]

layers = [1,2,3]

num_folds=5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_test,data_train=load_test_train_data()


input_size = len(data_train[0][0])
dark_zone_mics_index=[0,1,2,3,4,5,6,7,8,9,10,11]
bright_zone_mics_index=[12]

x_input = torch.zeros(1, 1, dtype=torch.float32)   # adjust as required
x_input[0, 0] = 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

if p==1:
    train_err_list=[]
    test_err_list=[]
    for neuron in neurons:
        fold_train_err = []
        fold_test_err= []
        print(f"\n--- Testing architecture: Layer 1: {neuron}")
        for folds in range(num_folds):
            print("folds", folds)
            data_test,data_train=load_test_train_data(random_seed=folds)
            data_test=data_test
            data_train=data_train

            X_train= data_train[0][0:50]
            X_test=data_test[0][0:50]
            X_train = X_train.to(torch.float32).to(device)
            X_test  = X_test.to(torch.float32).to(device)
            output_size=50
            
            RIRs_train=data_train[5]
            RIRs_test=data_test[5]

            filters_train= data_train[1][0:50]
            filters_test=data_test[1][0:50]
            #model=model_.to(device)
            model = torch.nn.Sequential(
                torch.nn.Linear(input_size, neuron),
                torch.nn.ReLU(),
                torch.nn.Linear(neuron, output_size)
            )
            
        
            optimizer = optim.Adam(model.parameters(), lr = 1e-3)
            for epoch in range(1, 41):
                print("epoch", epoch)
                total_loss=0
                total=0
                for i in range(len(X_train)):
                    X=X_train[i].unsqueeze(0).to(device)
                    filter=filters_train[i].unsqueeze(0).to(torch.float).to(device)
                    rir=RIRs_train[i]
                    optimizer.zero_grad()
                    coeff=model(X).to(device)
                    outputs = torch.matmul(filters_train.T.float() , coeff.T.float()).T.to(torch.float).to(device)
                    H=compute_H_matrix(rir)[0].to(device)
                    loss = 1/4*(MSE(outputs, filter) + Cosine_similarity(outputs, filter) + MSEP(outputs.reshape(3,1024), filter.reshape(3,1024), rir, x_input, bright_zone_mics_index, dark_zone_mics_index)[0] + AC_loss(outputs.reshape(3,1024), filter.reshape(3,1024), H, bright_zone_mics_index, dark_zone_mics_index))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * X.size(0) 
                    total += X.size(0)
            avg_loss = total_loss / total

            print("\n Training complete!")
            model.eval()
            with torch.no_grad():
                filters_loss_train = []
                filters_loss_test  = []

                for i in range(len(filters_train)):
                    n_srcs = 3
                    filter_len = 1024
                    feat_size = n_srcs * filter_len
                    
                    pred_filter_train=model(X_train)
                    pred_filter_test=model(X_test)


                    L1_train = MSE(filters_train,pred_filter_train)   
                    L1_test = MSE(filters_test,pred_filter_test)

                    L2_train = Cosine_similarity(filters_train,pred_filter_train)   # returns scalar
                    L2_test  = Cosine_similarity(filters_test,pred_filter_test)

                    rirs_train_i = RIRs_train[i]  # must be torch tensor (n_mics, n_srcs, n_rir_samples)
                    rirs_test_i  = RIRs_test[i]

                    if not isinstance(rirs_train_i, torch.Tensor):
                        rirs_train_i = torch.tensor(rirs_train_i, dtype=torch.float32)
                    if not isinstance(rirs_test_i, torch.Tensor):
                        rirs_test_i = torch.tensor(rirs_test_i, dtype=torch.float32)

                    x_input = torch.zeros(1, 1, dtype=torch.float32)   # adjust as required
                    x_input[0, 0] = 1.0

                    msep_B_train, msep_D_train = MSEP(filters_train,pred_filter_train, rirs_train_i, x_input, bright_zone_mics_index, dark_zone_mics_index)
                    msep_B_test,  msep_D_test  = MSEP(filters_test,pred_filter_test, rirs_test_i, x_input, bright_zone_mics_index, dark_zone_mics_index)

                    # If you only want bright-zone MSEP:
                    L3_train = msep_B_train
                    L3_test  = msep_B_test

                    L4_train = AC_loss(pred_filter_train,filters_train, compute_H_matrix(rirs_train_i)[0], bright_zone_mics_index, dark_zone_mics_index)
                    L4_test  = AC_loss(pred_filter_test,filters_test, compute_H_matrix(rirs_test_i)[0], bright_zone_mics_index, dark_zone_mics_index)

                    filters_loss_train.append(0.25 * (L1_train + L2_train + L3_train + L4_train))
                    filters_loss_test.append( 0.25 * (L1_test  + L2_test  + L3_test  + L4_test))

                # Average across samples
                train_err = torch.stack([t.flatten() for t in filters_loss_train]).mean().item()
                test_err  = torch.stack([t.flatten() for t in filters_loss_test]).mean().item()

                fold_train_err.append(train_err)
                fold_test_err.append(test_err)
            print(f"Neuron: {neuron}, Fold {folds+1} -- Train error: {train_err:.4f}, Test error: {test_err:.4f}")
        print(f"Neuron: {neuron} -- Average Train error: {np.mean(fold_train_err):.4f}, "
          f"Average Test error: {np.mean(fold_test_err):.4f}\n")
        train_err_list.append(np.mean(fold_train_err))
        test_err_list.append(np.mean(fold_test_err))
    for neuronss,train,test in zip(neurons,train_err_list,test_err_list):
        print(f"Average over 5 folds: (Neurons, Train error, Test error): ({neuronss}, {train:.20f}, {test:.20f})")
        
