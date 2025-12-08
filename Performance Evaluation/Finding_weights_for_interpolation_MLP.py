import os, sys, torch
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from Loss_functions import MSE, Cosine_similarity, MSEP, AC_loss, compute_H_matrix
from performance_evaluation_unfiltered import compute_pressure_with_input as cpwi
from Test_train_split import load_test_train_data, load_wav_file, L, J, x_input_kronecker, indeces_bright, indeces_dark

def load_data_and_model(chosen_model, filters_dir = "Saved Filters Speech/"):
    model = torch.load(filters_dir + chosen_model +"_filters_speech.pt")

    #---Load data and split into test and traning data---
    data_test, data_train, data_val = load_test_train_data()

    filters_val=data_val[1]
    filters_train=data_train[1]
    
    n_srcs_val=data_val[4]
    n_srcs_train=data_train[4]
    
    RIRs_val=data_val[5]
    RIRs_train=data_train[5]
    
    return n_srcs_val, n_srcs_train, filters_val, filters_train, RIRs_val, RIRs_train, model

def loss_functions(true_filter, predicted_filter, rir_test, wav_input, B_idx, D_idx):
    L = 3
    J = 1024
    mse_loss = MSE(predicted_filter, true_filter)
    cosine_loss = Cosine_similarity(predicted_filter.reshape(1, L*J), true_filter.reshape(1, L*J))
    msep_loss_B, _ = MSEP(predicted_filter, true_filter, rir_test, wav_input, B_idx, D_idx)
    MSPE_loss = msep_loss_B
    H, _ = compute_H_matrix(rir_test)
    AC_los = AC_loss(predicted_filter, true_filter, H, B_idx, D_idx)
    return  mse_loss, cosine_loss, MSPE_loss, AC_los

def loss_function_evaluation(RIR_test, selected_filters, wav_input, indeces_bright_test, indeces_dark_test, true_filter):
    mse=[]
    cosine=[]
    MSEP=[]
    AC=[]

    for i in tqdm(range(RIR_test.shape[0]), disable=not sys.stdout.isatty()):
        rirs = RIR_test[i]           # [n_mics, n_srcs, n_rir_samples]
        n_srcs = 3
        filter_len = 1024
        filters_flat = selected_filters[i].float()  # [3072]
        filters = filters_flat.reshape(n_srcs, filter_len)  # [3, 1024]
        true_filters_flat=true_filter[i].float()
        true_filters=true_filters_flat.reshape(n_srcs,filter_len)

        # Compute metrics    
        mse_loss, cosine_loss, MSPE_loss, AC_loss = loss_functions(true_filters, filters, rirs, wav_input, indeces_bright_test, indeces_dark_test)
        
        # Append results
        mse.append(mse_loss)
        cosine.append(cosine_loss)
        MSEP.append(MSPE_loss)
        AC.append(AC_loss)
        
    return np.array(mse),np.array(cosine), np.array(MSEP),np.array(AC)

chosen_model = "baseline"
x_input = x_input_kronecker
n_srcs_val, n_srcs_train, filters_val, filters_train, RIRs_val, RIRs_train, model_filters = load_data_and_model(chosen_model)
mse,cosine, MSEP,AC=loss_function_evaluation(RIRs_val, model_filters, x_input, indeces_bright, indeces_dark, filters_val)
list_index=[i for i in range(29)]
plt.scatter(list_index,mse,label=r'$\mathcal{L}_1$')
plt.scatter(list_index,cosine,label=r'$\mathcal{L}_2$')
plt.scatter(list_index,MSEP, label=r'$\mathcal{L}_3$')
plt.scatter(list_index,AC,label=r'$\mathcal{L}_4$')
plt.xlabel('Filter index')
plt.ylabel('Loss function value')
plt.grid()
plt.legend()
plt.show()

mean_mse=np.mean(mse)
mean_cosine=np.mean(cosine)
mean_msep=np.mean(MSEP)
mean_AC=np.mean(AC)

median_mse=np.median(mse)
median_cosine=np.median(cosine)
median_msep=np.median(MSEP)
median_AC=np.median(AC)

print(f'Means: MSE: {mean_mse:.3f}, Cosine similairty: {mean_cosine:.3f}, MSEP: {mean_msep:.3f}, AC:{mean_AC:.3f}')
print(f'Medians: MSE: {median_mse:.3f}, Cosine similairty: {median_cosine:.3f}, MSEP: {median_msep:.3f}, AC:{median_AC:.3f}')