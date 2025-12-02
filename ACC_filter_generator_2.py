
import numpy as np
from scipy.linalg import eigh
import os
from tqdm import tqdm
import RIR_generator as RG  # must provide x_input, J, N

dark_index  = [0,1,2,3,4,5,6,7,8,9,10,11]
bright_index = [12]

J = RG.J  # filter length
# L will be read from IR shape; do not hard-code
# N should be the number of time samples to average over (from RG)
N = RG.N

def load_ir(file_path):
    rir_dict = np.load(file_path, allow_pickle=True).item()
    return rir_dict

def toeplitz_x(x, n, K, J):
    """Zero-padded convolution matrix: X[k,j] = x[n - k - j]"""
    X = np.zeros((K, J), dtype=float)
    Lx = len(x)
    for k in range(K):
        for j in range(J):
            idx = n - k - j
            if 0 <= idx < Lx:
                X[k, j] = x[idx]
    return X

def build_h(rir):
    """
    Build h ∈ R^{LK × M} with columns h_m = [h_{m1}^T, …, h_{mL}^T]^T
    rir shape assumed (M, L, K).
    """
    M, L, K = rir.shape
    h = np.zeros((L*K, M), dtype=float)
    for m in range(M):
        col = []
        for l in range(L):
            col.extend(rir[m, l, :].tolist())
        h[:, m] = np.array(col)
    return h  # shape (L*K, M)

def R_c(x, rir):
    """
    Compute R_B, R_D as (1/N) * sum_n H_B[n] H_B[n]^T and similarly for dark.
    """
    M, L, K = rir.shape
    h = build_h(rir)  # (L*K, M)

    R_B = None
    R_D = None

    for n in range(N):
        Xn_small = toeplitz_x(x, n, K, J)             # (K × J)
        Xn = np.kron(np.eye(L), Xn_small)             # (L*K × L*J)
        Hn = Xn.T                                     # (L*J × L*K)

        H_B_n = Hn @ h[:, bright_index]               # (L*J × M_B)
        H_D_n = Hn @ h[:, dark_index]                 # (L*J × M_D)

        RB_n = H_B_n @ H_B_n.T                        # (L*J × L*J)
        RD_n = H_D_n @ H_D_n.T                        # (L*J × L*J)

        if R_B is None:
            R_B = np.zeros_like(RB_n)
            R_D = np.zeros_like(RD_n)

        R_B += RB_n
        R_D += RD_n

    R_B /= N
    R_D /= N
    return R_B, R_D

def acc_coeffs(R_B, R_D):
    # Regularize R_D as recommended in the paper
    eps = 1e-6
    R_D_reg = R_D + eps * np.eye(R_D.shape[0])
    lam, U = eigh(R_B, R_D_reg)          # ascending order
    u1 = U[:, -1]                         # dominant generalized eigenvector
    return u1

def load_save():
    for name in tqdm(os.listdir("Data Archive")):
        d = load_ir(f"Data Archive/{name}")
        ir = d["IR"]                      # shape (M, L, K)
        R_B, R_D = R_c(RG.x_input, ir)
        q = acc_coeffs(R_B, R_D)          # length L*J
        # optional normalization (Euclidean or dark-zone energy):
        # q = q / np.linalg.norm(q)
        M, L, K = ir.shape
        d.update({"q_acc1": q.reshape(L, J)})
        np.save(f"Data Archive/{name}", d, allow_pickle=True)
        print(f"Saved filter {name}")

if __name__ == "__main__":
    load_save()