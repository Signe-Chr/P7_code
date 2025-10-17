from PM import PM_solution
import numpy as np
from scipy.signal import fftconvolve
from scipy.io import wavfile
import os
import time
import matplotlib.pyplot as plt

q_vec, q_matrix = PM_solution()
print("q_vec:", q_vec)
print("q_matrix:", q_matrix)