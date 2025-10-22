from PM import PM_solution
from ACC import ACC_solution
from Junk.VAST_function import VAST_solution
import Room_configuration as rc
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pesq

start_time = time.time()
VAST_q_vec, VAST_q_matrix = VAST_solution(rc.V, rc.mu, rc.lambda_vals, rc.U, rc.r_B)
end_time = time.time()
print(f"VAST_solution execution time: {end_time - start_time} seconds")

start_time = time.time()
PM_q_vec, PM_q_matrix = PM_solution()
end_time = time.time()
print(f"PM_solution execution time: {end_time - start_time} seconds")
#print("q_vec:", q_vec)
#print("q_matrix:", q_matrix)

start_time = time.time()
ACC_q_vec, ACC_q_matrix = ACC_solution()
end_time = time.time()
print(f"ACC_solution execution time: {end_time - start_time} seconds")
#print("ACC_q_vec:", ACC_q_vec)
#print("ACC_q_matrix:", ACC_q_matrix)