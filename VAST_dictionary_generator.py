from VAST_function import design_vast_filter
import numpy as np
import os

room_dim=[8.12, 7.35, 3.00]
Center = [room_dim[0]/2, room_dim[1]/2, room_dim[2]/2]

J=256
N=2000
V=4
mu=0.5
fs_target=16000
absorption=0.2
max_order=10
reg_eps=1e-6
target_amplitude=0.3536
R = 1.0

wav_path = "Signe_sang.wav"
out_q_path = "q_solution_td.npy"

def sources_mics(R, Center, N_mics):
    mic_positions_list = []
    bright_zone_mics_index = []
    for i in range(N_mics):
        mic_positions_list.append([R * np.cos(2 * np.pi * i / N_mics) + Center[0],
                                   R * np.sin(2 * np.pi * i / N_mics) + Center[1],
                                   Center[2]])
        bright_zone_mics_index.append(i)
    
    mic_positions_list.append([Center[0], Center[1], Center[2]])
    dark_zone_mics_index = [N_mics]

    sources_position_list = [[Center[0]- 0.2, Center[1]-0.1, Center[2]],
                             [Center[0]- 0.2, Center[1]+0.1, Center[2]],
                             [Center[0]- 0.2, Center[1], Center[2]+0.2]]
    return sources_position_list, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index


sources_position, mic_positions, bright_zone_mics_index, dark_zone_mics_index = sources_mics(R, Center, 12)

import matplotlib.pyplot as plt
x_angle = np.pi/2
y_angle = np.pi/3
sp = np.array(sources_position)
print(*sp.T)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(*sp.T, c="blue")


rotation_x = np.array([[1,               0,                0],
                       [0, np.cos(x_angle), -np.sin(x_angle)],
                       [0, np.sin(x_angle),  np.cos(x_angle)]])
rs = np.matmul(rotation_x, sp.T)
print(*rs)
ax.scatter(*rs, c="orange")

plt.show()

exit()  # Don't need to run code after testing

q_matrix = design_vast_filter(sources_position, mic_positions, bright_zone_mics_index, dark_zone_mics_index,
                          wav_path, fs_target=fs_target, J=J, N=N, 
                          V=V, mu=mu, room_dim=room_dim, absorption=absorption, 
                          max_order=max_order, reg_eps=reg_eps, target_amplitude=target_amplitude)

dict_update = {
        'q_matrix': q_matrix, 
        'J': J,
        'N': N,
        'V': V,
        'mu': mu,
        'room_dim': room_dim,
        'sources_position': sources_position,
        'mic_positions': mic_positions,
        'bright_zone_mics_index': bright_zone_mics_index,
        'dark_zone_mics_index': dark_zone_mics_index,
        'Center': Center,
        'R': R}


def archive_q_matrix(q_matrix, archive_path, key_name):
    """
    Loads an existing filter archive (dictionary), adds the new q_matrix 
    under 'key_name', and resaves the entire dictionary to the same file.

    Args:
        q_matrix (np.ndarray): The filter coefficients to save.
        archive_path (str): Path to the .npy archive file.
        key_name (str): The unique key to identify this matrix in the archive.
    """
    
    dict_update = {
        'q_matrix': q_matrix, 
        'J': J,
        'N': N,
        'V': V,
        'mu': mu,
        'room_dim': room_dim,
        'sources_position': sources_position,
        'mic_positions': mic_positions,
        'bright_zone_mics_index': bright_zone_mics_index,
        'dark_zone_mics_index': dark_zone_mics_index}
    
    archive_dict = {}
    
    if os.path.exists(archive_path):
        try:
            # Load existing dictionary from file
            # .item() is needed to extract the dict from the 0-D array wrap
            loaded_data = np.load(archive_path, allow_pickle=True)
            if loaded_data.ndim == 0:
                archive_dict = loaded_data.item()
            else:
                # Handle case where file might contain a single raw array
                print(f"Warning: Existing file at {archive_path} does not look like a dictionary archive. Starting new archive.")
        except Exception as e:
            print(f"Warning: Could not load existing archive at {archive_path} due to error: {e}. Starting new archive.")
    
    # Update the dictionary with the new matrix
    archive_dict[key_name] = dict_update
    
    # Save the updated dictionary back, overwriting the old file
    np.save(archive_path, archive_dict, allow_pickle=True)
    print(f"Archived filter under key '{key_name}' and saved updated archive to {archive_path}.")

archive_q_matrix(q_matrix, "VAST_filter_archive.npy", "VAST_example_1")