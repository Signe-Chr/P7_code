from VAST_function import design_vast_filter
import numpy as np
import os

room_dim=[10,10,10]#[8.12, 7.35, 3.00]
Center = [room_dim[0]/2, room_dim[1]/2, room_dim[2]/2]

J=1024
N=2000
V=10
mu=0.5
fs_target=16000
absorption=0.2
max_order=10
reg_eps=1e-6
target_amplitude=0.3536
R = 1.0

wav_path = "Signe_sang.wav"
out_q_path = "VAST_filter_archive.npy"

def sources_mics(R, Center, N_mics):
    mic_positions_list = []
    dark_zone_mics_index = []
    for i in range(N_mics):
        mic_positions_list.append([R * np.cos(2 * np.pi * i / N_mics) + Center[0],
                                   R * np.sin(2 * np.pi * i / N_mics) + Center[1],
                                   Center[2]])
        dark_zone_mics_index.append(i)
    
    mic_positions_list.append([Center[0], Center[1], Center[2]])
    bright_zone_mics_index = [N_mics]

    sources_position_list = [[Center[0]- 0.2, Center[1]-0.1, Center[2]],
                             [Center[0]- 0.2, Center[1]+0.1, Center[2]],
                             [Center[0]- 0.2, Center[1], Center[2]+0.2]]
    return sources_position_list, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index


sources_position, mic_positions, bright_zone_mics_index, dark_zone_mics_index = sources_mics(R, Center, 12)

opdeling = 5

x_angle = np.pi/opdeling   # How much to rotate the phone around the x axis
y_angle = np.pi/opdeling
z_angle = np.pi/opdeling

sp = np.array(sources_position).T   # Convert to array with every columns being a vector with coordinates for 1 source
centroid_ori = np.mean(sp, axis=1, keepdims=True)  # Find the centroid, the output needs to be a column vector

# Calculate rotation matrices
rotation_x = np.array([[1,               0,                0],
                       [0, np.cos(x_angle), -np.sin(x_angle)],
                       [0, np.sin(x_angle),  np.cos(x_angle)]])
rotation_y = np.array([[ np.cos(y_angle), 0, np.sin(y_angle)],
                       [               0, 1,               0],
                       [-np.sin(y_angle), 0, np.cos(y_angle)]])
rotation_z = np.array([[np.cos(z_angle), -np.sin(z_angle), 0],
                       [np.sin(z_angle),  np.cos(z_angle), 0],
                       [              0,                0, 1]])
full_rotation_matrix = np.matmul(np.matmul(rotation_x, rotation_y), rotation_z)
# Rotation happens around origo, so the sources are centered before rotating, after rotation return to center position
rs = np.matmul(full_rotation_matrix, sp-centroid_ori) + centroid_ori


"""# This section is only to check if the sources are rotated correctly, can be removed/skipped with no repurcussions
####################################################
centroid_rot = np.mean(rs, axis=1)
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(*centroid_ori, c="black", label="Centroid of non-rotated sources")
ax.scatter(*sp, c="blue", label="Non-rotated sources")
ax.scatter(*centroid_rot, c="green", label="Centroid of rotated sources")
ax.scatter(*rs, c="orange", label="Rotated sources")
ax.legend()
plt.show()
####################################################
"""

def archive_q_matrix(q_matrix, archive_path, key_name, sp):
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
        'sources_position': sp,
        'mic_positions': mic_positions,
        'bright_zone_mics_index': bright_zone_mics_index,
        'dark_zone_mics_index': dark_zone_mics_index,
        'Center': Center,
        'R': R}
    
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

q_matrix = design_vast_filter(sources_position, mic_positions, bright_zone_mics_index, dark_zone_mics_index,
                          wav_path, fs_target=fs_target, J=J, N=N, 
                          V=V, mu=mu, room_dim=room_dim, absorption=absorption, 
                          max_order=max_order, reg_eps=reg_eps, target_amplitude=target_amplitude)

archive_q_matrix(q_matrix, out_q_path, "VAST_example_(0, 0, 0)", sources_position)

exit()


for i in range(opdeling):
    r_0 = np.linalg.matrix_power(rotation_x, i)
    rs = np.matmul(r_0, sp-centroid_ori) + centroid_ori
    for ii in range(opdeling):
        r_1 = np.linalg.matrix_power(rotation_y, ii)
        rs = np.matmul(r_1, rs-centroid_ori) + centroid_ori
        for iii in range(opdeling):
            r_2 = np.linalg.matrix_power(rotation_z, iii)
            rs = np.matmul(r_2, rs-centroid_ori) + centroid_ori
            q_matrix = design_vast_filter(rs, mic_positions, bright_zone_mics_index, dark_zone_mics_index,
                          wav_path, fs_target=fs_target, J=J, N=N, 
                          V=V, mu=mu, room_dim=room_dim, absorption=absorption, 
                          max_order=max_order, reg_eps=reg_eps, target_amplitude=target_amplitude)
            archive_q_matrix(q_matrix, out_q_path, f"VAST_example_{i,ii,iii}", rs)



#print(sp_)
#plt.tight_layout()
#plt.show()


#exit()




#
#print(q_matrix[0][:10])



