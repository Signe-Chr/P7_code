from VAST_function import design_vast_filter
import numpy as np
import os
import scipy.io.wavfile as wavfile
import VAST_function as vf
import torch
import pyroomacoustics as pra

wav_path = "Signe_sang.wav"
out_q_path = "PM_filter_archive.npy"

fs_wav, wav = wavfile.read(wav_path)
wav = wav[5*44100:7*44100]

room_dim=[8.12, 7.35, 3.00]#[10,10,10]
spatial_positions = [
    [          1.3,          6.05, room_dim[2]/2],  # 0 — upper left
    [room_dim[0]/2,          6.05, room_dim[2]/2],  # 1 — upper middle
    [         6.82,          6.05, room_dim[2]/2],  # 2 — upper right

    [          1.3, room_dim[1]/2, room_dim[2]/2],  # 3 — mid left
    [room_dim[0]/2, room_dim[1]/2, room_dim[2]/2],  # 4 — center
    [         6.82, room_dim[1]/2, room_dim[2]/2],  # 5 — mid right

    [          1.3,           1.3, room_dim[2]/2],  # 6 — bottom left
    [room_dim[0]/2,           1.3, room_dim[2]/2],  # 7 — bottom middle
    [         6.82,           1.3, room_dim[2]/2],  # 8 — bottom right
]

J = 1024
N = 2000#len(wav)
V = J*3
mu = 1
fs_target=16000
absorption=0.2
max_order=10
reg_eps=1e-6
target_amplitude = 0.080792
R = 1.0
N_mics = 13 - 1 #antal mics - 1 (ik spørg)
n_mics = 13
n_srcs = 3

def sources_mics(R, Center, N_mics):
    mic_positions_list = []
    dark_zone_mics_index = []
    for i in range(N_mics):
        mic_positions_list.append([R * np.cos(2 * np.pi * i / N_mics) + Center[0],
                                   R * np.sin(2 * np.pi * i / N_mics) + Center[1],
                                   Center[2]])
        dark_zone_mics_index.append(i)
    
    mic_positions_list.append([Center[0], Center[1], Center[2]+0.5])
    bright_zone_mics_index = [N_mics]

    sources_position_list = [[Center[0]- 0.2, Center[1]-0.1, Center[2]],
                             [Center[0]- 0.2, Center[1]+0.1, Center[2]],
                             [Center[0]- 0.2, Center[1], Center[2]+0.2]]
    return sources_position_list, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index

def rir_func(IR, n_mics, n_srcs, max_length=512):
    """
    Prepare RIR data as CNN input tensor
    Shape: (batch_size, channels, n_mics, n_srcs, time)
    """
    # Create a tensor to hold all RIRs
    rir_list = []

    for mic_idx in range(n_mics):
        rir_temp = []
        for src_idx in range(n_srcs):
            rir = IR[mic_idx][src_idx]
            # Truncate or zero-pad to max_length
            if len(rir) > max_length:
                rir = rir[:max_length]
            else:
                rir = np.pad(rir, (0, max_length - len(rir)))
            rir_temp.append(rir)
        rir_list.append(rir_temp)

    return np.array(rir_list)

def prepare_rir_input(IR, n_mics, n_srcs, max_length=512):
    """
    Prepare RIR data as CNN input tensor
    Shape: (batch_size, channels, n_mics, n_srcs, time)
    """
    # Create a tensor to hold all RIRs
    rir_tensor = torch.zeros(1, 1, n_mics, n_srcs, max_length)
    rir_list = []

    for mic_idx in range(n_mics):
        rir_temp = []
        for src_idx in range(n_srcs):
            rir = IR[mic_idx][src_idx]
            # Truncate or zero-pad to max_length
            if len(rir) > max_length:
                rir = rir[:max_length]
            else:
                rir = np.pad(rir, (0, max_length - len(rir)))
            rir_tensor[0, 0, mic_idx, src_idx, :] = torch.tensor(rir)
            rir_temp.append(rir)
        rir_list.append(rir_temp)

    
    return rir_tensor, np.array(rir_list)

def get_rir_and_clear_room(room_dims, source_pos, mic_pos, fs=16000, max_order=max_order):

    room = pra.ShoeBox(
        room_dims,
        fs=fs,
        materials=pra.Material(absorption),
        max_order=max_order)

    room.add_microphone_array(pra.MicrophoneArray(np.array(mic_pos).T, room.fs))

    for s in source_pos:
        room.add_source(s)

    room.compute_rir()

    # The result is RIR[mic_idx][source_idx]
    rir = room.rir

    return rir

def NN_input(N):
    NN_INPUT = []
    setup_information = []
    for i in spatial_positions[:N]:
        sources_position_list, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index = sources_mics(R, i, N_mics)
        IR = get_rir_and_clear_room(room_dim, sources_position_list, mic_positions_list, fs=16000, max_order=2)
        rir_tensor, rir_list = prepare_rir_input(IR, N_mics, n_srcs, max_length=512)
        NN_INPUT.append([rir_tensor, rir_list])
        setup_information.append([sources_position_list, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index])
    return NN_INPUT, setup_information


if __name__ == "__main__":
    opdeling = 4
    for j,position in enumerate(spatial_positions):
        Center=position
        sources_position, mic_positions, bright_zone_mics_index, dark_zone_mics_index = sources_mics(R, Center, 12)

        

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
        
        def archive_q_matrix(q_matrix, archive_path, key_name, sources_position):
            """
            Loads an existing filter archive (dictionary), adds the new q_matrix 
            under 'key_name', and resaves the entire dictionary to the same file.

            Args:
                q_matrix (np.ndarray): The filter coefficients to save.
                archive_path (str): Path to the .npy archive file.
                key_name (str): The unique key to identify this matrix in the archive.
            """
            IR, M_b, M_d = vf.setup_acoustic_scenario(sources_position, 
                            mic_positions, 
                            bright_zone_mics_index, 
                            dark_zone_mics_index, 
                            fs_target, 
                            room_dim, 
                            absorption, 
                            max_order)
            IR = rir_func(IR, len(mic_positions), len(sources_position), max_length=512)

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
                'R': R,
                'IR': IR}
            
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

        archive_q_matrix(q_matrix, out_q_path, "PM_key_(0, 0, 0, 0)", sources_position)

        for i in range(opdeling):
            r_0 = np.linalg.matrix_power(rotation_x, i)
            rs = np.matmul(r_0, sp-centroid_ori) + centroid_ori
            for ii in range(opdeling):
                r_1 = np.linalg.matrix_power(rotation_y, ii)
                rs = np.matmul(r_1, rs-centroid_ori) + centroid_ori
                for iii in range(opdeling):
                    r_2 = np.linalg.matrix_power(rotation_z, iii)
                    rs = np.matmul(r_2, rs-centroid_ori) + centroid_ori
                    print(rs.T)
                    q_matrix = design_vast_filter(rs.T, mic_positions, bright_zone_mics_index, dark_zone_mics_index,
                                wav_path, fs_target=fs_target, J=J, N=N, 
                                V=V, mu=mu, room_dim=room_dim, absorption=absorption, 
                                max_order=max_order, reg_eps=reg_eps, target_amplitude=target_amplitude)
                    archive_q_matrix(q_matrix, out_q_path, f"PM_key_{j,i,ii,iii}", rs.T)
