from VAST_filter_coefficients import VAST_solution
import numpy as np
import os
import scipy.io.wavfile as wavfile
import VAST_function as vf
import torch
import pyroomacoustics as pra
from VAST_filter_coefficients import design_vast_filter


J = 1024
N = 2000#len(wav)
V = J*3/2
mu = 1
fs_target=16000
absorption=0.2
max_order=10
reg_eps=1e-6
target_amplitude = 0.080792
dark_mic_radius = 1.0
room_dim = [10,10,10]
z_height=1.7
RT60s = [0.5*i for i in range(10)]
N_mics=12
wav_path="relaxing-guitar-loop-v5-245859.wav"

spatial_positions = [
    [room_dim[0]/2   , room_dim[1]/2   , z_height],  # 0 — Center
    [room_dim[0]/2   , room_dim[1]-1.1 , z_height],  # 1 — Up against one wall
    [room_dim[0]-1.1 , room_dim[1]-1.1 , z_height],  # 2 — Corner
]

def room_generator(room_dim, rt60, fs):
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(
        room_dim,
        fs=fs,
        materials=pra.Material(e_absorption),
        max_order=max_order)
    
    return room

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

    sources_position_list = [[Center[0] + 0.2, Center[1]-0.1, Center[2]],
                             [Center[0] + 0.2, Center[1]+0.1, Center[2]],
                             [Center[0] + 0.2, Center[1], Center[2]+0.2]]
    return sources_position_list, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index

def archive_q_matrix(q_matrix, archive_path, key_name, sources_position):
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



user_rotations=[np.pi/2,np.pi,np.pi*3/2,2*np.pi]
tilt_rotations_degree=[15,45,90]
tilt_rotations_radians=np.array(tilt_rotations_degree)*np.pi/180
for RT60 in RT60s:
    for spatial_position in spatial_positions:
        room = room_generator(room_dim, RT60, fs_target)
        sources_position_list, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index = sources_mics(dark_mic_radius, spatial_position, N_mics)
        room.add_microphone_array(pra.MicrophoneArray(np.array(mic_positions_list).T, room.fs))
        for s in sources_position_list:
            for user_rotation in user_rotations:
                user_orientation = np.array([[np.cos(user_rotation), -np.sin(user_rotation), 0],
                                [np.sin(user_rotation),  np.cos(user_rotation), 0],
                                [              0,                0, 1]]) #Add roation around z-axis for bright zone ear mic til JORD
                orientation_source_temp = np.matmul(user_orientation, np.array(s))
                for tilt_rotation in tilt_rotations_radians:
                    rotation_x = np.array([[1,               0,                0],
                            [0, np.cos(tilt_rotation), -np.sin(tilt_rotation)],
                            [0, np.sin(tilt_rotation),  np.cos(tilt_rotation)]])
                    orientation_source_final = np.matmul(rotation_x, orientation_source_temp)
                    room.add_source(orientation_source_final)
                    q = design_vast_filter(sources, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index,
                        wav_path, fs_target, J, N, 
                        V, mu, room_dim, absorption, 
                        max_order, reg_eps, target_amplitude = 1000)
                    

                    

                    
                    
            

            
        

