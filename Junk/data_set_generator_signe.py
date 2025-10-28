import numpy as np
import os
import scipy.io.wavfile as wavfile
import torch
import pyroomacoustics as pra
#from VAST_filter_coefficients import design_vast_filter
from tqdm import tqdm
from Junk.VAST_filter_coefficients_optimized_signe import design_vast_filter
from alive_progress import alive_bar


J = 1024
N = 2000#len(wav)
V = int(J*3/2)
mu = 1
reg_eps=1*10**(-6)
fs_target=16000
absorption=0.2
max_order=10
reg_eps=1e-6
target_amplitude = 0.080792
dark_mic_radius = 1.0
room_dim = [10,10,10]
z_height=1.7
RT60s = [0.5*i for i in range(5, 10)]
N_mics=12

wav_path = "relaxing-guitar-loop-v5-245859.wav"

out_q_path = "VAST_filter_archive.npy"

fs_wav, wav = wavfile.read(wav_path)

wav = np.mean(wav, axis=1)

wav = wav[5*44100:7*44100]


spatial_positions = [
    [room_dim[0]/2   , room_dim[1]/2   , z_height],  # 0 — Center
    [room_dim[0]/2   , room_dim[1]-1.1 , z_height],  # 1 — Up against one wall
    [room_dim[0]-1.1 , room_dim[1]-1.1 , z_height],  # 2 — Corner
]

def sources_mics(R, Center, N_mics):
    mic_positions_list = []
    direction_list = []
    dark_zone_mics_index = []
    for i in range(N_mics):
        angle = 2 * np.pi * i / N_mics
        mic_positions_list.append([R * np.cos(angle) + Center[0],
                                   R * np.sin(angle) + Center[1],
                                   Center[2]])
        direction_list.append(pra.directivities.HyperCardioid(
             pra.directivities.DirectionVector(np.pi-angle, degrees=False)
        ))
        dark_zone_mics_index.append(i)
    
    mic_positions_list.append([Center[0], Center[1], Center[2]+0.5])
    direction_list.append(pra.directivities.HyperCardioid(
         pra.directivities.DirectionVector(-90)
    ))
    bright_zone_mics_index = [N_mics]

    sources_position_list = [[Center[0] + 0.2, Center[1]-0.1, Center[2]],
                             [Center[0] + 0.2, Center[1]+0.1, Center[2]],
                             [Center[0] + 0.2, Center[1], Center[2]+0.2]]
    return sources_position_list, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index, direction_list

def archive_q_matrix(q_matrix, archive_path, key_name, sources_position, rt60, IR, mic_positions, spatial_position, R,user_orientation,phone_tilt,bright_zone_mics_index,dark_zone_mics_index):
            dict_update = {
                'q_matrix': q_matrix, 
                'J': J, #Order of filter
                'N': N, #Samples in wav file
                'V': V,
                'mu': mu,
                'room_dim': room_dim,
                'sources_position': sources_position,
                'mic_positions': mic_positions,
                'bright_zone_mics_index': bright_zone_mics_index,
                'dark_zone_mics_index': dark_zone_mics_index,
                'RT60': rt60,
                'User_orientation': user_orientation,
                'Phone_tilt': phone_tilt,
                'Spatial_position': spatial_position,
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
                        pass
                        # Handle case where file might contain a single raw array
                        #print(f"Warning: Existing file at {archive_path} does not look like a dictionary archive. Starting new archive.")
                except Exception as e:
                    pass
                    #print(f"Warning: Could not load existing archive at {archive_path} due to error: {e}. Starting new archive.")
            
            # Update the dictionary with the new matrix
            archive_dict[key_name] = dict_update
            
            # Save the updated dictionary back, overwriting the old file
            np.save(archive_path, archive_dict, allow_pickle=True)
            print(f"Archived filter under key '{key_name}' and saved updated archive to {archive_path}.")

def compute_center(points):
    points = np.asarray(points, dtype=float)    
    center = np.mean(points, axis=0)
    return center



user_rotations=[np.pi/2,np.pi,np.pi*3/2,2*np.pi]
tilt_rotations=[np.deg2rad(15),np.deg2rad(45),np.deg2rad(75)]



# Outer loop: RT60
# Compute the total number of iterations across all nested loops
total_iterations = len(RT60s) * len(spatial_positions) * len(user_rotations) * len(tilt_rotations)

with alive_bar(total_iterations, title='Designing VAST Filters', spinner='dots_waves') as bar:
    for i, RT60 in enumerate(RT60s):
        for ii, spatial_position in enumerate(spatial_positions):
            for iii, user_rotation in enumerate(user_rotations):
                for iv, tilt_rotation in enumerate(tilt_rotations):

                    # Compute sources and mic setup
                    sources_position_list, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index, mic_directions = \
                        sources_mics(dark_mic_radius, spatial_position, N_mics)

                    # Compute orientations
                    center_sources = np.mean(sources_position_list, axis=0)
                    user_orientation = np.array([
                        [np.cos(user_rotation), -np.sin(user_rotation), 0],
                        [np.sin(user_rotation),  np.cos(user_rotation), 0],
                        [0, 0, 1]
                    ])
                    orientation_source_temp = np.matmul(user_orientation, np.array(sources_position_list) - center_sources.T)
                    rotation_x = np.array([
                        [1, 0, 0],
                        [0, np.cos(tilt_rotation), -np.sin(tilt_rotation)],
                        [0, np.sin(tilt_rotation),  np.cos(tilt_rotation)]
                    ])
                    orientation_source_final = np.matmul(rotation_x, orientation_source_temp) + center_sources.T

                    # Design the VAST filter
                    q, IR = design_vast_filter(
                        orientation_source_final, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index,
                        wav_path, RT60, mic_directions, user_rotation, fs_target, J, N,
                        V, mu, room_dim, reg_eps, target_amplitude
                    )

                    # Archive the result
                    m_key = f"VAST_{i}_{ii}_{iii}_{iv}"
                    archive_q_matrix(q, out_q_path, m_key,
                                     orientation_source_final, RT60, IR, mic_positions_list,
                                     spatial_position, dark_mic_radius, user_rotation,
                                     tilt_rotation, bright_zone_mics_index, dark_zone_mics_index)

                    # Update the alive-progress bar text
                    bar.text = (
                        f"RT60={RT60:.1f}, Pos={np.round(spatial_position, 1)}, "
                        f"Rot={np.rad2deg(user_rotation):.0f}°, Tilt={np.rad2deg(tilt_rotation):.0f}°, "
                        f"Key={m_key}"
                    )
                    bar()  # increment the progress bar

