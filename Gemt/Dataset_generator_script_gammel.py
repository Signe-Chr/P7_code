import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import scipy.io.wavfile as wavfile
import pyroomacoustics as pra
import multiprocessing as mp
from Gemt.VAST_filter_coefficients import design_vast_filter
from tqdm import tqdm

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
    
    mic_positions_list.append([Center[0], Center[1]-0.1, Center[2]])
    direction_list.append(pra.directivities.HyperCardioid(
         pra.directivities.DirectionVector(-90)
    ))
    bright_zone_mics_index = [N_mics]

    sources_position_list = [[Center[0]-0.04, Center[1]-0.15, Center[2]-0.16],
                             [Center[0]+0.04, Center[1]-0.15, Center[2]-0.16],
                             [Center[0]     , Center[1]-0.15, Center[2]]]
    return sources_position_list, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index, direction_list

def archive_q_matrix(q_matrix, archive_path, key_name, sources_position, rt60, IR, mic_positions, room_dim, spatial_position, R, user_orientation, phone_tilt, bright_zone_mics_index, dark_zone_mics_index):
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
            
            """archive_dict = {}
            
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
            archive_dict[key_name] = dict_update"""
            
            os.makedirs(archive_path, exist_ok=True)
            path = os.path.join(archive_path, f"{key_name}.npy")

            # Save the updated dictionary back, overwriting the old file
            np.save(path, dict_update, allow_pickle=True)
            #print(f"Archived filter under key '{key_name}' and saved updated archive to {archive_path}.")

def compute_center(points):
    points = np.asarray(points, dtype=float)    
    center = np.mean(points, axis=0)
    return center

def main(orientation_source_final, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index,
         wav, RT60, mic_directions, user_rotation, fs_target, J, N, V, mu, room_dim, reg_eps, target_amplitude,
         i, ii, iii, iv, r, out_q_path, spatial_position, dark_mic_radius, tilt_rotation):
    q, IR = design_vast_filter(
        orientation_source_final, mic_positions_list,
        bright_zone_mics_index, dark_zone_mics_index,
        wav, RT60, mic_directions, user_rotation,
        fs_target, J, N, V, mu, room_dim, reg_eps, target_amplitude
    )

    m = f"VAST_{r}_{i}_{ii}_{iii}_{iv}"  # room, spatial position, user orientation, phone tilt
    #print(m, datetime.datetime.now())
    
    archive_q_matrix(
        q, out_q_path, m, orientation_source_final,
        RT60, IR, mic_positions_list, room_dim,
        spatial_position, dark_mic_radius, user_rotation, tilt_rotation,
        bright_zone_mics_index, dark_zone_mics_index
    )
    return

J = 1024
V = 1
mu = 1
fs_target = 16000
reg_term = 1e-6
target_amplitude = 0.080792 # WTF IS THISSSSS?????
dark_mic_radius = 0.5
rooms = [[1.8, 2, 2.3], [4.9, 5.7, 2.5], [8.8, 7, 3]]
z_height = 1.7
RT60s = np.linspace(0.1, 0.9, 5)
user_rotations = [np.pi/2, np.pi, np.pi*3/2, 2*np.pi]
tilt_rotations = [np.deg2rad(15), np.deg2rad(45), np.deg2rad(75)]
N_mics = 12
if __name__ == "__main__":
    wav_path = "relaxing-guitar-loop-v5-245859.wav"
    fs_wav, wav = wavfile.read(wav_path)
    wav = wav[5*44100:7*44100]

    out_q_path = "ACC_filter_archive"

    iteration_count=0
    total_iterations = len(RT60s) * len(rooms) * len(user_rotations) * len(tilt_rotations) * 3
    loop = tqdm(total=total_iterations)
    pool = mp.Pool(processes=mp.cpu_count()-1)
    for r, room_dim in enumerate(rooms):
        spatial_positions = [
                [room_dim[0]/2              , room_dim[1]/2              , z_height],   # 0 — Center
                [room_dim[0]/2              , room_dim[1]-dark_mic_radius, z_height],   # 1 — Up against one wall
                [room_dim[0]-dark_mic_radius, room_dim[1]-dark_mic_radius, z_height],   # 2 — Corner
            ]
        for i, RT60 in enumerate(RT60s):
            for ii, spatial_position in enumerate(spatial_positions):
                sources_position_list, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index, mic_directions = sources_mics(
                    dark_mic_radius, spatial_position, N_mics
                )
                for iii, user_rotation in enumerate(user_rotations):
                    user_orientation = np.array([
                        [np.cos(user_rotation), -np.sin(user_rotation), 0],
                        [np.sin(user_rotation),  np.cos(user_rotation), 0],
                        [                    0,                      0, 1]
                    ])  # rotation around z-axis for bright zone ear mic

                    center_sources = np.mean(sources_position_list, axis=0)
                    orientation_source_temp = np.matmul(user_orientation, np.array(sources_position_list) - center_sources.T)
                    for iv, tilt_rotation in enumerate(tilt_rotations):
                        rotation_x = np.array([
                            [1,                     0,                      0],
                            [0, np.cos(tilt_rotation), -np.sin(tilt_rotation)],
                            [0, np.sin(tilt_rotation),  np.cos(tilt_rotation)]
                        ])
                        
                        orientation_source_final = np.matmul(rotation_x, orientation_source_temp)
                        orientation_source_final += center_sources.T
                        args = (orientation_source_final, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index,
                            wav, RT60, mic_directions, user_rotation, fs_target, J, N, V, mu, room_dim, reg_term, target_amplitude,
                            i, ii, iii, iv, r, out_q_path, spatial_position, dark_mic_radius, tilt_rotation)
                        pool.apply_async(main, args=args, callback=lambda _:loop.update(1))
    pool.close()
    pool.join()
    print("Done!")
