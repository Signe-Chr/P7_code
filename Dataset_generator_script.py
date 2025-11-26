import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import scipy.io.wavfile as wavfile
import pyroomacoustics as pra
import multiprocessing as mp
from VAST_filter_coefficients import design_vast_filter
from tqdm import tqdm

def sources_mics(R, Center, M_D):
    mic_positions_list = []
    direction_list = []
    dark_zone_mics_index = []
    for i in range(M_D):
        angle = 2 * np.pi * i / M_D
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
    bright_zone_mics_index = [M_D]

    sources_position_list = [[Center[0]-0.04, Center[1]-0.12, Center[2]-0.16],
                             [Center[0]+0.04, Center[1]-0.12, Center[2]-0.16],
                             [Center[0]     , Center[1]-0.12, Center[2]]]
    return sources_position_list, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index, direction_list

def archive_q_matrix(q_matrix, archive_path, key_name, sources_position, rt60, IR, mic_positions,
                     room_dim, spatial_position, R, user_orientation, phone_tilt,
                     bright_zone_mics_index, dark_zone_mics_index, J, N, V, mu):
    dict_update = {
        'q_matrix': q_matrix, 
        'J': J, # Order of filter
        'N': N, # Samples in x_input (=1 if dirac delta)
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
    
    os.makedirs(archive_path, exist_ok=True)
    path = os.path.join(archive_path, f"{key_name}.npy")

    # Save the updated dictionary back, overwriting the old file
    np.save(path, dict_update, allow_pickle=True)

def main(orientation_source_final, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index,
         wav, RT60, mic_directions, user_rotation, fs_target, J, N, V, mu, room_dim, reg_eps, target_amplitude,
         i, ii, iii, iv, r, out_path, spatial_position, dark_mic_radius, tilt_rotation):
    q, IR = design_vast_filter(
        orientation_source_final, mic_positions_list,
        bright_zone_mics_index, dark_zone_mics_index,
        wav, RT60, mic_directions, user_rotation,
        fs_target, J, N, V, mu, room_dim, reg_eps, target_amplitude
    )

    m = f"VAST_{r}_{i}_{ii}_{iii}_{iv}"  # room, spatial position, user orientation, phone tilt
    #print(m, datetime.datetime.now())
    
    archive_q_matrix(
        q, out_path, m, orientation_source_final,
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
user_rotations = [0, np.pi/2, np.pi, np.pi*3/2]
tilt_rotations = [np.deg2rad(15), np.deg2rad(45), np.deg2rad(75)]
M_D = 12
'''wav_path = "relaxing-guitar-loop-v5-245859.wav"
fs_wav, wav = wavfile.read(wav_path)
x_input = wav[5*44100:7*44100] / (np.max(np.abs(wav)) + 1e-12)'''
x_input = np.array([1])
N = len(x_input)
if __name__ == "__main__":
    out_q_path = "ACC_filter_archive"
    raw_out = "/TOTAL DATA"

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
                    dark_mic_radius, spatial_position, M_D
                )
                center_sources = np.mean(sources_position_list, axis=0)
                for iii, tilt_rotation in enumerate(tilt_rotations):
                    rotation_x = np.array([
                            [np.cos(tilt_rotation), 0, -np.sin(tilt_rotation)],
                            [                    0, 1,                      0],
                            [np.sin(tilt_rotation), 0,  np.cos(tilt_rotation)]
                        ])

                    orientation_source_temp = np.matmul(rotation_x, (np.array(sources_position_list)-center_sources).T).T
                    orientation_source_temp += center_sources
                    for iv, user_rotation in enumerate(user_rotations):
                        user_orientation = np.array([
                            [np.cos(user_rotation), -np.sin(user_rotation), 0],
                            [np.sin(user_rotation),  np.cos(user_rotation), 0],
                            [                    0,                      0, 1]
                        ])  # rotation around z-axis for bright zone ear mic
                        
                        orientation_source_final = np.matmul(user_orientation, (orientation_source_temp-np.array(spatial_position)).T).T
                        orientation_source_final += np.array(spatial_position)
                        args = (orientation_source_final, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index,
                            x_input, RT60, mic_directions, user_rotation, fs_target, J, N, V, mu, room_dim, reg_term, target_amplitude,
                            i, ii, iii, iv, r, out_q_path+raw_out, spatial_position, dark_mic_radius, tilt_rotation)
                        pool.apply_async(main, args=args, callback=lambda _:loop.update(1))
    pool.close()
    pool.join()
    print("Done creating total data!")
    from sklearn.model_selection import train_test_split
    total = os.listdir(out_q_path+raw_out)
    train, test = train_test_split(total)
    os.makedirs(out_q_path+"/Train", exist_ok=True)
    os.makedirs(out_q_path+"/Test", exist_ok=True)
    for data in train:
        temp = np.load(os.path.join(out_q_path+raw_out, data), allow_pickle=True)
        path = os.path.join(out_q_path+"/Train", data)
        np.save(path, temp, allow_pickle=True)
    for data in test:
        temp = np.load(os.path.join(out_q_path+raw_out, data), allow_pickle=True)
        path = os.path.join(out_q_path+"/Test", data)
        np.save(path, temp, allow_pickle=True)
    print("Done!")
