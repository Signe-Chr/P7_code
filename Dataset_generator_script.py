import numpy as np
import os
import scipy.io.wavfile as wavfile
import pyroomacoustics as pra
from VAST_filter_coefficients import design_vast_filter
#from VAST_Filter_Design_Module_Optimized import design_vast_filter
import datetime
from tqdm import tqdm
import multiprocessing as mp



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
dark_mic_radius = 0.5
widths = range(2, 11)
depths = range(2, 11)
heights = range(4, 13)
rooms = [[width, depth, height/2] for width in widths for depth in depths for height in heights]+[[100, 100, 100]]  # if width<=depth (This code snippet can be added in the list loop if rooms like 2x10xz and 10x2xz are deemed equal)
z_height=1.7
#RT60s = [0.5*i for i in range(5, 10)]
RT60s = np.linspace(0.6, 0.8, 3)

user_rotations=[np.pi/2,np.pi,np.pi*3/2,2*np.pi]
tilt_rotations=[np.deg2rad(15),np.deg2rad(45),np.deg2rad(75)]

np.concatenate([[0.5], [np.deg2rad(15)], [np.pi/2], [1,1,1]])

N_mics=12

wav_path = "relaxing-guitar-loop-v5-245859.wav"

out_q_path = "VAST_filter_archive_730.npy"

fs_wav, wav = wavfile.read(wav_path)

wav = np.mean(wav, axis=1)

wav = wav[5*44100:7*44100]


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

    sources_position_list = [[Center[0]-0.1, Center[1]-0.2, Center[2]    ],
                             [Center[0]+0.1, Center[1]-0.2, Center[2]    ],
                             [Center[0]    , Center[1]-0.2, Center[2]+0.2]]
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
            #print(f"Archived filter under key '{key_name}' and saved updated archive to {archive_path}.")

def compute_center(points):
    points = np.asarray(points, dtype=float)    
    center = np.mean(points, axis=0)
    return center

def main(orientation_source_final, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index,
         wav, RT60, mic_directions, user_rotation, fs_target, J, N, V, mu, room_dim, reg_eps, target_amplitude,
         i, ii, iii, iv, out_q_path, spatial_position, dark_mic_radius, tilt_rotation):
    q, IR = design_vast_filter(
        orientation_source_final, mic_positions_list,
        bright_zone_mics_index, dark_zone_mics_index,
        wav, RT60, mic_directions, user_rotation,
        fs_target, J, N, V, mu, room_dim, reg_eps, target_amplitude
    )

    m = f"VAST_{i}_{ii}_{iii}_{iv}"  # room, spatial position, user orientation, phone tilt
    #print(m, datetime.datetime.now())
    
    archive_q_matrix(
        q, out_q_path, m, orientation_source_final,
        RT60, IR, mic_positions_list, room_dim,
        spatial_position, dark_mic_radius, user_rotation, tilt_rotation,
        bright_zone_mics_index, dark_zone_mics_index
    )
    pass

if __name__ == "__main__":
    iteration_count=0
    total_iterations = len(RT60s) * len(rooms) * len(user_rotations) * len(tilt_rotations) * 3
    loop = tqdm(total=total_iterations)
    #rooms_loop = tqdm(total=len(rooms))
    pool = mp.Pool(processes=mp.cpu_count()-1)
    for room_dim in rooms:
        if room_dim == rooms[-1]:
            spatial_positions = [[50, 50, z_height]]
        else:
            spatial_positions = [
                    [room_dim[0]/2              , room_dim[1]/2              , z_height],   # 0 — Center
                    [room_dim[0]/2              , room_dim[1]-dark_mic_radius, z_height],   # 1 — Up against one wall
                    [room_dim[0]-dark_mic_radius, room_dim[1]-dark_mic_radius, z_height],   # 2 — Corner
                ]
        #RT60_loop = tqdm(enumerate(RT60s), total=len(RT60s), leave=False)
        for i, RT60 in enumerate(RT60s):
            #RT60_loop.set_description(f"RT60 = {RT60}")
            #print(f"\nRT60 = {RT60}")
            #spatial_loop = tqdm(enumerate(spatial_positions), total=len(spatial_positions), leave=False)
            for ii, spatial_position in enumerate(spatial_positions):
                #spatial_loop.set_description(f"Spatial pos = {spatial_position}")
                #print(f"  Spatial position = {spatial_position}")
                sources_position_list, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index, mic_directions = sources_mics(
                    dark_mic_radius, spatial_position, N_mics
                )
                #rotation_loop = tqdm(enumerate(user_rotations), total=len(user_rotations), leave=False)
                for iii, user_rotation in enumerate(user_rotations):
                    #rotation_loop.set_description(f"User rotation = {round(user_rotation/np.pi*180, 2)} deg")
                    #print(f"    User rotation = {round(user_rotation / np.pi * 180, 2)} degrees")
                    user_orientation = np.array([
                        [np.cos(user_rotation), -np.sin(user_rotation), 0],
                        [np.sin(user_rotation),  np.cos(user_rotation), 0],
                        [                    0,                      0, 1]
                    ])  # rotation around z-axis for bright zone ear mic

                    center_sources = np.mean(sources_position_list, axis=0)
                    orientation_source_temp = np.matmul(user_orientation, np.array(sources_position_list) - center_sources.T)
                    #tilt_loop = tqdm(enumerate(tilt_rotations), total=len(tilt_rotations), leave=False)
                    for iv, tilt_rotation in enumerate(tilt_rotations):
                        #tilt_loop.set_description(f"Tilt = {round(tilt_rotation/np.pi*180, 2)} deg")
                        #print(f"      Phone tilt = {round(tilt_rotation / np.pi * 180, 2)} degrees")
                        rotation_x = np.array([
                            [1,                     0,                      0],
                            [0, np.cos(tilt_rotation), -np.sin(tilt_rotation)],
                            [0, np.sin(tilt_rotation),  np.cos(tilt_rotation)]
                        ])
                        
                        orientation_source_final = np.matmul(rotation_x, orientation_source_temp)
                        orientation_source_final += center_sources.T
                        args = (orientation_source_final, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index,
                            wav, RT60, mic_directions, user_rotation, fs_target, J, N, V, mu, room_dim, reg_eps, target_amplitude,
                            i, ii, iii, iv, out_q_path, spatial_position, dark_mic_radius, tilt_rotation)
                        pool.apply_async(main, args=args, callback=lambda _:loop.update(1))
                        #iteration_count += 1
                        #print(f'Completed iteration {iteration_count} out of {total_iterations}')
    #results = [pool.apply_async(main, (room, ), callback=lambda _:rooms_loop.update(1)) for room in rooms]
    pool.close()
    pool.join()
    #for room_dim in rooms_loop:
    #    rooms_loop.set_description(f"Room dim = {room_dim}")
    print("Done!")
