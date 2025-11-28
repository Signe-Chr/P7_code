import os
import numpy as np
import pyroomacoustics as pra
from tqdm import tqdm
from Dataset_class import J

def unit_vector_to_angles(v):
    x, y, z = v
    azimuth = np.arctan2(y, x)       # angle in XY-plane
    colatitude = np.arccos(z)        # angle down from +Z axis
    return azimuth, colatitude

def prepare_rir_input(IR, n_mics, n_srcs, max_length=512):
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

def setup_acoustic_scenario(sources, 
                        mic_positions_list, 
                        bright_zone_mics_index, 
                        dark_zone_mics_index, 
                        fs_target, 
                        room_dim, 
                        rt60,
                        mic_directions, 
                        user_rotation,
                        phone_tilt):
    """
    Sets up a pyroomacoustics simulation environment (ShoeBox) and computes RIRs.

    Returns:
        tuple: (IR, M_B, M_D)
            IR (list of lists): Room Impulse Responses.
            M_B (int): Number of bright zone microphones.
            M_D (int): Number of dark zone microphones.
    """
    sources_list = sources 

    M_B, M_D = len(bright_zone_mics_index), len(dark_zone_mics_index)

    # Define Room
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)
    room = pra.ShoeBox(
        room_dim,
        fs=fs_target,
        materials=pra.Material(e_absorption),
        max_order=max_order)

    # Define and Add Microphone Grid
    mic_positions = np.array(mic_positions_list).T
    mic_array = pra.MicrophoneArray(
        mic_positions,
        room.fs)
    room.add_microphone_array(mic_array)

    final_mic_dir = np.array([np.sin(user_rotation), -np.cos(user_rotation), 0])
    final_mic_dir /= np.linalg.norm(final_mic_dir)
    az, col = unit_vector_to_angles(final_mic_dir)
    room.mic_array.set_directivity(mic_directions[:-1]+[pra.directivities.HyperCardioid(
                    pra.directivities.DirectionVector(az, col, degrees=False))])
    
    # Add Sources (Loudspeakers)
    source_dir_vecs = [np.array([np.sin(phone_tilt)*np.cos(user_rotation), np.sin(phone_tilt)*np.sin(user_rotation), -np.cos(phone_tilt)]),
                       np.array([np.sin(phone_tilt)*np.cos(user_rotation), np.sin(phone_tilt)*np.sin(user_rotation), -np.cos(phone_tilt)]),
                       -final_mic_dir]
    source_dir_vecs[0] /= np.linalg.norm(source_dir_vecs[0])
    source_dir_vecs[1] /= np.linalg.norm(source_dir_vecs[1])
    source_directions = [pra.directivities.HyperCardioid(
                            pra.directivities.DirectionVector(
                                *unit_vector_to_angles(source_dir_vecs[0]), degrees=False)),
                         pra.directivities.HyperCardioid(
                            pra.directivities.DirectionVector(
                                *unit_vector_to_angles(source_dir_vecs[1]), degrees=False)),
                         pra.directivities.HyperCardioid(
                            pra.directivities.DirectionVector(
                                *unit_vector_to_angles(source_dir_vecs[2]), degrees=False))]
    for direc, s in enumerate(sources_list):
        room.add_source(s, directivity=source_directions[direc])

    # Compute RIRs
    #print(f"Computing RIRs for {mic_positions.shape[1]} mics (Bright: {M_B}, Dark: {M_D}) and {len(sources_list)} sources...")
    room.compute_rir()

    # RIRs are stored in room.rir: room.rir[mic_index][source_index]
    pre_IR = room.rir 
    IR = prepare_rir_input(pre_IR, mic_positions.shape[1], len(sources_list), max_length=512)

    return IR, M_B, M_D

def sources_mics(R, Center, M_D):
    # --- Save Coefficients ---
    #np.save(out_q_path, q_matrix)
    #print(f"Successfully designed filter and saved q_matrix to {out_q_path} in {time.perf_counter() - t_start_total:.2f} s")
    #IR = prepare_rir_input(IR_old, n_mics, n_srcs, max_length=512)
    mic_positions_list = []
    direction_list = []
    dark_zone_mics_index = []
    for i in range(M_D):
        angle = 2 * np.pi * i / M_D
        mic_positions_list.append([R * np.cos(angle) + Center[0],
                                   R * np.sin(angle) + Center[1],
                                   Center[2]])
        dir_vec = np.array([-np.cos(angle), -np.sin(angle), 0])
        dir_vec /= np.linalg.norm(dir_vec)
        direction_list.append(pra.directivities.HyperCardioid(
            pra.directivities.DirectionVector(*unit_vector_to_angles(dir_vec), degrees=False)))
        dark_zone_mics_index.append(i)
    
    mic_positions_list.append([Center[0], Center[1]-0.1, Center[2]])
    dir_vec = np.array([0, -1, 0])
    direction_list.append(pra.directivities.HyperCardioid(
        pra.directivities.DirectionVector(*unit_vector_to_angles(dir_vec), degrees=False)))
    bright_zone_mics_index = [M_D]

    sources_position_list = [[Center[0]-0.04, Center[1]-0.12, Center[2]-0.16],
                             [Center[0]+0.04, Center[1]-0.12, Center[2]-0.16],
                             [Center[0]     , Center[1]-0.12, Center[2]]]
    return sources_position_list, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index, direction_list

def archive_RIRs(archive_path, key_name, sources_position, rt60, IR, mic_positions,
                     room_dim, spatial_position, R, user_orientation, phone_tilt,
                     bright_zone_mics_index, dark_zone_mics_index):
    dict_update = {
        'J': J, # Order of filter
        'N': N, # Samples in x_input (=1 if dirac delta)
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

def generate_configurations(rooms, RT60s, user_rotations, tilt_rotations, dark_mic_radius, z_height, M_D):
    #total_iterations = len(RT60s) * len(rooms) * len(user_rotations) * len(tilt_rotations) * 3
    #loop = tqdm(total=total_iterations)
    #pool = mp.Pool(processes=mp.cpu_count()-1)
    args = []
    for r, room_dim in enumerate(rooms):
        spatial_positions = [
                [room_dim[0]/2                   , room_dim[1]/2                   , z_height],   # 0 — Center
                [room_dim[0]/2                   , room_dim[1]-dark_mic_radius-0.05, z_height],   # 1 — Up against one wall
                [room_dim[0]-dark_mic_radius-0.05, room_dim[1]-dark_mic_radius-0.05, z_height],   # 2 — Corner
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
                        args.append((orientation_source_final, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index,
                            x_input, RT60, mic_directions, user_rotation, fs_target, room_dim,
                            i, ii, iii, iv, r, save_path, spatial_position, dark_mic_radius, tilt_rotation))
    return args

M_D = 12
fs_target = 16000
dark_mic_radius = 0.5
rooms = [[1.8, 2, 2.3], [4.9, 5.7, 2.5], [8.8, 7, 3]]
z_height = 1.43
RT60s = np.linspace(0.3, 0.9, 4)
user_rotations = [0, np.pi/2, np.pi, np.pi*3/2]
tilt_rotations = [np.deg2rad(15), np.deg2rad(45), np.deg2rad(75)]
x_input = np.array([1])
N = len(x_input)
if __name__ == "__main__":
    save_path = "Data Archive"
    args = generate_configurations(rooms, RT60s, user_rotations, tilt_rotations, dark_mic_radius, z_height, M_D)
    start = 140
    for arg in tqdm(args[start:], initial=start, total=len(args)):
        (orientation_source_final, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index,
                            x_input, RT60, mic_directions, user_rotation, fs_target, room_dim,
                            i, ii, iii, iv, r, save_path, spatial_position, dark_mic_radius, phone_tilt) = arg
        m = f"RIR_{r}_{i}_{ii}_{iii}_{iv}"

        if os.path.exists(os.path.join(save_path, m+".npy")):
            print("Skipped already existing configuration")

        else:
            IR = setup_acoustic_scenario(orientation_source_final, 
                                mic_positions_list, 
                                bright_zone_mics_index, 
                                dark_zone_mics_index, 
                                fs_target, 
                                room_dim, 
                                RT60,
                                mic_directions, 
                                user_rotation,
                                phone_tilt)[0]

            archive_RIRs(save_path, m, orientation_source_final,
                RT60, IR, mic_positions_list, room_dim,
                spatial_position, dark_mic_radius, user_rotation, phone_tilt,
                bright_zone_mics_index, dark_zone_mics_index)