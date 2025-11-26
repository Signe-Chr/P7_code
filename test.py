import numpy as np, matplotlib.pyplot as plt
from Dataset_generator_script import RT60s, dark_mic_radius, M_D, sources_mics, tilt_rotations, user_rotations

spatial_positions = [
        [  5,   5, 1.5],   # 0 — Center
        [  5, 9.5, 1.5],   # 1 — Up against one wall
        [9.5, 9.5, 1.5],   # 2 — Corner
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

