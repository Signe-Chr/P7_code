import numpy as np
import pyroomacoustics as pra
import VAST_filter_coefficients as vfc
import Dataset_generator_script as dgs

sources, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index, mic_directions = dgs.sources_mics(dgs.dark_mic_radius, [1, 5, 1.7], 12)
mic_directions[:-1]+[pra.directivities.HyperCardioid(pra.directivities.DirectionVector(0))]
deep = vfc.setup_acoustic_scenario(sources, 
                        mic_positions_list, 
                        bright_zone_mics_index, 
                        dark_zone_mics_index, 
                        16000, 
                        [2, 10, 2], 
                        0.6,
                        mic_directions, 
                        90,
                        maxmax_order=20)

sources, mic_positions_list, bright_zone_mics_index, dark_zone_mics_index, mic_directions = dgs.sources_mics(dgs.dark_mic_radius, [5, 1, 1.7], 12)
wide = vfc.setup_acoustic_scenario(sources, 
                        mic_positions_list, 
                        bright_zone_mics_index, 
                        dark_zone_mics_index, 
                        16000, 
                        [10, 2, 2], 
                        0.6,
                        mic_directions, 
                        0,
                        maxmax_order=20)

opt = np.get_printoptions()
#np.set_printoptions(threshold=np.inf)
print(deep[0][-1][0])  # Get the ear microphone to compare
print(wide[0][-1][0])
np.set_printoptions(**opt)
