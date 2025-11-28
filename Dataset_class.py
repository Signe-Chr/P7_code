import numpy as np
from torch.utils.data import Dataset

L = 3       # Loudspeaker
J = 1024    # Filter order

class CustomDataset(Dataset):
    def __init__(self, data_path, filenames):
        self.data_path = data_path
        self.files = filenames
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = self.files[idx]
        dic = np.load(f"{self.data_path}/{file}", allow_pickle=True).item()
        bright_zone_mics_index = dic.get('bright_zone_mics_index', [])
        dark_zone_mics_index = dic.get('dark_zone_mics_index', [])
        srcs_pos = dic.get('sources_position', [])
        n_srcs = len(srcs_pos)
        IR = dic.get('IR', [0,0,0])

        rt60 = dic.get('RT60', 0)                         # 2.5
        phone_tilt = dic.get('Phone_tilt', 0)             # I radianer: 0.261, 0.785, 1.309
        user_orient = dic.get('User_orientation', 0)      # I radianer: 0, 1.57, 3.14, 4.71
        spatial = dic.get('Spatial_position', [0,0,0])    # (x, y, z): (5, 5 ,1.7) betyder i midten af rummet og i højde 1.7m
        spatial = np.array(spatial).ravel()                # flad ud til 1D
        room_dim = dic.get('room_dim', [0,0,0])
        X = np.concatenate([
            [rt60],
            [phone_tilt],
            [user_orient],
            spatial,
            room_dim
        ])
        y = np.ravel(dic.get('q_acc', np.zeros(L*J)))
        return X, y, bright_zone_mics_index, dark_zone_mics_index, n_srcs, IR, idx, srcs_pos