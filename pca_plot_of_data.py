import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from Test_train_split import load_test_train_data
import os
from sklearn.model_selection import train_test_split
from Dataset_class import CustomDataset
from torch.utils.data import DataLoader

data_dir = "Data Archive"
full_data = os.listdir(data_dir)

# Perform train/test split with fixed random seed
files_test=[]
files_train=[]
for file in full_data:
    room_id=file.split("_")[1]
    if int(room_id)==1:
        files_test.append(file)
    else:
        files_train.append(file)
files_train, files_validation = train_test_split(
    files_train, test_size=0.1, random_state=42, shuffle=True
)
small_room=[]
middle_room=[]
large_room=[]

rt_60_0=[]
rt_60_1=[]
rt_60_2=[]
rt_60_3=[]

spatial_0=[]
spatial_1=[]
spatial_2=[]

user_rotation_0=[]
user_rotation_1=[]
user_rotation_2=[]
user_rotation_3=[]
user_rotation_4=[]

phone_tilt_0=[]
phone_tilt_1=[]
phone_tilt_2=[]
phone_tilt_3=[]
for file in full_data:
    room_id=file.split("_")[1]
    rt_60_id=file.split("_")[2]
    spatial_id=file.split("_")[3]
    tilt_id=file.split("_")[4]
    user_rotation_id = file.split("_")[5].split(".")[0] 
    
    #Room
    if int(room_id)==0:
        small_room.append(file)
    elif int(room_id)==1:
        middle_room.append(file)
    elif int(room_id)==2:
        large_room.append(file)
        
     # --- RT60 ---
    if int(rt_60_id) == 0:
        rt_60_0.append(file)
    elif int(rt_60_id) == 1:
        rt_60_1.append(file)
    elif int(rt_60_id) == 2:
        rt_60_2.append(file)
    elif int(rt_60_id) == 3:
        rt_60_3.append(file)
    
    # --- Spatial ---
    if int(spatial_id) == 0:
        spatial_0.append(file)
    elif int(spatial_id) == 1:
        spatial_1.append(file)
    elif int(spatial_id) == 2:
        spatial_2.append(file)
    
    # --- User rotation ---
    if int(user_rotation_id) == 0:
        user_rotation_0.append(file)
    elif int(user_rotation_id) == 1:
        user_rotation_1.append(file)
    elif int(user_rotation_id) == 2:
        user_rotation_2.append(file)
    elif int(user_rotation_id) == 3:
        user_rotation_3.append(file)
    elif int(user_rotation_id) == 4:
        user_rotation_4.append(file)
    
    # --- Phone tilt ---
    if int(tilt_id) == 0:
        phone_tilt_0.append(file)
    elif int(tilt_id) == 1:
        phone_tilt_1.append(file)
    elif int(tilt_id)  == 2:
        phone_tilt_2.append(file)
    elif int(tilt_id) == 3:
        phone_tilt_3.append(file)
        
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA

# Define all splits as dictionaries
splits = {
    "room": {0: small_room, 1: middle_room, 2: large_room},
    "rt60": {0: rt_60_0, 1: rt_60_1, 2: rt_60_2, 3: rt_60_3},
    "spatial": {0: spatial_0, 1: spatial_1, 2: spatial_2},
    "user_rotation": {0: user_rotation_0, 1: user_rotation_1, 2: user_rotation_2, 3: user_rotation_3, 4: user_rotation_4},
    "phone_tilt": {0: phone_tilt_0, 1: phone_tilt_1, 2: phone_tilt_2, 3: phone_tilt_3}
}

# Store PCA results
filters_pca = {}

for split_name, categories in splits.items():
    filters_pca[split_name] = {}
    for cat_id, file_list in categories.items():
        # Skip empty lists
        if len(file_list) == 0:
            continue
        
        # Create dataset and dataloader
        dataset = CustomDataset(data_dir, file_list)
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        
        # Get batch
        data = [batch for batch in loader][0]
        filters = data[1]  # assuming filters are at index 1
        
        # PCA
        pca = PCA(n_components=3)
        filters_transformed = pca.fit_transform(filters)
        
        # Store
        filters_pca[split_name][cat_id] = filters_transformed

print(np.shape(filters_pca['rt60'][0]))
# --- 3D plot ---
splits_info = {
    "room": {0: "Small room", 1: "Middle room", 2: "Large room"},
    "rt60": {0: "RT60_0", 1: "RT60_1", 2: "RT60_2", 3: "RT60_3"},
    "spatial": {0: "Spatial_0", 1: "Spatial_1", 2: "Spatial_2"},
    "user_rotation": {0: "UserRot_0", 1: "UserRot_1", 2: "UserRot_2", 3: "UserRot_3", 4: "UserRot_4"},
    "phone_tilt": {0: "Tilt_0", 1: "Tilt_1", 2: "Tilt_2", 3: "Tilt_3"}
}

colors_list = ['red', 'green', 'blue', 'orange', 'purple', 'brown']  # more than enough for categories

# Create figure with 5 subplots
fig = plt.figure(figsize=(40, 8), constrained_layout=True) 

for i, (split_name, categories) in enumerate(splits_info.items(), 1):
    ax = fig.add_subplot(3, 2, i, projection='3d')  # 3x2 grid, subplot i
    
    for j, (cat_id, label) in enumerate(categories.items()):
        if cat_id in filters_pca[split_name]:
            data = filters_pca[split_name][cat_id]
            ax.scatter(
                data[:,0], data[:,1], data[:,2],
                s=20, depthshade=True, color=colors_list[j % len(colors_list)],
                label=label
            )
    
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend(fontsize=12)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Colormap for categories
cmap = cm.get_cmap('tab10')

# Create figure with 3 rows, 2 columns
fig, axes = plt.subplots(3, 2, figsize=(20, 18), constrained_layout=True)

# Flatten axes for easy iteration
axes = axes.flatten()

for i, (split_name, categories) in enumerate(splits_info.items()):
    ax = axes[i]
    
    for j, (cat_id, label) in enumerate(categories.items()):
        if cat_id in filters_pca[split_name]:
            data = filters_pca[split_name][cat_id]
            ax.scatter(
                data[:,0], data[:,1],
                s=50,
                color=cmap(j % 10),
                edgecolor='k',
                label=label,
                alpha=0.8
            )
    
    ax.set_xlabel("PC1" )
    ax.set_ylabel("PC2")
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1),fontsize=20)
    ax.grid(True)

# Hide the 6th subplot (empty)
axes[-1].axis('off')

plt.show()

