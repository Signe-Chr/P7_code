import torch
import numpy as np

L = 3       # Loudspeaker
J = 1024    # Filter order

dummy_input = np.array([0.2,    # Reverberation, float
                        1,      # Position,      encoded int - mid, wall, corner
                        0,      # Orientation,   degrees
                        15])    # Tilt,          degrees

dummy_q = np.zeros(L*J)

model = torch.nn.Sequential(
    torch.nn.Linear(4, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 1024)
)
