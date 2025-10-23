import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pesq
import torch


model = torch.load("filter_mlp_model_full.pth", weights_only=False)
