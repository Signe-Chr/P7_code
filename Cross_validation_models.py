import torch.nn as nn

input_size = 9
output_size = 3072             

class FilterNet_regression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size))

    def forward(self, x):
        return self.net(x)

class FilterNet_interpolation(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size))
        
    def forward(self, x):
        return self.net(x)

class FilterNet_classification(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

