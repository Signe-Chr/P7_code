import torch.nn as nn

input_size = 9
output_size = 3072             

class FilterNet_regression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
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
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(128, output_size)  # logits
        )

    def forward(self, x):
        return self.net(x)  # raw logits

#cross validation
def cv_models(input_size, output_size):
    models = []
    neurons = [128, 256, 512]

    layers = [1,2,3]
    for layer in layers:
        for neuron1 in neurons:
            if layer == 1:
                class FilterNet(nn.Module):
                    def __init__(self, input_size, output_size):
                        super().__init__()
                        self.net = nn.Sequential(
                            nn.Linear(input_size, neuron1),
                            nn.ReLU(),
                            nn.Linear(neuron1, output_size)
                        )
                        self.name=f"Number of layers:{layer}, neurons layer 1:{neuron1}"

                    def forward(self, x):
                        return self.net(x)
                models.append(FilterNet(input_size, output_size))
            for neuron2 in neurons:
                if layer == 2:
                        class FilterNet(nn.Module):
                            def __init__(self, input_size, output_size):
                                super().__init__()
                                self.net = nn.Sequential(
                                    nn.Linear(input_size, neuron1),
                                    nn.ReLU(),
                                    nn.Linear(neuron1, neuron2),
                                    nn.ReLU(),
                                    nn.Linear(neuron2, output_size)
                                )
                                self.name=f"Number of layers:{layer}, neurons layer 1:{neuron1}, neurons layer 2:{neuron2}"
                                
                            def forward(self, x):
                                return self.net(x)
                        models.append(FilterNet(input_size, output_size))
                for neuron3 in neurons:
                    if layer == 3:
                        class FilterNet(nn.Module):
                            def __init__(self, input_size, output_size):
                                super().__init__()
                                self.net = nn.Sequential(
                                    nn.Linear(input_size, neuron1),
                                    nn.ReLU(),
                                    nn.Linear(neuron1, neuron2),
                                    nn.ReLU(),
                                    nn.Linear(neuron2, neuron3),
                                    nn.ReLU(),
                                    nn.Linear(neuron3, output_size)
                                )
                                self.name=f"Number of layers:{layer}, neurons layer 1:{neuron1}, neurons layer 2:{neuron2}, neurons layer 3:{neuron3}"
                            def forward(self, x):
                                return self.net(x)
                        models.append(FilterNet(input_size, output_size))
    return models