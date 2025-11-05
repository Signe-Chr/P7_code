import numpy as np
import torch.nn as nn

L = []

input_size = 9
output_size = 3072

Ü = [128, 256, 512]

Æ = [1,2,3]
for æ in Æ:
    for i in Ü:
        if æ == 1:
            class FilterNet(nn.Module):
                def __init__(self, input_size, output_size):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(input_size, i),
                        nn.ReLU(),
                        nn.Linear(i, output_size)
                    )

                def forward(self, x):
                    return self.net(x)
            L.append(FilterNet(input_size, output_size))
        for q in Ü:
            if æ == 2:
                    class FilterNet(nn.Module):
                        def __init__(self, input_size, output_size):
                            super().__init__()
                            self.net = nn.Sequential(
                                nn.Linear(input_size, i),
                                nn.ReLU(),
                                nn.Linear(i, q),
                                nn.ReLU(),
                                nn.Linear(q, output_size)
                            )

                        def forward(self, x):
                            return self.net(x)
                    L.append(FilterNet(input_size, output_size))
            for qq in Ü:
                if æ == 3:
                    class FilterNet(nn.Module):
                        def __init__(self, input_size, output_size):
                            super().__init__()
                            self.net = nn.Sequential(
                                nn.Linear(input_size, i),
                                nn.ReLU(),
                                nn.Linear(i, q),
                                nn.ReLU(),
                                nn.Linear(q, qq),
                                nn.ReLU(),
                                nn.Linear(qq, output_size)
                            )

                        def forward(self, x):
                            return self.net(x)
                    L.append(FilterNet(input_size, output_size))


                
#standalone regression
class FilterNet_(nn.Module):
                def __init__(self, input_size, output_size):
                    super().__init__()
                    self.net = nn.Sequential(
                    nn.Linear(input_size, 512),
                    nn.ReLU(),
                    nn.Linear(512, output_size))

                def forward(self, x):
                    return self.net(x)

model_ = FilterNet_(input_size, output_size)

#standalone interpolatio
input_size1 = 9
#output_size1 = #dict_size

class FilterNet_(nn.Module):
                def __init__(self, input_size, output_size):
                    super().__init__()
                    self.net = nn.Sequential(
                    nn.Linear(input_size, 512),
                    nn.ReLU(),
                    nn.Linear(512, output_size))

                def forward(self, x):
                    return self.net(x)

model_ = FilterNet_(input_size1, output_size1)