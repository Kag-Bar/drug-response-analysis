import torch.nn as nn

class SimpleClassificationNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleClassificationNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),  # Input layer
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),  # Hidden layer 1
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),  # Hidden layer 2
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),  # Output layer
        )
        self.softmax = nn.Softmax(dim=1)  # Softmax activation for probabilities

    def forward(self, x):
        x = self.model(x)
        return self.softmax(x)