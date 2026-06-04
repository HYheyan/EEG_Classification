import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_shape=(1, 59, 282), hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        input_dim = 1
        for value in input_shape:
            input_dim *= value

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
