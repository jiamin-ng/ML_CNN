import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # create the layers in the CNN model
        # Convolution Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        # Max Pooling Layer: downsample by factor of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layer 1 
        self.fc1 = nn.Linear(in_features=32 * 16 * 16, out_features=128)

        # Fully Connected Layer 2
        self.fc2 = nn.Linear(in_features=128, out_features=4)

        # Added Dropout Layer
        self.dropout = nn.Dropout(p=0.3)    # 30% dropout rate

        # Activation Function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply convolution + ReLU + pooling
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # Flatten feature maps 
        x = x.view(x.size(0), -1)

        # Fully connected layers + ReLU
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x) # Apply dropout after fc1

        # Output layer
        x = self.fc2(x)

        return x