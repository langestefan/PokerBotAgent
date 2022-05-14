import torch.nn as nn

class PasaNet(nn.Module): # CNN for image classification
    def __init__(self):
        super().__init__()
        
        self.cn1 = nn.Sequential(
            
            # Conv layer 1
            nn.Conv2d(1, 8, kernel_size = 3, padding = 2),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            
            # Pool layer
            nn.MaxPool2d(2, 2),
            
            # Conv layer 2
            nn.Conv2d(8, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            
            # Pool layer
            nn.MaxPool2d(2, 2),
            
            # Conv layer 3
            nn.Conv2d(32, 16, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            
            # Pool layer
            nn.MaxPool2d(2, 2),
            
            # Flatten
            nn.Flatten()
        )

        self.fc1 = nn.Sequential(
            
            # FC layer 1
            nn.Linear(16*16, 4),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.cn1(x)  
        x = self.fc1(x)
        return x