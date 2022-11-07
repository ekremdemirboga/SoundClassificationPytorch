from torch import nn
from torchsummary import summary


class CNNNetwork(nn.Module):
    
    def __init__(self):
        super().__init__()
        ## 4 conv block / flatten / linear / softmax
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1, ## no of input channels
                out_channels=16, ## no of output channels
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, ## no of input channels
                out_channels=32, ## no of output channels
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, ## no of input channels
                out_channels=64, ## no of output channels
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, ## no of input channels
                out_channels=128, ## no of output channels
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128*5*7, 10) # (size of input, no of classes in dataset)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions
        

#cnn = CNNNetwork()
#summary(cnn, (1,64,44)) #(no of chnls, no of melbands, time axis?)

