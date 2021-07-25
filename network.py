import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import alexnet


class PartialAlexNet(nn.Module):
    def __init__(self,
                 pretrained_path: str = "alex_pretrained/partial.pt"):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3,
                      out_channels=96, 
                      kernel_size=(11, 11), 
                      stride=(4, 4), 
                      padding=(4, 4)),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5,
                                 alpha=0.0001, 
                                 beta=0.75, 
                                 k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96,
                      out_channels=256,
                      kernel_size=5,
                      padding=2,
                      groups=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5,
                                 alpha=0.0001,
                                 beta=0.75, 
                                 k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.net.load_state_dict(torch.load(pretrained_path))
    
    def forward(self, x):
        # Shape of x: (batch_size, 256, 13, 13)
        x = self.net(x)

        # Shape of x: (batch_size, 256, 15, 15)
        x = F.pad(x, (1,) * 4)
        
        return x


class SSSN(nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        # kwargs are used as the buffer that can make the code
        #  more generalizable in train.py

        # Create the final networks
        self.net = nn.Sequential(
            PartialAlexNet(),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.5),
            nn.Linear(in_features=57600,
                      out_features=num_classes)
        )

    def forward(self, x):
        """Output the result"""
        return self.net(x)


class DSSN(nn.Module):
    def __init__(self,
                 num_classes: int,
                 mode: str,):
        super().__init__()

        # Mode for combine the two layers
        # Include ["add", "mul", "cat"]
        assert mode in ["add", "mul", "cat"], "Mode unsupported. Choose from ['add', 'mul', 'cat']."
        self.mode = mode

        # Create two different channel networks
        self.model_channel_one = PartialAlexNet()
        self.model_channel_two = PartialAlexNet()

        # Network after combination
        if self.mode == "cat":
            self.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Dropout(0.3),
                nn.Linear(in_features=57600 * 2, out_features=num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Dropout(0.3),
                nn.Linear(in_features=57600, out_features=num_classes)
            )

    def forward(self, input_one, input_two):
        # Output of two differnt kinds of channels
        out_one = self.model_channel_one(input_one)
        out_two = self.model_channel_two(input_two)

        # Combine two network
        if self.mode == "cat":
            out = torch.cat([out_one, out_two], dim=1)
        elif self.mode == "mul":
            out = out_one * out_two
        else:
            out = out_one + out_two

        out = self.classifier(out)

        return out
