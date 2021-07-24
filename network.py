import torch
import torch.nn as nn
from torchvision.models import alexnet


class ParticialAlexNet(nn.Module):
    def __init__(self, freeze_k: int):
        super().__init__()

        # Load the pretrained alexnet
        #  and freeze the Convnet layers
        alex_pretrained = alexnet(pretrained=True)
        self.freeze_alex_layers(alex_pretrained, freeze_k)
        self.net = nn.Sequential(
            alex_pretrained.features,
            alex_pretrained.avgpool
        )

    @staticmethod
    def freeze_alex_layers(model: nn.Module, k_layers: int):
        """Freeze the NN for the specific layers in AlexNet"""
        k_layers *= 2
        for name, param in model.named_parameters():
            param.requires_grad = False
            k_layers -= 1

            if k_layers == 0:
                break

    def forward(self, x):
        return self.net(x)


class SSSN(nn.Module):
    def __init__(self, num_classes: int, freeze_k: int, **kwargs):
        super().__init__()
        # kwargs are used as the buffer that can make the code
        #  more generalizable in train.py

        # Create the final networks
        self.net = nn.Sequential(
            ParticialAlexNet(freeze_k=freeze_k),
            nn.Flatten(start_dim=1),
            nn.Dropout(0.3),
            nn.Linear(in_features=9216,
                      out_features=num_classes)
        )

    def forward(self, x):
        """Output the result"""
        return self.net(x)


class DSSN(nn.Module):
    def __init__(self, num_classes: 5, mode: str,
                 freeze_k: int):
        super().__init__()

        # Mode for combine the two layers
        # Include ["add", "mul", "cat"]
        assert mode in ["add", "mul", "cat"], "Mode unsupported. Choose from ['add', 'mul', 'cat']"
        self.mode = mode

        # Create two different channel networks
        self.model_mag = ParticialAlexNet(freeze_k=freeze_k)
        self.model_strain = ParticialAlexNet(freeze_k=freeze_k)

        # Network after combination
        if self.mode == "cat":
            self.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Dropout(0.3),
                nn.Linear(in_features=9216 * 2, out_features=num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(start_dim=1),
                nn.Dropout(0.3),
                nn.Linear(in_features=9216, out_features=num_classes)
            )

    def forward(self, mag_input, strain_input):
        # Output of two differnt kinds of channels
        out_mag = self.model_mag(mag_input)
        out_strain = self.model_strain(strain_input)

        # Combine two network
        if self.mode == "cat":
            out = torch.cat([out_mag, out_strain], dim=1)
        elif self.mode == "mul":
            out = out_mag * out_strain
        else:
            out = out_mag + out_strain

        out = self.classifier(out)

        return out

