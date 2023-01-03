# region packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

# endregion


class MyConvAutoEncoder1(torch.nn.Module):
    def __init__(self, img_chan):
        super().__init__()
        n_chan1 = 32
        n_chan2 = 8

        self.activation = nn.ReLU()

        # Encoder ----------
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels=img_chan, out_channels=n_chan1, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_chan1),
            self.activation,
            nn.MaxPool2d(2)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan1, out_channels=n_chan2, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_chan2),
            self.activation,
            nn.MaxPool2d(2)
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan2, out_channels=n_chan2, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_chan2),
            self.activation,
            # nn.MaxPool2d(2)
        )

        # Decoder ----------
        self.decoder1 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan2, out_channels=n_chan2, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_chan2),
            self.activation,
            nn.Upsample(scale_factor=2)
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan2, out_channels=n_chan1, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(n_chan1),
            self.activation,
            nn.Upsample(scale_factor=2)
        )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(in_channels=n_chan1, out_channels=img_chan, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(img_chan),
            self.activation,
            # nn.Upsample(scale_factor=2)
        )

    def forward(self, x):
        y = self.encoder1(x)
        y = self.encoder2(y)
        y = self.encoder3(y)
        y = self.decoder1(y)
        y = self.decoder2(y)
        y = self.decoder3(y)
        return y


class MyDenseAutoEncoder1(torch.nn.Module):
    def __init__(self, nfeatures1):
        super().__init__()
        nfeatures2 = 128
        self.encoder1 = nn.Sequential(nn.Linear(nfeatures1, nfeatures2),
                                      nn.ReLU())
        self.encoder2 = nn.Sequential(nn.Linear(nfeatures2, nfeatures2),
                                      nn.ReLU())
        self.decoder1 = nn.Sequential(nn.Linear(nfeatures2, nfeatures2),
                                      nn.ReLU())
        self.decoder2 = nn.Sequential(nn.Linear(nfeatures2, nfeatures1),
                                      nn.ReLU())

    def forward(self, x):
        y = self.encoder1(x)
        y = self.encoder2(y)
        y = self.decoder1(y)
        y = self.decoder2(y)
        return y

