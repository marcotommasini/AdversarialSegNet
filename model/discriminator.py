import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self, num_classes, ndf=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        # x = self.up_sample(x)
        # x = self.sigmoid(x)
        return x


class DiscriminatorSEP(nn.Module):
    def __init__(self, num_classes, ndf=64):
        super(DiscriminatorSEP, self).__init__()

        self.conv1_1 = nn.Conv2d(num_classes, num_classes, kernel_size=4, groups=num_classes, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(num_classes, ndf, kernel_size=1)

        self.conv2_1 = nn.Conv2d(ndf, ndf, kernel_size=4, groups=ndf, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(ndf, ndf * 2, kernel_size=1)

        self.conv3_1 = nn.Conv2d(ndf * 2, ndf * 2, kernel_size=4, groups=ndf * 2, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=1)

        self.conv4_1 = nn.Conv2d(ndf * 4, ndf * 4, kernel_size=4, groups=ndf * 4, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=1)

        self.classifier_1 = nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, groups=ndf * 8, stride=2, padding=1)
        self.classifier_2 = nn.Conv2d(ndf * 8, 1, kernel_size=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.leaky_relu(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.leaky_relu(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.leaky_relu(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.leaky_relu(x)

        x = self.classifier_1(x)
        x = self.classifier_2(x)
        # x = self.up_sample(x)
        # x = self.sigmoid(x)

        return x
