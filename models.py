import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class CNN(nn.Module):

    def __init__(self, input_shape=(1, 128, 128), dropout=0.25, num_classes=10):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 12, 5)
        self.conv2 = nn.Conv2d(12, 64, 5)

        self.max_pool = nn.MaxPool2d((5, 5))

        self.conv_dropout = nn.Dropout2d(dropout)

        input = Variable(torch.rand(1, *input_shape))
        output_feat = self._forward_features(input, False)
        self.num_flat_features = output_feat.data.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.num_flat_features, 480)
        self.fc2 = nn.Linear(480, 150)
        self.fc3 = nn.Linear(150, num_classes)

        self.fc_dropout = nn.Dropout(2*dropout)

    def down_sample(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

    def _forward_features(self, x, train):
        x = self.down_sample(x)
        x = self.max_pool(x)
        if train:
            x = self.conv_dropout(x)
        return x

    def forward(self, x):
        x = self._forward_features(x, self.train)

        x = x.view(-1, self.num_flat_features)

        x = F.relu(self.fc1(x))
        # if self.train:
        #     x = self.fc_dropout(x)
        x = F.relu(self.fc2(x))
        if self.train:
            x = self.fc_dropout(x)
        x = self.fc3(x)

        return x

    def visualise_features(self, x, max_only=False):
        f = self._forward_features(x, False)
        if f.device.type == 'cuda':
            f = f.cpu()
        if max_only:
            f = np.max(f[0].numpy(), axis=0)
            f = cv.resize(f, x[0][0].shape)
            f = f/(np.max(f)-np.min(f))
        else:
            f = [cv.resize(i.numpy(), x[0][0].shape) for i in f[0]]
            f = np.asarray([i/(np.max(i)-np.min(i)) for i in f])

        return f
