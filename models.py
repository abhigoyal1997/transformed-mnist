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


class AGCNN(nn.Module):

    def __init__(self, input_shape=(1, 128, 128), dropout=0.25, num_classes=10, backbone='cnn', threshold=0.2):
        super(CNN, self).__init__()
        self.threshold = threshold

        if backbone == 'cnn':
            self.global_branch = CNN(input_shape=input_shape, dropout=dropout, num_classes=num_classes)
            self.local_branch = CNN(input_shape=input_shape, dropout=dropout, num_classes=num_classes)

        self.num_flat_features = self.global_branch.num_flat_features + self.local_branch.num_flat_features
        self.fc1 = nn.Linear(self.num_flat_features, self.global_branch.fc1.out_features)
        self.fc1 = nn.Linear(self.global_branch.fc2.in_features, self.global_branch.fc2.out_features)
        self.fc3 = nn.Linear(self.global_branch.fc3.in_features, num_classes)

        self.fc_dropout = nn.Dropout(2*dropout)

    def forward(self, x):
        x1 = self.global_branch.down_sample(x)


def get_attention_mask(x, thre):

    x = torch.abs(x)
    heatmap, _ = torch.max(x, dim=1, keepdim=True)
    mask = F.upsample(heatmap, scale_factor=32, mode='bilinear')
    mask_max, _ = mask.max(dim=2, keepdim=True)
    mask_max, _ = mask_max.max(dim=3, keepdim=True)
    mask_max = mask_max.expand(mask.size())
    mask = mask / mask_max
    mask = torch.ge(mask, thre)

    return (mask.data).cpu().numpy(), heatmap


def get_attention_xy(mask):
    mask_shape = mask.shape
    mask = np.uint8(mask)
    attention_xy = np.zeros((mask_shape[0], 4))
    for i in range(mask_shape[0]):
        this_mask = mask[i, :, :, :]
        this_mask = this_mask.reshape(mask_shape[2], mask_shape[3])
        contours = cv2.findContours(this_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        attention_xy[i, :] = find_attention_xy(contours)
    return attention_xy


def find_attention_xy(contours):

    size = 0
    tx, ty, bx, by = 0, 0, 0, 0
    for cnt in contours[1]:
        cnt = np.squeeze(cnt, axis=1)
        x1, y1 = np.min(cnt, axis=0)
        x2, y2 = np.max(cnt, axis=0)
        if (y2 - y1) * (x2 - x1) > size:
            tx, ty = x1, y1
            bx, by = x2, y2
            size = (y2 - y1) * (x2 - x1)
    return tx, ty, bx, by


def get_attention_img(img, attention_xy):
    samples = attention_xy.shape[0]
    for i in range(samples):
        tx, ty, bx, by = attention_xy[i, :]
        if tx == 0 and ty == 0 and bx == 0 and by ==0:
            this_attention_img = img[i, :, :, :]
        else:
            this_attention_img = img[i, :, int(ty):int(by), int(tx):int(bx)]

        this_attention_img = torch.unsqueeze(this_attention_img, 0)
        this_attention_img = F.upsample(this_attention_img, size=(img.size()[2], img.size()[3]), mode='bilinear')

        if i == 0:
            attention_img = this_attention_img
        else:
            attention_img = torch.cat((attention_img, this_attention_img), dim=0)
    return attention_img