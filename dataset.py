import numpy as np

from PIL import Image

from torch.utils.data import Dataset


class MNISTDataset(Dataset):

    def __init__(self, npz_file, transform=None, num_classes=10, train_data=True):
        data = np.load(npz_file)
        self.x = data['x']
        self.len = self.x.shape[0]
        self.train_data = train_data
        if self.train_data:
            self.y = np.zeros((self.len, 10), dtype=np.float)
            self.y[np.arange(self.len), data['y']] = 1
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = Image.fromarray(self.x[idx])

        if self.transform is not None:
            x = self.transform(x)

        if self.train_data:
            y = self.y[idx]
            sample = {'x': x, 'y': y}
        else:
            sample = {'x': x}

        return sample
