import numpy as np

from PIL import Image

from torch.utils.data import Dataset


class MNISTDataset(Dataset):

    def __init__(self, npz_file, transform=None, num_classes=10):
        data = np.load(npz_file)
        self.x = data['x']
        self.len = self.x.shape[0]
        self.y = np.zeros((self.len, 10), dtype=np.float)
        self.y[np.arange(self.len), data['y']] = 1
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        y = self.y[idx]
        x = Image.fromarray(self.x[idx])

        if self.transform is not None:
            x = self.transform(x)

        sample = {'x': x, 'y': y}
        return sample
