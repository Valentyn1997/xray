import torch
import cv2
import numpy as np


class GrayScale(object):
    """GrayScales image"""

    def __call__(self, sample):
        sample['image'] = cv2.cvtColor(sample['image'], cv2.COLOR_RGB2GRAY)
        return sample


class Padding(object):
    """Pad image with background color (for the same shape in batch)"""

    def __init__(self, centered=True, max_shape=(600, 600), color=0):
        self.centered = centered
        self.max_shape = max_shape
        self.color = color

    def __call__(self, sample):
        if self.centered:
            # Centering & padding with black color (for the same dimension)
            tb, uneven_tb = int((self.max_shape[0] - sample['image'].shape[0]) / 2), (self.max_shape[0] - sample['image'].shape[0]) % 2
            lr, uneven_lr = int((self.max_shape[1] - sample['image'].shape[1]) / 2), (self.max_shape[1] - sample['image'].shape[1]) % 2
        else:
            tb, lr = 0, 0
            uneven_tb, uneven_lr = self.max_shape[0] - sample['image'].shape[0], self.max_shape[1] - sample['image'].shape[1]

        try:
            sample['image'] = cv2.copyMakeBorder(sample['image'], tb, tb + uneven_tb, lr, lr + uneven_lr,
                                                 cv2.BORDER_CONSTANT, value=self.color)
        except cv2.error:
            print(f'Too big image: {sample["image"].shape}')

        return sample


class Resize(object):
    """Resizes image"""

    def __init__(self, out_dim):
        """
        Initialization
        :param out_dim: size of images in batch, all the images will be resized to fit
        """
        self.out_dim = out_dim

    def __call__(self, sample):
        sample['image'] = cv2.resize(sample['image'], self.out_dim)
        return sample


class HistEqualisation(object):
    """Equalizes histogram of image"""

    def __init__(self, active: bool):
        """
        Initialization
        :param active: no equalisation, when False
        """
        self.active = active

    def __call__(self, sample):
        if self.active:
            sample['image'] = cv2.equalizeHist(sample['image'])
        return sample


class MinMaxNormalization(object):
    """Normalizes image pixels to [0, 1] interval"""

    def __call__(self, sample):
        sample['image'] = (sample['image'] - np.min(sample['image'])) / (np.max(sample['image']) - np.min(sample['image']))
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # add color axis
        image = np.empty((1, *sample['image'].shape))
        image[0, :, :] = sample['image']
        sample['image'] = torch.from_numpy(image).float()

        sample['label'] = torch.tensor(sample['label']).int() if sample['label'] is not None else torch.Tensor()
        sample['patient'] = torch.tensor(int(sample['patient'])).int() if sample['patient'] is not None else torch.Tensor()
        return sample
