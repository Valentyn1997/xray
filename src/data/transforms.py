import cv2
import numpy as np
import torch


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

    def __init__(self, out_dim, keep_aspect_ratio=False):
        """
        Initialization
        :param out_dim: size of images in batch, all the images will be resized to fit
        """
        self.out_dim = out_dim
        self.keep_aspect_ratio = keep_aspect_ratio

    def __call__(self, sample):
        out_dim = self.out_dim
        if self.keep_aspect_ratio:
            if sample['image'].shape[0] >= sample['image'].shape[1]:
                scale_factor = sample['image'].shape[1] / sample['image'].shape[0]
                out_dim = (int(self.out_dim[0] * scale_factor), int(self.out_dim[1]))
            else:
                scale_factor = sample['image'].shape[0] / sample['image'].shape[1]
                out_dim = (int(self.out_dim[0]), int(self.out_dim[1] * scale_factor))
        sample['image'] = cv2.resize(sample['image'], out_dim)
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

        # create pixel-wise masking
        sample['mask'] = (sample['image'] != 0.0).float()

        sample['label'] = torch.tensor(sample['label']).int() if sample['label'] is not None else torch.Tensor()
        sample['patient'] = torch.tensor(int(sample['patient'])).int() if sample['patient'] is not None else torch.Tensor()
        return sample


class MedianFilter(object):
    """Creates median filter"""

    def __call__(self, sample):

        sample['image'] = cv2.medianBlur(sample['image'], 19)
        return sample


class OtsuFilter(object):
    """
    Apply Otsu filter and cut rest of the image
    First add histogram equalization to the image, smooth the noise with median filter, run otsu filter and
    apply mask on original image
    """

    def __init__(self, active: bool):
        """
        Initialization
        :param active: activate otsu filter when true
        """
        self.active = active

    def __call__(self, sample):
        if self.active:
            # getimage
            img = sample['image']
            # add histogram equalization for better performance
            img_equ = cv2.equalizeHist(img)
            # add median filter for better performance
            img_median = cv2.medianBlur(img_equ, 19)
            # get threshold with otsu filter
            th = cv2.threshold(img_median, 0, 255, (cv2.THRESH_BINARY + cv2.THRESH_OTSU))[1]
            th[th > 0] = 1
            # apply filter on original image
            img = img * th
            sample['image'] = img
        return sample


class AdaptiveHistogramEqualization(object):
    """Apply adaptive histogram equalization"""

    def __init__(self, active: bool):
        """
        Initialize adaptive histogram equalization
        :param active: activate adaptive histogram equlization when true
        """
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.active = active

    def __call__(self, sample):
        if self.active:
            #apply adaptive histogram equalization
            sample['image'] = self.clahe.apply(sample['image'])
        return sample
