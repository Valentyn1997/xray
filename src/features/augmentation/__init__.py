import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from skimage import img_as_ubyte
from skimage import img_as_float
import cv2
import matplotlib.pyplot as plt
from src.data import DataGenerator, TrainValTestSplitter


class Augmentation:
    """Augment data"""

    def __init__(self, seq, random_state=42):
        """
        Initialization
        :param seq: augmentation sequence from imgaug
        """
        self.seq = seq
        if random_state is not None:
            ia.seed(random_state)

    def augment(self, image):
        """
        Transforms the image into the right format and
        augment image
        :param image: list of images (*dim) or array (n, ch, *dim)
        :return: augmented images (n, ch, *dim)
        """
        if isinstance(image, list):
            output_img = self._augment_list(image)
        if isinstance(image, np.ndarray):
            output_img = self._augment_array(image)

        return output_img


    def _augment_array(self, image):
        """
        Transforms array of images into the right format and
        augment image
        :param image: array (n, ch, *dim)
        :return: augmented images (n, ch, *dim)
        """
        # Takes the image array and transforms to uint8
        inp_int = img_as_ubyte(image)
        # Change the channel to the last position
        inp_rearrange = np.rollaxis(inp_int, 1, 4)
        # Augment image according to specification from input
        inp_aug = self.seq.augment_images(inp_rearrange)
        # Change the channel back to second position
        inp_aug_rearrange = np.rollaxis(inp_aug, 3, 1)
        inp_aug_array = img_as_float(inp_aug_rearrange)

        return inp_aug_array

    def _augment_list(self, image):
        """
        Transforms list of images into the right format and
        augment image
        :param image: list [image (*dim), ...]
        :return: augmented images (n, ch, *dim)
        """
        # Takes image list and transforms to uint8
        inp_int = [img_as_ubyte(i) for i in image]
        # Augment image according to specification from input
        inp_aug = self.seq.augment_images(inp_int)
        # Takes augmented images and transforms to float
        inp_aug_float = [img_as_float(i) for i in inp_aug]
        # Transform to array and add one channel
        array_img = np.asarray(inp_aug_float)
        img_aug_list = array_img[:, np.newaxis, :, :]

        return img_aug_list


#### Example and test
# # set augmentation to flip upside down
# seq = iaa.Sequential([
#     iaa.Flipud(1)
# ])
#
# aug = Augmentation(seq)
#
# # Test on array from train generator
#
# np.seterr(divide='ignore', invalid='ignore')
# batch_size = 32
# image_resolution = (512, 512)
# hist_equalisation = True
# splitter = TrainValTestSplitter()
# train_generator = DataGenerator(filenames=splitter.data_train.path, batch_size=batch_size, dim=image_resolution,
#                                 hist_equalisation=hist_equalisation)
#
#
# test_array = train_generator[0]
# test_array_aug = aug.augment(test_array)
#
#
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
# ax[0].imshow(test_array[0][0], cmap='gray', vmin=0, vmax=1)
# ax[1].imshow(test_array_aug[0][0], cmap='gray', vmin=0, vmax=1)
# plt.show()
#
# # Test on list of images
#
# img = cv2.imread('../../data/train/XR_HAND/patient00008/study1_positive/image1.png')
# img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# img2 = cv2.imread('../../data/train/XR_HAND/patient00008/study1_positive/image2.png')
# img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
#
# test_list = [img, img2]
# test_list_aug = aug.augment(test_list)
#
# fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
# ax[0].imshow(test_list[0], cmap='gray')
# ax[1].imshow(test_list_aug[0][0], cmap='gray', vmin=0, vmax=1)
# plt.show()

