import numpy as np
import imgaug.augmenters as iaa
from skimage import img_as_uint


class Augmentation:
    """Augment data"""

    def __init__(self, image, seq, random_state=42):
        """
        Initialization
        :param image: array or array of images(array)
        :param seq: augmentation sequence from imgaug
        """
        self.image = image
        self.seq = seq
        if random_state is not None:
            iaa.seed(random_state)

    def augment(self):
        # Takes the image array and transforms it to int
        inp_int = img_as_uint(self.img)
        # Change the channel to the last position
        inp_rearrange = np.rollaxis(inp_int, 1, 4)
        # Augment image according to specification from input
        inp_aug = self.seq.augment_images(inp_rearrange)
        # Change the channel back to second position
        inp_aug_rearrange = np.rollaxis(inp_aug, 3, 1)

        return inp_aug_rearrange
