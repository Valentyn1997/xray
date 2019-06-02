import numpy as np
import imgaug as ia
from skimage import img_as_ubyte
from skimage import img_as_float


class Augmentation:
    """Augment data"""

    def __init__(self, image, seq, random_state=42):
        """
        Initialization
        :param image: (ch, *dim) or (n, ch, *dim)
        :param seq: augmentation sequence from imgaug
        """
        self.image = image
        self.seq = seq
        if random_state is not None:
            ia.seed(random_state)

    def augment(self):
        """
        Transforms the image into the right format and
        augment image
        :return: augmented images (n, ch, *dim)
        """
        # Takes the image array and transforms to uint8
        inp_int = img_as_ubyte(self.image)
        # Change the channel to the last position
        inp_rearrange = np.rollaxis(inp_int, 1, 4)
        # Augment image according to specification from input
        inp_aug = self.seq.augment_images(inp_rearrange)
        # Change the channel back to second position
        inp_aug_rearrange = np.rollaxis(inp_aug, 3, 1)
        inp_aug_float = img_as_float(inp_aug_rearrange)

        return inp_aug_float
