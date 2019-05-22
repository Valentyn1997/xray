import numpy as np
import cv2
import glob
import pandas as pd
import re
from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt


class TrainValTestSplitter:

    def __init__(self, path_to_data='..\\..\\data\\train\\XR_HAND\\*\\*\\*',
                 show_labels_dist=False):
        """
        Train-validation-test splitter, stores all the filenames
        :param path_to_data: for glob.glob to find all the images path
        :param show_labels_dist: show plot of distributions of labels
        """
        self.data = pd.DataFrame()
        self.data['path'] = glob.glob(path_to_data)
        self.data['label'] = self.data['path']. \
            apply(lambda path: len(re.findall('positive', path)))
        self.data['patient'] = self.data['path']. \
            apply(lambda path: re.findall('[0-9]{5}', path)[0])
        if show_labels_dist:
            self.data['label'].hist()
            plt.title('Labels distribution')
            plt.show()
        self._split_data()

    def _split_stats(self, df):
        print(f'Size: {len(df)}')
        print(f'Percentage from original data: {len(df)/len(self.data)}')
        print(f'Percentage of negatives: {len(df[df.label == 0])/len(df)}')
        print(f'Number of patients: {len(df.patient.unique())}')

    def _split_data(self):
        """
        Creates data_train, data_val, data_test dataframes with filenames
        """
        # train | validate test split
        splitter = GroupShuffleSplit(n_splits=1,
                                     test_size=0.3, random_state=42)
        negative_data = self.data[self.data.label == 0]
        generator = splitter.split(negative_data.label,
                                   groups=negative_data['patient'])
        idx_train, idx_validate_test = next(generator)

        print('=================Train subset=================')
        self.data_train = negative_data.iloc[idx_train, :]. \
            reset_index(drop=True)
        self._split_stats(self.data_train)

        # validate | test split
        data_val_test = pd.concat(
            [self.data[self.data.label == 1],
             self.data.iloc[negative_data.iloc[idx_validate_test, :].index]])
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.50,
                                     random_state=42)
        generator = splitter.split(data_val_test.label,
                                   groups=data_val_test['patient'])
        idx_val, idx_test = next(generator)

        print('=============Validation subset===============')
        self.data_val = data_val_test.iloc[idx_val, :]
        self.data_val = self.data_val.sample(len(self.data_val)) \
            .reset_index(drop=True)
        self._split_stats(self.data_val)

        print('=================Test subset=================')
        self.data_test = data_val_test.iloc[idx_test, :]
        self.data_test = self.data_test.sample(
            len(self.data_test)).reset_index(drop=True)
        self._split_stats(self.data_test)


class DataGenerator:
    """Generates data"""

    def __init__(self, filenames, batch_size=16, dim=(512, 512), n_channels=1,
                 shuffle=True, true_labels=None, random_state=42):
        """Initialization
        :param filenames: list of filenames, e.g. from TrainValTestSplitter
        :param batch_size: size of batch
        :param dim: size of images in batch, all the images will be resized to
        fit
        :param n_channels: 1 - for black/white
        :param shuffle: shuffle all the data after epoch end
        :param true_labels: list of true labels (for validation and split)
        """
        self.dim = dim
        self.batch_size = batch_size
        self.filenames = filenames
        self.n_channels = n_channels
        self.shuffle = shuffle
        if random_state is not None:
            np.random.seed(random_state)
        self.on_epoch_end()
        self.input_shape = (self.batch_size, self.n_channels, *self.dim)
        self.true_labels = np.array(true_labels)

    def __len__(self) -> int:
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.filenames) / self.batch_size))

    def __getitem__(self, index) -> np.array:
        """Generate one batch of data"""
        if index == -1:
            index = len(self) - 1

        # Generate indexes of the batch
        indexes = self.indexes[
                  index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_filenames_temp = [self.filenames[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_filenames_temp)

        return X

    def get_true_labels(self) -> np.array:
        return self.true_labels[self.indexes[0:len(self) * self.batch_size]]

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.filenames))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_filenames_temp) -> np.array:
        """Generates data containing batch_size samples"""
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty(self.input_shape)

        # Generate data
        for i, filename in enumerate(list_filenames_temp):
            # Store sample
            img = cv2.imread(filename)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            tb, uneven_tb = int((512 - img.shape[0])/2), (512 - img.shape[0])%2
            lr, uneven_lr = int((512 - img.shape[1])/2), (512 - img.shape[1])%2
            img = cv2.copyMakeBorder(img,
                                     tb, tb+uneven_tb,
                                     lr, lr+uneven_lr,
                                     cv2.BORDER_CONSTANT, value=0)
            img = cv2.resize(img, self.dim)
            img = img * 1 / 255
            X[i, 0, :, :] = img

        return X
