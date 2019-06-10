import numpy as np
import cv2
import glob
import pandas as pd
import re
from sklearn.model_selection import GroupShuffleSplit
from src import XR_HAND_PATH
import matplotlib.pyplot as plt

from torch.utils.data import Dataset


class TrainValTestSplitter:

    def __init__(self, path_to_data=XR_HAND_PATH, show_labels_dist=False):
        """
        Train-validation-test splitter, stores all the filenames
        :param path_to_data: for glob.glob to find all the images path
        :param show_labels_dist: show plot of distributions of labels
        """
        path_to_data = f'{path_to_data}/*/*/*'
        self.data = pd.DataFrame()
        self.data['path'] = glob.glob(path_to_data)
        self.data['label'] = self.data['path'].apply(lambda path: len(re.findall('positive', path)))
        self.data['patient'] = self.data['path'].apply(lambda path: re.findall('[0-9]{5}', path)[0])
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
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        negative_data = self.data[self.data.label == 0]
        generator = splitter.split(negative_data.label, groups=negative_data['patient'])
        idx_train, idx_validate_test = next(generator)

        print('=================Train subset=================')
        self.data_train = negative_data.iloc[idx_train, :].reset_index(drop=True)
        self._split_stats(self.data_train)

        # validate | test split
        data_val_test = pd.concat([self.data[self.data.label == 1], self.data.iloc[negative_data.iloc[idx_validate_test, :].index]])
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
        generator = splitter.split(data_val_test.label, groups=data_val_test['patient'])
        idx_val, idx_test = next(generator)

        print('=============Validation subset===============')
        self.data_val = data_val_test.iloc[idx_val, :]
        self.data_val = self.data_val.sample(len(self.data_val)).reset_index(drop=True)
        self._split_stats(self.data_val)

        print('=================Test subset=================')
        self.data_test = data_val_test.iloc[idx_test, :]
        self.data_test = self.data_test.sample(len(self.data_test)).reset_index(drop=True)
        self._split_stats(self.data_test)


class MURASubset(Dataset):

    def __init__(self, filenames, transform=None, n_channels=1, true_labels=None, patients=None):
        """Initialization
        :param filenames: list of filenames, e.g. from TrainValTestSplitter
        :param true_labels: list of true labels (for validation and split)
        """
        self.transform = transform
        self.filenames = list(filenames)
        self.n_channels = n_channels
        self.true_labels = true_labels
        self.patients = patients

    def __len__(self) -> int:
        """Denotes the number of batches per epoch"""
        return len(self.filenames)

    def __getitem__(self, index) -> np.array:
        """Reads sample"""
        image = cv2.imread(self.filenames[index])
        label = self.true_labels[index] if self.true_labels is not None else None
        patient = self.patients[index] if self.true_labels is not None else None

        sample = {'image': image, 'label': label, 'patient': patient}

        if self.transform:
            sample = self.transform(sample)

        return sample

