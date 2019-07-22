import torch
from sklearn.metrics import roc_auc_score


class TopK:
    """calculate top k loss for an element of pixelwise losses"""

    def __init__(self, loss):
        """
        initialize top k
        :param loss: list of pixelwise losses
        """
        self.loss = loss

    def calculate_topk(self, loss, k):
        """
        calculate top k loss for pixelwise loss
        :param loss: element of pixelwise loss list
        :param k: int number of highest lost
        :return: top k loss
        """
        # convert to tensor
        loss = torch.from_numpy(loss)
        # stick all tensors in one long tensor
        flat_loss = loss.view(-1)
        # find the top k losses
        top_loss = torch.topk(flat_loss, k=k, dim=-1, largest=True, sorted=False)
        # calculate mean out of top k losses
        score = torch.mean(top_loss.values, dim=-1).item()
        return score

    def get_topk(self, k):
        """
        calculate list of topk
        :return: list of top k lost
        """
        loss_topk = [self.calculate_topk(loss=elem, k=k) for elem in self.loss]
        return loss_topk

    def get_range_topk_auc(self, start, end, step, label):
        """
        calculate optimal k for auc score
        :param start: int starting point for k
        :param end: int ending point for k
        :param step: int step size for k
        :param label: list with labels
        :return: dictionary with k and the correspondending auc score
        """
        range_steps = range(start, end, step)
        label_list = label
        roc_auc_list = []
        k_list = []

        for k in range_steps:
            # calculate top k loss for k
            loss_topk = self.get_topk(k=k)
            # calculate roc_auc
            roc_auc = roc_auc_score(label_list, loss_topk)
            # append to list
            roc_auc_list.append(roc_auc)
            k_list.append(k)

        out = {'K': k_list, 'AUC': roc_auc_list}
        return out

# # Example:
# import os
# import sys
#
# import imgaug.augmenters as iaa
# import mlflow.pytorch
# import numpy as np
# import torch
# from pprint import pprint
# from torch.utils.data import DataLoader
# from torchvision.transforms import Compose
# from tqdm import tqdm
# import pandas as pd
# import cv2
# import torch.nn as nn
# from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, confusion_matrix
# from tqdm import tqdm
# from typing import List
# from sklearn.utils.multiclass import unique_labels
#
#
# import matplotlib.pyplot as plt
#
# nb_dir = os.path.split(os.getcwd())[0]
# if nb_dir not in sys.path:
#     sys.path.append(nb_dir)
#
#
# from src import MODELS_DIR, MLFLOW_TRACKING_URI, DATA_PATH
# from src.data import TrainValTestSplitter, MURASubset
# from src.data.transforms import *
# from src.features.augmentation import Augmentation
# from src.models.autoencoders import BottleneckAutoencoder, BaselineAutoencoder
# from src.models.gans import DCGAN
# from src.models.vaetorch import VAE
# from src.models import BaselineAutoencoder
# from src.features.pixelwise_loss import PixelwiseLoss
# from src.models.autoencoders import MaskedMSELoss
#
# num_workers = 7
# log_to_mlflow = False
# device = "cuda"
#
# # Mlflow parameters
# run_params = {
#     'batch_size': 32,
#     'image_resolution': (512, 512),
#     'num_epochs': 1000,
#     'batch_normalisation': True,
#     'pipeline': {
#         'hist_equalisation': True,
#         'data_source': 'XR_HAND_PHOTOSHOP',
#     },
#     'masked_loss_on_val': True,
#     'masked_loss_on_train': True,
#     'soft_labels': True,
#     'glr': 0.001,
#     'dlr': 0.00005,
#     'z_dim': 1000,
#     'lr': 0.0001
# }
#
#
# # Preprocessing pipeline
#
# composed_transforms_val = Compose([GrayScale(),
#                                    HistEqualisation(active=run_params['pipeline']['hist_equalisation']),
#                                    Resize(run_params['image_resolution'], keep_aspect_ratio=True),
#                                    Augmentation(iaa.Sequential([iaa.PadToFixedSize(512, 512, position='center')])),
#                                    # Padding(max_shape=run_params['image_resolution']),
#                                    # max_shape - max size of image after augmentation
#                                    MinMaxNormalization(),
#                                    ToTensor()])
#
# # get data
#
# data_path = f'{DATA_PATH}/{run_params["pipeline"]["data_source"]}'
# print(data_path)
# splitter = TrainValTestSplitter(path_to_data=data_path)
#
# validation = MURASubset(filenames=splitter.data_val.path, true_labels=splitter.data_val.label,
#                         patients=splitter.data_val.patient, transform=composed_transforms_val)
#
# val_loader = DataLoader(validation, batch_size=run_params['batch_size'], shuffle=True, num_workers=num_workers)
#
# # get model (change path to path to a trained model
#
# model = torch.load('/mnt/data2/xray/viet/xray_1/models/baseline_autoencoder.pt')
#
# # set loss function
#
# outer_loss = MaskedMSELoss(reduction='none')
# model.eval().to(device)
#
# evaluation = PixelwiseLoss(model=model, model_class='CAE',device=device, loss_function=outer_loss, masked_loss_on_val=True)
# loss_dict = evaluation.get_loss(data = val_loader)
#
# # init topk
# topk = TopK(loss=loss_dict['loss'])
# # calculate topk for k = 50
# topk_baseline = topk.get_topk(k=50)
# # values of auc for different k
# label_list = [elem.item() for elem in loss_dict['label']]
# topk_baseline_range = topk.get_range_topk_auc(start=1, end=250000, step=1000, label=label_list)
