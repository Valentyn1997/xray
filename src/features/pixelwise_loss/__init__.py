import torch
from tqdm import tqdm


class PixelwiseLoss:

    def __init__(self, model, model_class, device, loss_function, masked_loss_on_val):
        """
        Initialize evluation of pixelwise loss

        :param model: trained pytorch model
        :param model_class: string, which model class: CAE, VAE
        :param device: string, which device usually cpu or cuda
        :param loss_function: function, which loss should be used example: nn.MSELoss
        :param masked_loss_on_val: bool, should masked loss should be used
        """

        self.model = model
        self.device = device
        self.loss_function = loss_function
        self.model_class = model_class
        self.masked_loss_on_val = masked_loss_on_val

    def get_loss(self, data):
        """
        Apply model on data and get pixelwise loss

        :param data: data to be evaluated, example from dataloader
        :return: dictionary with pixelwise loss, label, patient, filepath
        """

        # set model to evaluation, so no weights will be updated
        model = self.model
        model.eval()
        with torch.no_grad():
            # Initiate list to save results
            pixelwise_loss = []
            true_labels = []
            patient = []
            path = []

            # iterate over data with batch
            for batch_data in tqdm(data, desc='evaluation', total=len(data)):
                # load data from batch
                inp = batch_data['image'].to(self.device)
                mask = batch_data['mask'].to(self.device)

                if self.model_class == 'CAE':
                    # apply model on image
                    output = model(inp)
                    # calculate loss per pixel
                    loss = self.loss_function(output, inp, mask) if self.masked_loss_on_val else self.loss_function(
                        output, inp)
                    loss = loss.cpu().numpy()
                elif self.model_class == 'VAE':
                    # apply model on image
                    output, mu, var = self(inp)
                    # calculate loss per pixel
                    loss = self.loss_function(output, inp, mu, var, reduction='none')
                    loss = loss.numpy()

                # append values to list
                pixelwise_loss.extend(loss)
                true_labels.extend(batch_data['label'])
                patient.extend(batch_data['patient'])
                path.extend(batch_data['filename'])

            # create dictionary with results
            out = {'loss': pixelwise_loss, 'label': true_labels, 'patient': patient, 'path': path}
            return out

# Example:
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
# from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
# from tqdm import tqdm
# from typing import List
#
# import matplotlib.pyplot as plt
#
# from src import MODELS_DIR, MLFLOW_TRACKING_URI, DATA_PATH
# from src.data import TrainValTestSplitter, MURASubset
# from src.data.transforms import *
# from src.features.augmentation import Augmentation
# from src.models.autoencoders import BottleneckAutoencoder, BaselineAutoencoder
# from src.models.gans import DCGAN
# from src.models.vaetorch import VAE
# from src.models import BaselineAutoencoder
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
#         'data_source': 'XR_HAND_CROPPED',
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
# model = torch.load(path)
#
# # set loss function
#
# outer_loss = nn.MSELoss(reduction='none')
# model.eval().to(device)
#
# evaluation = PixelwiseLoss(model=model, model_class='VAE',
# device=device, loss_function=outer_loss, masked_loss_on_val=True)
# loss_dict = evaluation.get_loss(data = val_loader)
