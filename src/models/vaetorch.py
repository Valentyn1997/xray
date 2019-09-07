import os
from typing import List

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from src import TMP_IMAGES_DIR, TOP_K
from src.models.autoencoders import BaselineAutoencoder
from src.models.outlier_scoring import TopK, Mean
from src.utils import log_artifact, calculate_metrics


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size, 6, 6)


class MaskedL1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        Masked L1 implementation
        :param reduction: the same, as in nn.L1Loss
        """
        super(MaskedL1Loss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target, mask):
        """
        calculates masked loss
        :param input: input image as array
        :param target: reconstructed image as array
        :param mask: mask of image as array
        :return: masked loss
        """
        loss = F.smooth_l1_loss(input * mask, target * mask, reduction='none')
        if self.reduction == 'mean':
            loss = torch.sum(loss) / torch.sum(mask)
        return loss


class VAE(nn.Module):
    """ Variational Convolutional Autoencoder using torch library """

    def __init__(self, device, h_dim=18432, z_dim=512,
                 encoder_in_chanels: List[int] = (1, 16, 32, 64, 128, 256),
                 encoder_out_chanels: List[int] = (16, 32, 64, 128, 256, 512),
                 encoder_kernel_sizes: List[int] = (4, 4, 4, 4, 4, 4),
                 encoder_strides: List[int] = (2, 2, 2, 2, 2, 2),
                 decoder_in_chanels: List[int] = (512, 256, 128, 64, 32, 16),
                 decoder_out_chanels: List[int] = (256, 128, 64, 32, 16, 1),
                 decoder_kernel_sizes: List[int] = (4, 4, 4, 4, 4, 6),
                 decoder_strides: List[int] = (2, 2, 2, 2, 2, 2),
                 internal_activation=nn.ReLU,
                 batch_normalisation=True,
                 final_activation=nn.Sigmoid,
                 lr=1e-5,
                 masked_loss_on_val=False,
                 masked_loss_on_train=False,
                 *args, **kwargs):
        super(VAE, self).__init__()

        self.device = device
        self.hyper_parameters = locals()
        self.hyper_parameters.pop('self')

        # Encoder initialization
        self.encoder_layers = []
        for i in range(len(encoder_in_chanels)):
            self.encoder_layers.append(nn.Conv2d(encoder_in_chanels[i], encoder_out_chanels[i],
                                                 kernel_size=encoder_kernel_sizes[i],
                                                 stride=encoder_strides[i]))
            if batch_normalisation:
                self.encoder_layers.append(nn.BatchNorm2d(encoder_out_chanels[i]))
            self.encoder_layers.append(internal_activation())
        self.encoder_layers.append(Flatten())

        # Decoder initialization
        self.decoder_layers = []
        self.decoder_layers.append(UnFlatten())
        for i in range(len(decoder_in_chanels)):
            self.decoder_layers.append(nn.ConvTranspose2d(decoder_in_chanels[i], decoder_out_chanels[i],
                                                          kernel_size=decoder_kernel_sizes[i],
                                                          stride=decoder_strides[i]))
            if not i == len(decoder_in_chanels) - 1:
                # no batch norm and no internal activation after last convolution
                if batch_normalisation:
                    self.decoder_layers.append(nn.BatchNorm2d(decoder_out_chanels[i]))
                self.decoder_layers.append(internal_activation())
        self.decoder_layers.append(final_activation())

        self.encoder = nn.Sequential(*self.encoder_layers)
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.decoder = nn.Sequential(*self.decoder_layers)

        # Losses
        self.masked_loss_on_val = masked_loss_on_val
        self.masked_loss_on_train = masked_loss_on_train

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def reparameterize(self, mu, var):
        """
        Reparametrization of distribution parameters: mean and variance
        :param mu: mean
        :param var: variance
        """
        std = var.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, var = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, var)
        return z, mu, var

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

    def forward(self, x):
        h = self.encoder(x)
        z, mu, var = self.bottleneck(h)
        z = self.fc3(z)
        return self.decoder(z), mu, var

    @staticmethod
    def loss(recon_x, x, mu, var, reduction='mean', mask=None):
        """
        Kullback Leibler divergence + Binary Cross Entropy combined loss
        :param recon_x: reconstructed image
        :param x: original image
        :param mu: mean
        :param var: variance
        :param reduction: reduction type
        :return: loss
        """
        KLD = 0
        BCE = F.binary_cross_entropy(recon_x, x, size_average=False, reduction=reduction)
        if reduction == 'mean':
            KLD = -0.5 * torch.sum(1 + var - mu ** 2 - var.exp())
        elif reduction == 'none':
            KLD = -0.5 * (1 + var - mu ** 2 - var.exp())
        return BCE + KLD

    @staticmethod
    def lossMSE(recon_x, x, mu, var, reduction='mean'):
        """
        Kullback Leibler divergence + MSE combined loss
        :param recon_x: reconstructed image
        :param x: original image
        :param mu: mean
        :param var: variance
        :param reduction: reduction type
        :return: loss
        """
        KLD = 0
        MSE = F.mse_loss(recon_x, x, size_average=False, reduction=reduction)
        if reduction == 'mean':
            KLD = -0.5 * torch.sum(1 + var - mu ** 2 - var.exp())
        elif reduction == 'none':
            KLD = -0.5 * (1 + var - mu ** 2 - var.exp())
        return MSE + KLD

    @staticmethod
    def loss_L1(recon_x, x, mu, var, reduction='mean', mask=None):
        """
        Kullback Leibler divergence + Smooth L1 combined loss
        :param recon_x: reconstructed image
        :param x: original image
        :param mu: mean
        :param var: variance
        :param reduction: reduction type
        :return: loss
        """
        KLD = 0
        if mask is not None:
            L1 = MaskedL1Loss(reduction=reduction)(recon_x, x, mask)
        else:
            L1 = F.smooth_l1_loss(recon_x, x, reduction=reduction)
        if reduction == 'mean':
            KLD = -0.5 * torch.sum(1 + var - mu ** 2 - var.exp())
        elif reduction == 'none':
            KLD = -0.5 * (1 + var - mu ** 2 - var.exp())
        # print(L1.shape, KLD.shape)
        return L1, KLD

    @staticmethod
    def loss_pixel(recon_x, x):
        """
        Pixel-wise loss
        :param recon_x: reconstructed image
        :param x: original image
        :return: loss
        """
        return recon_x - x

    def evaluate(self, loader, log_to_mlflow=False):
        """
        Computes ROC-AUC, APS
        :param loader: data loader
        :param log_to_mlflow: boolean variable to enable logging
        :return: calculated metrics
        """

        self.eval()
        scores_l1 = []
        scores_l1_top_k = []
        scores_l1_kld = []
        scores_l1_kld_top_k = []
        true_labels = []

        with torch.no_grad():

            for batch_data in tqdm(loader, desc='Validation', total=len(loader)):
                inp = batch_data['image'].to(self.device)
                mask = batch_data['mask'].to(self.device) if self.masked_loss_on_val else None
                labels = batch_data['label']

                # forward pass
                output, mu, var = self(inp)
                L1, KLD = self.loss_L1(output, inp, mu, var, reduction='none', mask=mask)

                score_l1 = Mean.calculate(L1, masked_loss=self.masked_loss_on_val, mask=mask)
                score_l1_top_k = TopK.calculate(L1, TOP_K, reduce_to_mean=True)
                score_l1_kl = score_l1 + KLD.to('cpu').numpy().sum(axis=1)
                score_l1_kl_top_k = score_l1_top_k + KLD.to('cpu').numpy().sum(axis=1)

                scores_l1.extend(score_l1)
                scores_l1_top_k.extend(score_l1_top_k)
                scores_l1_kld.extend(score_l1_kl)
                scores_l1_kld_top_k.extend(score_l1_kl_top_k)
                true_labels.extend(labels.numpy())

            metrics_l1 = calculate_metrics(np.array(scores_l1_kld), np.array(true_labels), 'l1')
            metrics_l1_top_k = calculate_metrics(np.array(scores_l1_top_k), np.array(true_labels), 'l1_top_k')
            metrics_l1_kld = calculate_metrics(np.array(scores_l1_kld), np.array(true_labels), 'l1_kld')
            metrics_l1_kld_top_k = calculate_metrics(np.array(scores_l1_kld_top_k), np.array(true_labels),
                                                     'l1_kld_top_k')
            metrics = {**metrics_l1, **metrics_l1_top_k, **metrics_l1_kld, **metrics_l1_kld_top_k}

            if log_to_mlflow:
                for (metric, value) in metrics.items():
                    mlflow.log_metric(metric, value)

            return metrics

    def forward_and_save_one_image(self, inp_image, label, epoch, path=TMP_IMAGES_DIR, to_mlflow=False,
                                   is_remote=False):
        """
        Save random sample of original and reconstructed image
        :param inp_image: input original image
        :param label: label of image
        :param epoch: number of epoch
        :param device: device type: CPU or CUDA GPU
        :param path: path to save
        """
        self.eval()
        with torch.no_grad():
            inp = inp_image.to(self.device)
            output, _, _ = self(inp)
            output_img = output.to('cpu')

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            ax[0].imshow(inp_image.numpy()[0, 0, :, :], cmap='gray', vmin=0, vmax=1)
            ax[1].imshow(output_img.numpy()[0, 0, :, :], cmap='gray', vmin=0, vmax=1)
            path = f'{path}/epoch{epoch}_label{int(label)}.png'
            plt.savefig(path)
            plt.close(fig)

            if to_mlflow:
                log_artifact(path, 'images', is_remote=is_remote)
                os.remove(path)

    summary = BaselineAutoencoder.__dict__["summary"]

    save_to_mlflow = BaselineAutoencoder.__dict__["save_to_mlflow"]

    def train_on_batch(self, batch_data, *args, **kwargs):
        """
        Performs one step of gradient descent on batch_data
        :param batch_data: Data of batch
        :param args:
        :param kwargs:
        :return: Dict of losses
        """
        # Switching to train mode
        self.train()

        # Format input batch
        inp = Variable(batch_data['image']).to(self.device)
        mask = batch_data['mask'].to(self.device) if self.masked_loss_on_train else None

        # forward pass
        output, mu, logvar = self(inp)
        L1, KLD = VAE.loss_L1(output, inp, mu, logvar, mask=mask)
        loss = L1 + 0.001 * KLD

        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'l1': float(L1.data),
                'kld': float(KLD.data)}
