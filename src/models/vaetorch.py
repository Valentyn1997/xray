import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
import numpy as np
import mlflow
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List

from src import TMP_IMAGES_DIR


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size, 6, 6)


class VAE(nn.Module):
    """ Variational Convolutional Autoencoder using torch library """

    def __init__(self, device, h_dim=18432, z_dim=2,
                 encoder_in_chanels: List[int] = (1, 16, 32, 64, 128, 256),
                 encoder_out_chanels: List[int] = (16, 32, 64, 128, 256, 512),
                 encoder_kernel_sizes: List[int] = (4, 4, 4, 4, 4, 4),
                 encoder_strides: List[int] = (2, 2, 2, 2, 2, 2),
                 decoder_in_chanels: List[int] = (512, 256, 128, 64, 32, 16),
                 decoder_out_chanels: List[int] = (256, 128, 64, 32, 16, 1),
                 decoder_kernel_sizes: List[int] = (4, 4, 4, 4, 4, 6),
                 decoder_strides: List[int] = (2, 2, 2, 2, 2, 2),
                 internal_activation=nn.ReLU,
                 final_activation=nn.Sigmoid
                 ):
        super(VAE, self).__init__()
        self.device = device

        # Encoder initialization
        self.encoder_layers = []
        for i in range(len(encoder_in_chanels)):
            self.encoder_layers.append(nn.Conv2d(encoder_in_chanels[i], encoder_out_chanels[i],
                                                 kernel_size=encoder_kernel_sizes[i],
                                                 stride=encoder_strides[i]))
            self.encoder_layers.append(internal_activation())
        self.encoder_layers.append(Flatten())

        # Decoder initialization
        self.decoder_layers = []
        self.decoder_layers.append(UnFlatten())
        for i in range(len(decoder_in_chanels)):
            self.decoder_layers.append(nn.ConvTranspose2d(decoder_in_chanels[i], decoder_out_chanels[i],
                                                          kernel_size=decoder_kernel_sizes[i],
                                                          stride=decoder_strides[i]))
            # if not i == len(decoder_in_chanels):
            self.decoder_layers.append(internal_activation())
        self.decoder_layers.append(final_activation())

        self.encoder = nn.Sequential(*self.encoder_layers)
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        self.decoder = nn.Sequential(*self.decoder_layers)

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
    def loss(recon_x, x, mu, var, reduction='mean'):
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

    def evaluate(self, loader, type, device, log_to_mlflow=False, opt_threshold=None):
        """
        Computes ROC-AUC, F1-score, MSE and optimal threshold for model
        :param loader: data loader
        :param type: test or validation evaluation
        :param device: device type: CPU or CUDA GPU
        :param log_to_mlflow: boolean variable to enable logging
        :param opt_threshold: prespecified optimal threshold
        :return: calculated metrics
        """
        self.eval()
        with torch.no_grad():
            losses = []
            true_labels = []
            for batch_data in tqdm(loader, desc=type, total=len(loader)):
                inp = batch_data['image'].to(device)

                # forward pass
                output, mu, var = self(inp)
                loss = self.loss(output, inp, mu, var, reduction='none')
                losses.extend(loss.to('cpu').numpy().mean(axis=1))
                true_labels.extend(batch_data['label'].numpy())

            losses = np.array(losses)
            true_labels = np.array(true_labels)

            # ROC-AUC
            roc_auc = roc_auc_score(true_labels, losses)
            # MSE
            mse = losses.mean()
            # F1-score & optimal threshold
            if opt_threshold is None:  # validation
                precision, recall, thresholds = precision_recall_curve(y_true=true_labels, probas_pred=losses)
                f1_scores = (2 * precision * recall / (precision + recall))
                f1 = np.nanmax(f1_scores)
                opt_threshold = thresholds[np.argmax(f1_scores)]
            else:  # testing
                y_pred = (losses > opt_threshold).astype(int)
                f1 = f1_score(y_true=true_labels, y_pred=y_pred)

            print(f'ROC-AUC on {type}: {roc_auc}')
            print(f'MSE on {type}: {mse}')
            print(f'F1-score on {type}: {f1}. Optimal threshold on {type}: {opt_threshold}')

            metrics = {"roc-auc": roc_auc,
                       "mse": mse,
                       "f1-score": f1,
                       "optimal mse threshold": opt_threshold}

            if log_to_mlflow:
                for (metric, value) in metrics.items():
                    mlflow.log_metric(metric, value)
            return metrics

    def forward_and_save_one_image(self, inp_image, label, epoch, device, path=TMP_IMAGES_DIR):
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
            inp = inp_image.to(device)
            output, _, _ = self(inp)
            output_img = output.to('cpu')

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            ax[0].imshow(inp_image.numpy()[0, 0, :, :], cmap='gray', vmin=0, vmax=1)
            ax[1].imshow(output_img.numpy()[0, 0, :, :], cmap='gray', vmin=0, vmax=1)
            plt.savefig(f'{path}/epoch{epoch}_label{int(label)}.png')
            plt.close(fig)