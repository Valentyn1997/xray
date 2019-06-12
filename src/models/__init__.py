import torch.nn as nn
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
import mlflow
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from src import TMP_IMAGES_DIR
from typing import List


class BaselineAutoencoder(nn.Module):
    def __init__(self,
                 encoder_in_chanels: List[int] = (1, 16, 32, 32, 64, 64, 128, 128, 256, 256),
                 encoder_out_chanels: List[int] = (16, 32, 32, 64, 64, 128, 128, 256, 256, 512),
                 encoder_kernel_sizes: List[int] = (3, 4, 3, 4, 3, 4, 3, 4, 3, 4),
                 encoder_strides: List[int] = (1, 2, 1, 2, 1, 2, 1, 2, 1, 2),
                 decoder_in_chanels: List[int] = (512, 256, 128, 64, 32, 16),
                 decoder_out_chanels: List[int] = (256, 128, 64, 32, 16, 1),
                 decoder_kernel_sizes: List[int] = (4, 4, 4, 4, 4, 3),
                 decoder_strides: List[int] = (2, 2, 2, 2, 2, 1),
                 use_batchnorm: bool = True,
                 internal_activation=nn.ReLU,
                 final_activation=nn.Tanh):

        super(BaselineAutoencoder, self).__init__()

        self.encoder_layers = []
        for i in range(len(encoder_in_chanels)):
            self.encoder_layers.append(nn.Conv2d(encoder_in_chanels[i], encoder_out_chanels[i],
                                                 kernel_size=encoder_kernel_sizes[i],
                                                 stride=encoder_strides[i], padding=1, bias=not use_batchnorm))
            if use_batchnorm:
                self.encoder_layers.append(nn.BatchNorm2d(encoder_out_chanels[i]))
            self.encoder_layers.append(internal_activation())

        self.decoder_layers = []
        for i in range(len(decoder_in_chanels)):
            self.decoder_layers.append(nn.ConvTranspose2d(decoder_in_chanels[i], decoder_out_chanels[i],
                                                          kernel_size=decoder_kernel_sizes[i],
                                                          stride=decoder_strides[i], padding=1, bias=not use_batchnorm))
            if use_batchnorm and i < len(decoder_in_chanels) - 1:  # no batch norm after last convolution
                self.decoder_layers.append(nn.BatchNorm2d(decoder_out_chanels[i]))
            self.decoder_layers.append(internal_activation())
        self.decoder_layers.append(final_activation())

        self.encoder = nn.Sequential(*self.encoder_layers)
        self.decoder = nn.Sequential(*self.decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def forward_and_save_one_image(self, inp_image, label, epoch, device, path=TMP_IMAGES_DIR):
        self.eval()
        with torch.no_grad():
            inp = inp_image.to(device)
            output = self(inp)
            output_img = output.to('cpu')

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            ax[0].imshow(inp_image.numpy()[0, 0, :, :], cmap='gray', vmin=0, vmax=1)
            ax[1].imshow(output_img.numpy()[0, 0, :, :], cmap='gray', vmin=0, vmax=1)
            plt.savefig(f'{path}/epoch{epoch}_label{int(label)}.png')
            plt.close(fig)

    def evaluate(self, loader, type, loss, device, log_to_mlflow=False, opt_threshold=None):

        self.eval()
        with torch.no_grad():
            losses = []
            true_labels = []
            for batch_data in tqdm(loader, desc=type, total=len(loader)):
                inp = batch_data['image'].to(device)

                # forward pass
                output = self(inp)
                losses.extend(loss(output, inp).to('cpu').numpy().mean(axis=(1, 2, 3)))
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


class BottleneckAutoencoder(BaselineAutoencoder):
    def __init__(self,
                 encoder_in_chanels: List[int] = (1, 16, 32, 64, 128, 256),
                 encoder_out_chanels: List[int] = (16, 32, 64, 128, 256, 256),
                 encoder_kernel_sizes: List[int] = (3, 4, 4, 4, 4, 1),
                 encoder_strides: List[int] = (1, 2, 2, 2, 2, 1),
                 decoder_in_chanels: List[int] = (256, 256, 128, 64, 32, 16),
                 decoder_out_chanels: List[int] = (256, 128, 64, 32, 16, 1),
                 decoder_kernel_sizes: List[int] = (1, 4, 4, 4, 4, 3),
                 decoder_strides: List[int] = (1, 2, 2, 2, 2, 1),
                 use_batchnorm: bool = True,
                 internal_activation=nn.ReLU,
                 final_activation=nn.Sigmoid):

        super(BottleneckAutoencoder, self).__init__()

        self.encoder_layers = []
        for i in range(len(encoder_in_chanels)):
            self.encoder_layers.append(nn.Conv2d(encoder_in_chanels[i], encoder_out_chanels[i],
                                                 kernel_size=encoder_kernel_sizes[i],
                                                 stride=encoder_strides[i], padding=1, bias=not use_batchnorm))
            if i < len(encoder_in_chanels) - 1:
                self.encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
            if use_batchnorm:
                self.encoder_layers.append(nn.BatchNorm2d(encoder_out_chanels[i]))
            self.encoder_layers.append(internal_activation())

        self.decoder_layers = []
        for i in range(len(decoder_in_chanels)):
            self.decoder_layers.append(nn.ConvTranspose2d(decoder_in_chanels[i], decoder_out_chanels[i],
                                                          kernel_size=decoder_kernel_sizes[i], stride=decoder_strides[i],
                                                          padding=1, bias=not use_batchnorm))
            if i < len(decoder_in_chanels) - 1:
                self.decoder_layers.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
            if use_batchnorm and i < len(decoder_in_chanels) - 1:  # no batch norm after last convolution
                self.decoder_layers.append(nn.BatchNorm2d(decoder_out_chanels[i]))
            if i < len(decoder_in_chanels) - 1:
                self.decoder_layers.append(internal_activation())
            else:
                self.decoder_layers.append(final_activation())

        self.encoder = nn.Sequential(*self.encoder_layers)
        self.decoder = nn.Sequential(*self.decoder_layers)

    def forward(self, x):
        maxpool_ind = []
        for layer in self.encoder_layers:
            if isinstance(layer, nn.MaxPool2d):
                x, ind = layer(x)
                maxpool_ind.append(ind)
            else:
                x = layer(x)

        for layer in self.decoder_layers:
            if isinstance(layer, nn.MaxUnpool2d):
                ind = maxpool_ind.pop(-1)
                x = layer(x, ind)
            else:
                x = layer(x)
        return x
