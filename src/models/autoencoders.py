from typing import List

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from tqdm import tqdm

from src import TMP_IMAGES_DIR
from src.models.torchsummary import summary
from src.utils import save_model


class MaskedMSELoss(nn.Module):
    def __init__(self, reduction='mean'):
        """
        Masked MSE implementation
        :param reduction: the same, as in nn.MSELoss
        """
        super(MaskedMSELoss, self).__init__()
        self.reduction = reduction
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, input, target, mask):
        """
        calculates masked loss
        :param input: input image as array
        :param target: reconstructed image as array
        :param mask: mask of image as array
        :return: masked loss
        """
        loss = self.criterion(input * mask, target * mask)
        if self.reduction == 'mean':
            loss = torch.sum(loss) / torch.sum(mask)
        return loss


class BaselineAutoencoder(nn.Module):
    def __init__(self,
                 device,
                 encoder_in_chanels: List[int] = (1, 16, 32, 32, 64, 64, 128, 128, 256, 256),
                 encoder_out_chanels: List[int] = (16, 32, 32, 64, 64, 128, 128, 256, 256, 512),
                 encoder_kernel_sizes: List[int] = (3, 4, 3, 4, 3, 4, 3, 4, 3, 4),
                 encoder_strides: List[int] = (1, 2, 1, 2, 1, 2, 1, 2, 1, 2),
                 decoder_in_chanels: List[int] = (512, 256, 128, 64, 32, 16),
                 decoder_out_chanels: List[int] = (256, 128, 64, 32, 16, 1),
                 decoder_kernel_sizes: List[int] = (4, 4, 4, 4, 4, 3),
                 decoder_strides: List[int] = (2, 2, 2, 2, 2, 1),
                 batch_normalisation: bool = True,
                 internal_activation=nn.ReLU,
                 final_activation=nn.Tanh,
                 masked_loss_on_val=False,
                 masked_loss_on_train=False,
                 lr=0.001,
                 *args, **kwargs):

        super(BaselineAutoencoder, self).__init__()

        self.device = device
        self.hyper_parameters = locals()
        self.hyper_parameters.pop('self')

        # Encoder initialization
        self.encoder_layers = []
        for i in range(len(encoder_in_chanels)):
            self.encoder_layers.append(nn.Conv2d(encoder_in_chanels[i], encoder_out_chanels[i],
                                                 kernel_size=encoder_kernel_sizes[i],
                                                 stride=encoder_strides[i], padding=1, bias=not batch_normalisation))
            if batch_normalisation:
                self.encoder_layers.append(nn.BatchNorm2d(encoder_out_chanels[i]))
            self.encoder_layers.append(internal_activation())

        # Decoder initialization
        self.decoder_layers = []
        for i in range(len(decoder_in_chanels)):
            self.decoder_layers.append(nn.ConvTranspose2d(decoder_in_chanels[i], decoder_out_chanels[i],
                                                          kernel_size=decoder_kernel_sizes[i],
                                                          stride=decoder_strides[i], padding=1,
                                                          bias=not batch_normalisation))
            if batch_normalisation and i < len(decoder_in_chanels) - 1:  # no batch norm after last convolution
                self.decoder_layers.append(nn.BatchNorm2d(decoder_out_chanels[i]))
            self.decoder_layers.append(internal_activation())
        self.decoder_layers.append(final_activation())

        self.encoder = nn.Sequential(*self.encoder_layers)
        self.decoder = nn.Sequential(*self.decoder_layers)

        # Losses
        self.masked_loss_on_val = masked_loss_on_val
        self.masked_loss_on_train = masked_loss_on_train
        self.inner_loss = MaskedMSELoss() if self.masked_loss_on_train else nn.MSELoss()
        self.outer_loss = MaskedMSELoss(reduction='none') if self.masked_loss_on_val else nn.MSELoss(reduction='none')

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def forward_and_save_one_image(self, inp_image, label, epoch, path=TMP_IMAGES_DIR):
        """
        Reconstructs one image and writes two images (original and reconstructed) in one figure to :param path.
        :param inp_image: Image for evaluation
        :param label: Label of image
        :param epoch: Epoch
        :param path: Path to save image to
        """
        # Evaluation mode
        self.eval()
        with torch.no_grad():
            # Format input batch
            inp = inp_image.to(self.device)

            # Forward pass
            output = self(inp)
            output_img = output.to('cpu')

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            ax[0].imshow(inp_image.numpy()[0, 0, :, :], cmap='gray', vmin=0, vmax=1)
            ax[1].imshow(output_img.numpy()[0, 0, :, :], cmap='gray', vmin=0, vmax=1)
            plt.savefig(f'{path}/epoch{epoch}_label{int(label)}.png')
            plt.close(fig)

        return output

    def evaluate(self, loader, type, log_to_mlflow=False, val_metrics=None):
        """
        Evaluates model on given validation test subset
        :param loader: DataLoader of validation/test
        :param type: 'validation' or 'test'
        :param log_to_mlflow: Log metrics to Mlflow
        :param val_metrics: For :param type = 'test' only. Metrcis should contain optimal threshold
        :return: Dict of calculated metrics
        """

        # Extracting optimal threshold
        opt_threshold = val_metrics['optimal mse threshold'] if val_metrics is not None else None

        # Evaluation mode
        self.eval()
        with torch.no_grad():
            scores = []
            true_labels = []
            for batch_data in tqdm(loader, desc=type, total=len(loader)):
                # Format input batch
                inp = batch_data['image'].to(self.device)
                mask = batch_data['mask'].to(self.device)

                # Forward pass
                output = self(inp)
                loss = self.outer_loss(output, inp, mask) if self.masked_loss_on_val else self.outer_loss(output, inp)

                # Scores, based on MSE - higher MSE correspond to abnormal image
                if self.masked_loss_on_val:
                    sum_loss = loss.to('cpu').numpy().sum(axis=(1, 2, 3))
                    sum_mask = mask.to('cpu').numpy().sum(axis=(1, 2, 3))
                    score = sum_loss / sum_mask
                else:
                    score = loss.to('cpu').numpy().mean(axis=(1, 2, 3))

                scores.extend(score)
                true_labels.extend(batch_data['label'].numpy())

        scores = np.array(scores)
        true_labels = np.array(true_labels)

        # ROC-AUC
        roc_auc = roc_auc_score(true_labels, scores)
        # Mean score on validation
        mse = scores.mean()
        # F1-score & optimal threshold
        if opt_threshold is None:  # validation
            precision, recall, thresholds = precision_recall_curve(y_true=true_labels, probas_pred=scores)
            f1_scores = (2 * precision * recall / (precision + recall))
            f1 = np.nanmax(f1_scores)
            opt_threshold = thresholds[np.argmax(f1_scores)]
        else:  # testing
            y_pred = (scores > opt_threshold).astype(int)
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
        inp = batch_data['image'].to(self.device)
        mask = batch_data['mask'].to(self.device)

        # Forward pass
        output = self(inp)
        loss = self.inner_loss(inp, output, mask) if self.masked_loss_on_train else self.inner_loss(inp, output)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'mse': float(loss.data)}

    def summary(self, image_resolution):
        """
        Print summary of model
        :param image_resolution: input image resolution (H, W)
        :return: number of trainable parameters
        """
        model_summary, trainable_params = summary(self, input_size=(1, *image_resolution), device=self.device)
        return trainable_params

    def save_to_mlflow(self):
        save_model(self, log_to_mlflow=True)


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
                 batch_normalisation: bool = True,
                 internal_activation=nn.ReLU,
                 final_activation=nn.Sigmoid,
                 lr=0.001,
                 *args, **kwargs):

        super(BottleneckAutoencoder, self).__init__(*args, **kwargs)
        self.hyper_parameters = locals()
        self.hyper_parameters.pop('self')

        # Encoder initialization
        self.encoder_layers = []
        for i in range(len(encoder_in_chanels)):
            self.encoder_layers.append(nn.Conv2d(encoder_in_chanels[i], encoder_out_chanels[i],
                                                 kernel_size=encoder_kernel_sizes[i],
                                                 stride=encoder_strides[i], padding=1, bias=not batch_normalisation))
            if i < len(encoder_in_chanels) - 1:
                self.encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
            if batch_normalisation:
                self.encoder_layers.append(nn.BatchNorm2d(encoder_out_chanels[i]))
            self.encoder_layers.append(internal_activation())

        # Decoder initialization
        self.decoder_layers = []
        for i in range(len(decoder_in_chanels)):
            self.decoder_layers.append(nn.ConvTranspose2d(decoder_in_chanels[i], decoder_out_chanels[i],
                                                          kernel_size=decoder_kernel_sizes[i],
                                                          stride=decoder_strides[i],
                                                          padding=1, bias=not batch_normalisation))
            if i < len(decoder_in_chanels) - 1:
                self.decoder_layers.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
            if batch_normalisation and i < len(decoder_in_chanels) - 1:  # no batch norm after last convolution
                self.decoder_layers.append(nn.BatchNorm2d(decoder_out_chanels[i]))
            if i < len(decoder_in_chanels) - 1:
                self.decoder_layers.append(internal_activation())
            else:
                self.decoder_layers.append(final_activation())

        self.encoder = nn.Sequential(
            *self.encoder_layers)  # Not used in forward pass, but without it summary() doesn't work
        self.decoder = nn.Sequential(
            *self.decoder_layers)  # Not used in forward pass, but without it summary() doesn't work

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

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


class SkipConnection(BottleneckAutoencoder):
    """Similar to bottleneck autoencoder but with some skip connection after ReLu"""
    def __init__(self,
                 encoder_in_chanels: List[int] = (1, 16, 32, 64, 128, 256),
                 encoder_out_chanels: List[int] = (16, 32, 64, 128, 256, 256),
                 encoder_kernel_sizes: List[int] = (3, 4, 4, 4, 4, 1),
                 encoder_strides: List[int] = (1, 2, 2, 2, 2, 1),
                 decoder_in_chanels: List[int] = (256, 512, 256, 64, 32, 16),
                 decoder_out_chanels: List[int] = (256, 128, 64, 32, 16, 1),
                 decoder_kernel_sizes: List[int] = (1, 4, 4, 4, 4, 3),
                 decoder_strides: List[int] = (1, 2, 2, 2, 2, 1),
                 batch_normalisation: bool = True,
                 skip_connection_encoder: List[bool] = (False, False, False, True, True, False),
                 skip_connection_decoder: List[bool] = (False, True, True, False, False, False),
                 lr=0.001,
                 internal_activation=nn.ReLU,
                 final_activation=nn.Sigmoid,
                 *args, **kwargs):

        super(SkipConnection, self).__init__(*args, **kwargs)

        self.hyper_parameters = locals()
        self.hyper_parameters.pop('self')

        # Encoder initialization
        self.encoder_layers = []
        for i in range(len(encoder_in_chanels)):
            self.encoder_layers.append(nn.Conv2d(encoder_in_chanels[i], encoder_out_chanels[i],
                                                 kernel_size=encoder_kernel_sizes[i],
                                                 stride=encoder_strides[i], padding=1, bias=not batch_normalisation))
            self.encoder_layers.append(internal_activation())
            if batch_normalisation:
                self.encoder_layers.append(nn.BatchNorm2d(encoder_out_chanels[i]))
            if i < len(encoder_in_chanels) - 1:
                self.encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))

        # Decoder initialization
        self.decoder_layers = []
        for i in range(len(decoder_in_chanels)):
            self.decoder_layers.append(nn.ConvTranspose2d(decoder_in_chanels[i], decoder_out_chanels[i],
                                                          kernel_size=decoder_kernel_sizes[i],
                                                          stride=decoder_strides[i],
                                                          padding=1, bias=not batch_normalisation))
            if i < len(decoder_in_chanels) - 1:
                self.decoder_layers.append(internal_activation())
            if batch_normalisation and i < len(decoder_in_chanels) - 1:  # no batch norm after last convolution
                self.decoder_layers.append(nn.BatchNorm2d(decoder_out_chanels[i]))
            if i < len(decoder_in_chanels) - 1:
                self.decoder_layers.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
            else:
                self.decoder_layers.append(final_activation())

        self.skip_connection_encoder = skip_connection_encoder
        self.skip_connection_decoder = skip_connection_decoder

        self.encoder = nn.ModuleList(
            self.encoder_layers)
        self.decoder = nn.ModuleList(
            self.decoder_layers)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        maxpool_ind = []
        skip_connection_value = []
        m = 0  # index relu for encoder
        n = 0  # index of skip connection value
        o = 0  # index convolutional layer decoder

        for layer in self.encoder:
            if isinstance(layer, nn.MaxPool2d):
                x, ind = layer(x)
                maxpool_ind.append(ind)
            elif isinstance(layer, nn.ReLU):
                x = layer(x)
                # save value if skip connection is true
                if self.skip_connection_encoder[m]:
                    skip_connection_value.append(x)
                m = m + 1
            else:
                x = layer(x)
        # reverse the order of the list since decoder works the other way around
        skip_connection_value.reverse()

        for layer in self.decoder:

            if isinstance(layer, nn.MaxUnpool2d):
                ind = maxpool_ind.pop(-1)
                x = layer(x, ind)
            elif isinstance(layer, nn.ConvTranspose2d):
                if self.skip_connection_decoder[o]:
                    # add values of skip connection
                    x = torch.cat((x, skip_connection_value[n]), dim=1)
                    n = n + 1
                x = layer(x)
                o = o + 1
            else:
                x = layer(x)

        return x


class Bottleneck(BaselineAutoencoder):
    """Differs from bottleneckAutoencoder by switching the order of batchnormalization and activation"""
    def __init__(self,
                 encoder_in_chanels: List[int] = (1, 16, 32, 64, 128, 256),
                 encoder_out_chanels: List[int] = (16, 32, 64, 128, 256, 256),
                 encoder_kernel_sizes: List[int] = (3, 4, 4, 4, 4, 3),
                 encoder_strides: List[int] = (1, 2, 2, 2, 2, 1),
                 decoder_in_chanels: List[int] = (256, 256, 128, 64, 32, 16),
                 decoder_out_chanels: List[int] = (256, 128, 64, 32, 16, 1),
                 decoder_kernel_sizes: List[int] = (3, 4, 4, 4, 4, 3),
                 decoder_strides: List[int] = (1, 2, 2, 2, 2, 1),
                 batch_normalisation: bool = True,
                 internal_activation=nn.ReLU,
                 final_activation=nn.Sigmoid,
                 lr=0.001,
                 *args, **kwargs):

        super(Bottleneck, self).__init__(*args, **kwargs)
        self.hyper_parameters = locals()
        self.hyper_parameters.pop('self')

        # Encoder initialization
        self.encoder_layers = []
        for i in range(len(encoder_in_chanels)):
            self.encoder_layers.append(nn.Conv2d(encoder_in_chanels[i], encoder_out_chanels[i],
                                                 kernel_size=encoder_kernel_sizes[i],
                                                 stride=encoder_strides[i], padding=1, bias=not batch_normalisation))
            if i < len(encoder_in_chanels) - 1:
                self.encoder_layers.append(nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True))
            self.encoder_layers.append(internal_activation())
            if batch_normalisation:
                self.encoder_layers.append(nn.BatchNorm2d(encoder_out_chanels[i]))

        # Decoder initialization
        self.decoder_layers = []
        for i in range(len(decoder_in_chanels)):
            self.decoder_layers.append(nn.ConvTranspose2d(decoder_in_chanels[i], decoder_out_chanels[i],
                                                          kernel_size=decoder_kernel_sizes[i],
                                                          stride=decoder_strides[i],
                                                          padding=1, bias=not batch_normalisation))
            if i < len(decoder_in_chanels) - 1:
                self.decoder_layers.append(nn.MaxUnpool2d(kernel_size=2, stride=2))
            if i < len(decoder_in_chanels) - 1:
                self.decoder_layers.append(internal_activation())
            else:
                self.decoder_layers.append(final_activation())
            if batch_normalisation and i < len(decoder_in_chanels) - 1:  # no batch norm after last convolution
                self.decoder_layers.append(nn.BatchNorm2d(decoder_out_chanels[i]))

        self.encoder = nn.ModuleList(
            self.encoder_layers)  # Not used in forward pass, but without it summary() doesn't work
        self.decoder = nn.ModuleList(
            self.decoder_layers)  # Not used in forward pass, but without it summary() doesn't work

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

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
