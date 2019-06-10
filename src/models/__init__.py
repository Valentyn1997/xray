import torch.nn as nn
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
import mlflow
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from src import TMP_IMAGES_DIR


class BaselineAutoencoder(nn.Module):
    def __init__(self):
        super(BaselineAutoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def forward_and_save_one_image(self, inp_image, label, epoch, device, path=TMP_IMAGES_DIR):
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
    def __init__(self):
        super(BottleneckAutoencoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=1, stride=1)

        self.trconv6 = nn.ConvTranspose2d(256, 256, kernel_size=1, stride=1)
        self.unpool5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.trconv5 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.unpool4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.trconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.unpool3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.trconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.trconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.trconv1 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x, ind1 = self.pool1(x)

        x = self.conv2(x)
        x = nn.ReLU()(x)
        x, ind2 = self.pool2(x)

        x = self.conv3(x)
        x = nn.ReLU()(x)
        x, ind3 = self.pool3(x)
        #
        x = self.conv4(x)
        x = nn.ReLU()(x)
        x, ind4 = self.pool4(x)
        #
        x = self.conv5(x)
        x = nn.ReLU()(x)
        x, ind5 = self.pool5(x)

        x = self.conv6(x)
        x = nn.ReLU()(x)
        x = self.trconv6(x)
        x = nn.ReLU()(x)

        x = self.unpool5(x, ind5)
        x = self.trconv5(x)

        x = self.unpool4(x, ind4)
        x = nn.ReLU()(x)
        x = self.trconv4(x)

        x = self.unpool3(x, ind3)
        x = nn.ReLU()(x)
        x = self.trconv3(x)

        x = self.unpool2(x, ind2)
        x = nn.ReLU()(x)
        x = self.trconv2(x)

        x = self.unpool1(x, ind1)
        x = nn.ReLU()(x)
        x = self.trconv1(x)
        x = nn.Sigmoid()(x)

        return x
