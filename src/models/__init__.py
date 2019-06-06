import torch.nn as nn
import torch
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
import mlflow
from tqdm import tqdm
import numpy as np


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

    def evaluate(self, generator, type, loss, device, log_to_mlflow=False, opt_threshold=None):

        with torch.no_grad():
            losses = []
            for batch in tqdm(range(len(generator)), desc=type):
                inp = Variable(torch.from_numpy(generator[batch]).float()).to(device)

                # forward pass
                output = self(inp)
                losses.extend(loss(output, inp).to('cpu').numpy().mean(axis=(1, 2, 3)))

            generator.on_epoch_end()

            losses = np.array(losses)
            true_labels = generator.get_true_labels()

            # ROC-AUC
            roc_auc = roc_auc_score(true_labels, losses)
            print(f'ROC-AUC on {type}: {roc_auc}')

            # MSE
            mse = losses.mean()
            print(f'MSE on {type}: {mse}')

            # F1-score & optimal threshold
            if opt_threshold is None:  # validation
                precision, recall, thresholds = precision_recall_curve(y_true=true_labels, probas_pred=losses)
                f1_scores = (2 * precision * recall / (precision + recall))
                f1 = np.nanmax(f1_scores)
                opt_threshold = thresholds[np.argmax(f1_scores)]
            else:  # testing
                y_pred = (losses > opt_threshold).astype(int)
                f1 = f1_score(y_true=true_labels, y_pred=y_pred)

            print(f'F1-score on {type}: {f1}. Optimal threshold on {type}: {opt_threshold}')

            if log_to_mlflow:
                mlflow.log_metric("roc-auc", roc_auc)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("f1-score", f1)
                mlflow.log_metric("optimal mse threshold", opt_threshold)

            return opt_threshold


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
