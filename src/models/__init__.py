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

            losses = np.array(losses)
            true_labels = generator.get_true_labels()

            # ROC-AUC
            roc_auc = roc_auc_score(true_labels, losses)
            print(f'ROC-AUC on {type}: {roc_auc}')

            # MSE
            mse = losses.mean()
            print(f'MSE on {type}: {mse}')

            # F1-score & optimal threshold
            if opt_threshold is None: #validation
                precision, recall, thresholds = precision_recall_curve(y_true=true_labels, probas_pred=losses)
                f1_scores = (2 * precision * recall / (precision + recall))
                f1 = np.nanmax(f1_scores)
                opt_threshold = thresholds[np.argmax(f1_scores)]
            else:
                y_pred = (losses > opt_threshold).astype(int)
                f1 = f1_score(y_true=true_labels, y_pred=y_pred)

            print(f'F1-score on {type}: {f1}. Optimal threshold on {type}: {opt_threshold}')

            if log_to_mlflow:
                mlflow.log_metric("roc-auc", roc_auc)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("f1-score", f1)
                mlflow.log_metric("optimal mse threshold", opt_threshold)

            return opt_threshold
