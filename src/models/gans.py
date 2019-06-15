from typing import List

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from tqdm import tqdm

from src import TMP_IMAGES_DIR
from src.models.torchsummary import summary


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self,
                 decoder_in_chanels: List[int] = (256, 256, 128, 64, 32, 16, 8, 4),
                 decoder_out_chanels: List[int] = (256, 128, 64, 32, 16, 8, 4, 1),
                 decoder_kernel_sizes: List[int] = (4, 4, 4, 4, 4, 4, 4, 4),
                 decoder_strides: List[int] = (1, 2, 2, 2, 2, 2, 2, 2),
                 decoder_paddings: List[int] = (0, 1, 1, 1, 1, 1, 1, 1),
                 use_batchnorm=True,
                 internal_activation=nn.ReLU,
                 final_activation=nn.Sigmoid):
        super(Generator, self).__init__()

        self.decoder_layers = []
        for i in range(len(decoder_in_chanels)):
            self.decoder_layers.append(nn.ConvTranspose2d(decoder_in_chanels[i], decoder_out_chanels[i],
                                                          kernel_size=decoder_kernel_sizes[i],
                                                          stride=decoder_strides[i],
                                                          padding=decoder_paddings[i],
                                                          bias=not use_batchnorm))
            if use_batchnorm and i < len(decoder_in_chanels) - 1:  # no batch norm after last convolution
                self.decoder_layers.append(nn.BatchNorm2d(decoder_out_chanels[i]))
            if i < len(decoder_in_chanels) - 1:
                self.decoder_layers.append(internal_activation())
            else:
                self.decoder_layers.append(final_activation())

        self.generator = nn.Sequential(*self.decoder_layers)

    def forward(self, x):
        return self.generator(x)


class Discriminator(nn.Module):
    def __init__(self,
                 encoder_in_chanels: List[int] = (1, 4, 8, 16, 32, 64, 128, 256),
                 encoder_out_chanels: List[int] = (4, 8, 16, 32, 64, 128, 256, 1),
                 encoder_kernel_sizes: List[int] = (4, 4, 4, 4, 4, 4, 4, 4),
                 encoder_strides: List[int] = (2, 2, 2, 2, 2, 2, 2, 1),
                 encoder_paddings: List[int] = (1, 1, 1, 1, 1, 1, 1, 0),
                 use_batchnorm: bool = True,
                 internal_activation=nn.ReLU,
                 final_activation=nn.Sigmoid):
        super(Discriminator, self).__init__()

        self.encoder_layers = []
        for i in range(len(encoder_in_chanels)):
            self.encoder_layers.append(nn.Conv2d(encoder_in_chanels[i], encoder_out_chanels[i],
                                                 kernel_size=encoder_kernel_sizes[i],
                                                 stride=encoder_strides[i],
                                                 padding=encoder_paddings[i],
                                                 bias=not use_batchnorm))
            if use_batchnorm:
                self.encoder_layers.append(nn.BatchNorm2d(encoder_out_chanels[i]))
            if i < len(encoder_in_chanels) - 1:
                self.encoder_layers.append(internal_activation())
            else:
                self.encoder_layers.append(final_activation())

        self.discriminator = nn.Sequential(*self.encoder_layers)

    def forward(self, x):
        return self.discriminator(x)


class DCGAN(nn.Module):

    def __init__(self, device, use_batchnorm=True, *args, **kwargs):
        super(DCGAN, self).__init__()
        self.generator = Generator(use_batchnorm=use_batchnorm)
        self.discriminator = Discriminator(use_batchnorm=use_batchnorm)
        weights_init(self.generator)
        weights_init(self.discriminator)

        # Initialize BCELoss function
        self.inner_loss = nn.BCELoss()
        self.outer_loss = nn.BCELoss(reduction='none')

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        self.fixed_noise = torch.randn(32, 256, 1, 1, device=device)

        # Establish convention for real and fake labels during training
        self.real_label = 1
        self.fake_label = 0

        # Setup Adam optimizers for both G and D
        self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))

        self.errD = None
        self.errG = None
        self.iter = 0

    def to(self, *args, **kwargs):
        self.generator.to(*args, **kwargs)
        self.discriminator.to(*args, **kwargs)
        return self

    def forward(self, x, discriminator=True):
        if discriminator:
            return self.discriminator(x)
        else:
            return self.generator(x)

    def summary(self, device, image_resolution):
        print('Generator:')
        model_summary, trainable_paramsG = summary(self.generator, input_size=(256, 1, 1), device=device)
        print('Discriminator:')
        model_summary, trainable_paramsD = summary(self.discriminator, input_size=(1, *image_resolution), device=device)
        return trainable_paramsG + trainable_paramsD

    def train_on_batch(self, batch_data, device, epoch, num_epochs):

        # Format batch
        real_inp = batch_data['image'].to(device)
        b_size = real_inp.size(0)
        label = torch.full((b_size,), self.real_label, device=device)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # Train with all-real batch
        self.discriminator.zero_grad()
        # Forward pass real batch through D
        output = self.discriminator(real_inp).view(-1)
        # Calculate loss on all-real batch
        errD_real = self.inner_loss(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        # D_x = output.mean().item()

        # Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, 256, 1, 1, device=device)
        # Generate fake image batch with G
        fake = self.generator(noise)
        label.fill_(self.fake_label)
        # Classify all fake batch with D
        output = self.discriminator(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = self.inner_loss(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        # D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        self.errD = float(errD.data)
        # Update D
        self.optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        self.generator.zero_grad()
        label.fill_(self.real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = self.discriminator(fake).view(-1)
        # Calculate G's loss based on this output
        errG = self.inner_loss(output, label)
        # Calculate gradients for G
        errG.backward()
        # D_G_z2 = output.mean().item()
        # Update G
        self.errG = float(errG.data)
        self.optimizerG.step()

        # Output training stats
        # if self.iter % 50 == 0:
        #     print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
        #           % (epoch, num_epochs, errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        # G_losses.append(errG.item())
        # D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        path = TMP_IMAGES_DIR
        if self.iter % 100 == 0:
            with torch.no_grad():
                fake = self.generator(self.fixed_noise).detach().cpu()
            img = vutils.make_grid(fake, padding=20, normalize=False)
            vutils.save_image(img, f'{path}/epoch{epoch}.png')

        self.iter += 1

        return errG + errD

    def evaluate(self, loader, type, device, log_to_mlflow=False, val_metrics=None):

        opt_threshold = val_metrics['optimal discriminator_proba threshold'] if val_metrics is not None else None
        self.eval()
        with torch.no_grad():
            scores = []
            true_labels = []
            for batch_data in tqdm(loader, desc=type, total=len(loader)):
                inp = batch_data['image'].to(device)

                # forward pass
                output = self.discriminator(inp)
                score = 1 - output.to('cpu').numpy().reshape(-1)
                scores.extend(score)
                true_labels.extend(batch_data['label'].numpy())

            scores = np.array(scores)
            true_labels = np.array(true_labels)

            # ROC-AUC
            roc_auc = roc_auc_score(true_labels, scores)
            # Discriminator proba
            discriminator_proba = scores.mean()
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
            print(f'Discriminator proba of fake on {type}: {discriminator_proba}')
            print(f'F1-score on {type}: {f1}. Optimal threshold on {type}: {opt_threshold}')

            metrics = {"d_loss_train": self.errD,
                       "g_loss_train": self.errG,
                       "roc-auc": roc_auc,
                       "discriminator_proba": discriminator_proba,
                       "f1-score": f1,
                       "optimal discriminator_proba threshold": opt_threshold}

            if log_to_mlflow:
                for (metric, value) in metrics.items():
                    mlflow.log_metric(metric, value)

            return metrics
