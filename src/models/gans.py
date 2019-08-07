from typing import List

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, average_precision_score
from torch.distributions.uniform import Uniform
from torch.nn import Parameter
from tqdm import tqdm

from src import TMP_IMAGES_DIR
from src.models.torchsummary import summary
from src.utils import save_model


# custom weights initialization called on discriminator and generator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class Generator(nn.Module):
    d_z = 2048

    def __init__(self,
                 decoder_in_chanels: List[int] = (d_z, 1024, 512, 256, 128, 64, 32, 16),
                 decoder_out_chanels: List[int] = (1024, 512, 256, 128, 64, 32, 16, 1),
                 decoder_kernel_sizes: List[int] = (4, 4, 4, 4, 4, 4, 4, 4),
                 decoder_strides: List[int] = (1, 2, 2, 2, 2, 2, 2, 2),
                 decoder_paddings: List[int] = (0, 1, 1, 1, 1, 1, 1, 1),
                 batch_normalisation=True,
                 spectral_normalisation=True,
                 internal_activation=nn.ReLU,
                 final_activation=nn.Sigmoid):
        super(Generator, self).__init__()
        self.hyper_parameters = locals()
        self.hyper_parameters.pop('self')

        # Decoder initialization
        self.decoder_layers = []
        for i in range(len(decoder_in_chanels)):
            trans_conv = nn.ConvTranspose2d(decoder_in_chanels[i], decoder_out_chanels[i],
                                            kernel_size=decoder_kernel_sizes[i],
                                            stride=decoder_strides[i],
                                            padding=decoder_paddings[i],
                                            bias=not batch_normalisation)

            if spectral_normalisation:
                self.decoder_layers.append(SpectralNorm(trans_conv))
            else:
                self.decoder_layers.append(trans_conv)
            if batch_normalisation and i < len(decoder_in_chanels) - 1:  # no batch norm after last convolution
                self.decoder_layers.append(nn.BatchNorm2d(decoder_out_chanels[i]))
            if i < len(decoder_in_chanels) - 1:
                self.decoder_layers.append(internal_activation())
            else:
                self.decoder_layers.append(final_activation())

        # self.attn1 = Self_Attn(128, 'relu')
        # self.attn2 = Self_Attn(64, 'relu')
        self.generator = nn.Sequential(*self.decoder_layers)

    def forward(self, x):
        return self.generator(x)


class Discriminator(nn.Module):
    def __init__(self,
                 encoder_in_chanels: List[int] = (1, 4, 8, 16, 32, 64, 128, 256),
                 encoder_out_chanels: List[int] = (4, 8, 16, 32, 64, 128, 256, 1),
                 encoder_kernel_sizes: List[int] = (4, 4, 4, 4, 4, 4, 4, 4, 4),
                 encoder_strides: List[int] = (2, 2, 2, 2, 2, 2, 2, 1),
                 encoder_paddings: List[int] = (1, 1, 1, 1, 1, 1, 1, 0),
                 batch_normalisation: bool = True,
                 spectral_normalisation=True,
                 internal_activation=nn.ReLU,
                 final_activation=nn.Sigmoid):
        super(Discriminator, self).__init__()
        self.hyper_parameters = locals()
        self.hyper_parameters.pop('self')

        # Encoder initialization
        self.encoder_layers = []
        for i in range(len(encoder_in_chanels)):
            conv = nn.Conv2d(encoder_in_chanels[i], encoder_out_chanels[i],
                             kernel_size=encoder_kernel_sizes[i],
                             stride=encoder_strides[i],
                             padding=encoder_paddings[i],
                             bias=not batch_normalisation)
            if spectral_normalisation:
                self.encoder_layers.append(SpectralNorm(conv))
            else:
                self.encoder_layers.append(conv)

            if batch_normalisation:
                self.encoder_layers.append(nn.BatchNorm2d(encoder_out_chanels[i]))
            if i < len(encoder_in_chanels) - 1:
                self.encoder_layers.append(internal_activation())
            else:
                self.encoder_layers.append(final_activation())

        self.discriminator = nn.Sequential(*self.encoder_layers)

    def forward(self, x):
        return self.discriminator(x)


class DCGAN(nn.Module):

    def __init__(self, device, batch_normalisation=True, spectral_normalisation=True,
                 soft_labels=True, dlr=0.00005, glr=0.001, soft_delta=0.1, *args, **kwargs):
        super(DCGAN, self).__init__()

        self.hyper_parameters = locals()
        self.hyper_parameters.pop('self')
        self.device = device

        self.soft_labels = soft_labels
        self.d_z = Generator.d_z
        self.generator = Generator(batch_normalisation=batch_normalisation,
                                   spectral_normalisation=spectral_normalisation)
        self.discriminator = Discriminator(batch_normalisation=batch_normalisation,
                                           spectral_normalisation=spectral_normalisation)
        self.hyper_parameters['discriminator'] = self.discriminator.hyper_parameters
        self.hyper_parameters['generator'] = self.generator.hyper_parameters

        weights_init(self.generator)
        weights_init(self.discriminator)

        # Create batch of latent vectors that we will use to visualize the progression of the generator
        self.fixed_noise = torch.randn(32, self.d_z, 1, 1, device=self.device)

        # Establish convention for real and fake labels during training
        self.real_label = 0
        self.fake_label = 1
        self.soft_delta = soft_delta
        self.real_labels_softener = Uniform(low=0.0, high=self.soft_delta) if bool(self.fake_label) else Uniform(
            low=-self.soft_delta, high=0.0)
        self.fake_labels_softener = Uniform(low=-self.soft_delta, high=0.0) if bool(self.fake_label) else Uniform(
            low=0.0, high=self.soft_delta)

        # Losses
        self.inner_loss = nn.BCELoss()
        self.outer_loss = nn.BCELoss(reduction='none')

        # Optimizers for discriminator and generator
        self.optimizerD = torch.optim.Adam(self.discriminator.parameters(), lr=dlr, betas=(0.5, 0.999))
        self.optimizerG = torch.optim.Adam(self.generator.parameters(), lr=glr, betas=(0.5, 0.999))

        # Placeholders for losses of disriminator and generator
        self.loss_D = None
        self.loss_G = None

    def to(self, *args, **kwargs):
        self.generator.to(*args, **kwargs)
        self.discriminator.to(*args, **kwargs)
        return self

    def forward(self, x, discriminator=True):
        if discriminator:
            return self.discriminator(x)
        else:
            return self.generator(x)

    def summary(self, image_resolution):
        """
        Print summary of model
        :param image_resolution: input image resolution (H, W)
        :return: number of trainable parameters
        """
        print('Generator:')
        model_summary, trainable_paramsG = summary(self.generator, input_size=(self.d_z, 1, 1), device=self.device)
        print('Discriminator:')
        model_summary, trainable_paramsD = summary(self.discriminator, input_size=(1, *image_resolution),
                                                   device=self.device)
        return trainable_paramsG + trainable_paramsD

    def train_on_batch(self, batch_data, epoch, *args, **kwargs):
        """
        Performs one step of gradient descent on batch_data
        :param batch_data: Data of batch
        :param args:
        :param kwargs:
        :return: Dict of losses
        """
        # Switching to train modes
        self.discriminator.train()
        self.generator.train()

        # Format input batch
        real_inp = batch_data['image'].to(self.device)  # Real images
        b_size = real_inp.size(0)  # Batch size
        noise_inp = torch.randn(b_size, self.d_z, 1, 1, device=self.device)  # Noise for generator
        label = torch.full((b_size,), self.real_label, device=self.device)  # Real labels vector
        if self.soft_labels:
            label += self.real_labels_softener.sample((b_size,)).to(self.device)

        # Backward pass (1) Update discriminator network: maximize log(D(x)) + log(1 - D(G(z)))
        self.discriminator.zero_grad()
        output = self.discriminator(real_inp).view(-1)
        # Train with all-real batch
        loss_D_real = self.inner_loss(output, label)
        loss_D_real.backward()

        # Train with all-fake batch
        fake = self.generator(noise_inp)
        label.fill_(self.fake_label)
        if self.soft_labels:
            label += self.fake_labels_softener.sample((b_size,)).to(self.device)
        output = self.discriminator(fake.detach()).view(-1)  # detached fake - not to backprop on generator
        loss_D_fake = self.inner_loss(output, label)
        # Calculate the gradients for this batch
        loss_D_fake.backward()
        # Add the gradients from the all-real and all-fake batches
        self.loss_D = float((loss_D_real + loss_D_real).data)
        # Update discriminator
        self.optimizerD.step()

        # (2) Update generator network: maximize log(D(G(z)))
        self.generator.zero_grad()
        label.fill_(self.real_label)  # fake labels are real for generator cost
        # Since we just updated discriminator, perform another forward pass of all-fake batch through D
        output = self.discriminator(fake).view(-1)
        # Calculate G's loss based on this output
        loss_G = self.inner_loss(output, label)
        # Calculate gradients for G
        loss_G.backward()
        # Update G
        self.loss_G = float(loss_G.data)
        self.optimizerG.step()

        return {'generator loss': float(self.loss_G),
                'discriminator loss': float(self.loss_D)}

    def visualize_generator(self, epoch, *args, **kwargs):
        # Check how the generator is doing by saving G's output on fixed_noise
        path = TMP_IMAGES_DIR
        # Evaluation mode
        self.generator.eval()

        with torch.no_grad():
            fake = self.generator(self.fixed_noise).detach().cpu()
        img = vutils.make_grid(fake, padding=20, normalize=False)
        vutils.save_image(img, f'{path}/epoch{epoch}.png')

    def evaluate(self, loader, type, log_to_mlflow=False, val_metrics=None):
        """
        Evaluates discriminator on given validation test subset
        :param loader: DataLoader of validation/test
        :param type: 'validation' or 'test'
        :param log_to_mlflow: Log metrics to Mlflow
        :param val_metrics: For :param type = 'test' only. Metrcis should contain optimal threshold
        :return: Dict of calculated metrics
        """
        # Extracting optimal threshold
        opt_threshold = val_metrics['optimal discriminator_proba threshold'] if val_metrics is not None else None

        # Evaluation mode
        self.discriminator.eval()
        with torch.no_grad():
            scores = []
            true_labels = []
            for batch_data in tqdm(loader, desc=type, total=len(loader)):
                # Format input batch
                inp = batch_data['image'].to(self.device)

                # Forward pass
                output = self.discriminator(inp).to('cpu').numpy().reshape(-1)

                # Scores, based on output of discriminator - Higher score must correspond to positive labeled images
                score = output if bool(self.real_label) else 1 - output

                scores.extend(score)
                true_labels.extend(batch_data['label'].numpy())

            scores = np.array(scores)
            true_labels = np.array(true_labels)

            # ROC-AUC
            roc_auc = roc_auc_score(true_labels, scores)

            # APS
            aps = average_precision_score(true_labels, scores)

            # Mean discriminator proba on validation batch
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
            print(f'Discriminator proba on {type}: {discriminator_proba}')
            print(f'F1-score on {type}: {f1}. Optimal threshold on {type}: {opt_threshold}')

            metrics = {"d_loss_train": self.loss_D,
                       "g_loss_train": self.loss_G,
                       "roc-auc": roc_auc,
                       "aps": aps,
                       "discriminator_proba": discriminator_proba,
                       "f1-score": f1,
                       "optimal discriminator_proba threshold": opt_threshold}

            if log_to_mlflow:
                for (metric, value) in metrics.items():
                    mlflow.log_metric(metric, value)

            return metrics

    def save_to_mlflow(self):
        save_model(self, log_to_mlflow=True)
