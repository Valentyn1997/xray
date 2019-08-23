import os

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.autograd import Variable
from tqdm import tqdm

from src import TMP_IMAGES_DIR, TOP_K
from src.models.autoencoders import MaskedMSELoss
from src.models.gans import SpectralNorm
from src.models.outlier_scoring import Mean, TopK
from src.models.torchsummary import summary
from src.utils import save_model, log_artifact, calculate_metrics


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


class Generator(nn.Module):
    """Generator."""

    def __init__(self, image_size=64, z_dim=100, conv_dim=64):
        super(Generator, self).__init__()
        self.hyper_parameters = locals()
        self.hyper_parameters.pop('self')
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        repeat_num = int(np.log2(self.imsize)) - 3
        mult = 2 ** repeat_num  # 8
        layer1.append(SpectralNorm(nn.ConvTranspose2d(z_dim, conv_dim * mult, 4)))
        layer1.append(nn.BatchNorm2d(conv_dim * mult))
        layer1.append(nn.ReLU())

        curr_dim = conv_dim * mult

        layer2.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer2.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer2.append(nn.ReLU())

        curr_dim = int(curr_dim / 2)

        layer3.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
        layer3.append(nn.BatchNorm2d(int(curr_dim / 2)))
        layer3.append(nn.ReLU())

        if self.imsize >= 64:
            layer4 = []
            curr_dim = int(curr_dim / 2)
            layer4.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer4.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer4.append(nn.ReLU())
            self.l4 = nn.Sequential(*layer4)
            if self.imsize == 64:
                curr_dim = int(curr_dim / 2)

        if self.imsize == 128:
            layer5 = []
            curr_dim = int(curr_dim / 2)
            layer5.append(SpectralNorm(nn.ConvTranspose2d(curr_dim, int(curr_dim / 2), 4, 2, 1)))
            layer5.append(nn.BatchNorm2d(int(curr_dim / 2)))
            layer5.append(nn.ReLU())
            self.l5 = nn.Sequential(*layer5)
            curr_dim = int(curr_dim / 2)

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(nn.ConvTranspose2d(curr_dim, 1, 4, 2, 1))
        last.append(nn.Tanh())
        self.last = nn.Sequential(*last)

        if self.imsize == 128:
            self.attn1 = Self_Attn(self.imsize, 'relu')
            self.attn2 = Self_Attn(self.imsize // 2, 'relu')
        if self.imsize == 64:
            self.attn1 = Self_Attn(128, 'relu')
            self.attn2 = Self_Attn(64, 'relu')

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.l1(z)
        out = self.l2(out)
        out = self.l3(out)

        if self.imsize == 64:
            out, p1 = self.attn1(out)
            out = self.l4(out)
            out, p2 = self.attn2(out)
        if self.imsize == 128:
            out = self.l4(out)
            out, p1 = self.attn1(out)
            out = self.l5(out)
            out, p2 = self.attn2(out)
        out = self.last(out)

        return out, p1, p2


class Encoder(nn.Module):
    """Encoder"""

    def __init__(self, image_size=64, z_dim=100, conv_dim=64):
        super(Encoder, self).__init__()
        self.hyper_parameters = locals()
        self.hyper_parameters.pop('self')
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        layer1.append(SpectralNorm(nn.Conv2d(1, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize >= 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim * 2

        if self.imsize == 128:
            layer5 = []
            layer5.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer5.append(nn.LeakyReLU(0.1))
            self.l5 = nn.Sequential(*layer5)
            curr_dim = curr_dim * 2

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        self.mean = nn.Sequential(
            nn.Conv2d(curr_dim, z_dim, 4),
            nn.Tanh(),
        )
        self.log_var = nn.Sequential(
            nn.Conv2d(curr_dim, z_dim, 4),
            nn.Tanh(),
        )

        if self.imsize == 128:
            self.attn1 = Self_Attn(self.imsize * 4, 'relu')
            self.attn2 = Self_Attn(self.imsize * 8, 'relu')
        if self.imsize == 64:
            self.attn1 = Self_Attn(256, 'relu')
            self.attn2 = Self_Attn(512, 'relu')

    def forward(self, x):
        out = self.l1(x)
        out = self.l2(out)
        out = self.l3(out)
        out = self.l4(out)
        out, p1 = self.attn1(out)
        out = self.l5(out)
        out, p2 = self.attn2(out)
        mean = self.mean(out)
        log_var = self.log_var(out)

        return mean.squeeze(), log_var.squeeze(), p1, p2


class Discriminator(nn.Module):
    """Discriminator, Auxiliary Classifier."""

    def __init__(self, image_size=64, z_dim=100, conv_dim=64):
        super(Discriminator, self).__init__()
        self.hyper_parameters = locals()
        self.hyper_parameters.pop('self')
        self.imsize = image_size
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        # For inference x
        layer1.append(SpectralNorm(nn.Conv2d(1, conv_dim, 4, 2, 1)))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = conv_dim

        layer2.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer2.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        layer3.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
        layer3.append(nn.LeakyReLU(0.1))
        curr_dim = curr_dim * 2

        if self.imsize >= 64:
            layer4 = []
            layer4.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer4.append(nn.LeakyReLU(0.1))
            self.l4 = nn.Sequential(*layer4)
            curr_dim = curr_dim * 2

        if self.imsize == 128:
            layer5 = []
            layer5.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim * 2, 4, 2, 1)))
            layer5.append(nn.LeakyReLU(0.1))
            self.l5 = nn.Sequential(*layer5)
            curr_dim = curr_dim * 2

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)

        last.append(SpectralNorm(nn.Conv2d(curr_dim, 1, 4)))
        last.append(nn.Sigmoid())
        self.last = nn.Sequential(*last)

        if self.imsize == 128:
            self.attn1 = Self_Attn(self.imsize * 4, 'relu')
            self.attn2 = Self_Attn(self.imsize * 8, 'relu')
        if self.imsize == 64:
            self.attn1 = Self_Attn(256, 'relu')
            self.attn2 = Self_Attn(512, 'relu')

    def forward(self, x):
        # Inference x
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x3, p1 = self.attn1(x)
        x = self.l5(x3)
        x4, p2 = self.attn2(x)
        x5 = self.last(x4)

        return x5.squeeze(), x4, x3, p1, p2


class Codescriminator(nn.Module):
    def __init__(self, z_dim=100):
        super(Codescriminator, self).__init__()
        self.hyper_parameters = locals()
        self.hyper_parameters.pop('self')
        self.z_dim = z_dim
        layer1 = []
        layer2 = []
        layer3 = []
        last = []

        ## For inference x
        curr_dim = int(self.z_dim / 2)
        layer1.append(nn.Linear(self.z_dim, curr_dim))
        layer1.append(nn.LeakyReLU(0.1))

        curr_dim = int(curr_dim / 2)
        layer2.append(nn.Linear(curr_dim * 2, curr_dim))
        layer2.append(nn.LeakyReLU(0.1))

        # curr_dim = int(curr_dim / 2)
        # layer3.append(nn.Linear(curr_dim * 2, curr_dim))
        # layer3.append(nn.LeakyReLU(0.1))

        last.append(nn.Linear(curr_dim, 1))
        last.append(nn.Sigmoid())

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        # self.l3 = nn.Sequential(*layer3)
        self.last = nn.Sequential(*last)

    def forward(self, x):
        x = self.l1(x.squeeze())
        x = self.l2(x)
        # x = self.l3(x)
        x = self.last(x)

        return x.squeeze()


class AlphaGan(nn.Module):
    def __init__(self, device, dlr=0.00005, gelr=0.001, z_dim=100, masked_loss_on_val=True,
                 image_resolution=(128, 128), *args, **kwargs):
        super(AlphaGan, self).__init__()

        self.hyper_parameters = locals()
        self.hyper_parameters.pop('self')
        self.device = device
        self.d_z = z_dim

        self.generator = Generator(image_size=image_resolution[0], z_dim=self.d_z)
        self.discriminator = Discriminator(image_size=image_resolution[0], z_dim=self.d_z)
        self.encoder = Encoder(image_size=image_resolution[0], z_dim=self.d_z)
        self.codescriminator = Codescriminator(z_dim=self.d_z)
        self.hyper_parameters['discriminator'] = self.discriminator.hyper_parameters
        self.hyper_parameters['generator'] = self.generator.hyper_parameters
        self.hyper_parameters['encoder'] = self.encoder.hyper_parameters
        self.hyper_parameters['codiscriminator'] = self.codescriminator.hyper_parameters

        # Create batch of latent vectors that we will use to visualize the progression of the generator
        self.fixed_noise = torch.randn(32, self.d_z, device=self.device)

        # Losses
        # self.inner_loss = nn.BCELoss()
        self.l1 = nn.L1Loss()
        self.bce = nn.BCELoss()
        self.masked_loss_on_val = masked_loss_on_val
        self.outer_loss = MaskedMSELoss(reduction='none') if self.masked_loss_on_val else nn.MSELoss(reduction='none')

        # Optimizers for discriminator and generator

        self.e_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder.parameters()), gelr,
                                            (0.5, 0.999))
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()), gelr,
                                            (0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()), dlr,
                                            betas=(0.5, 0.999))
        self.c_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.codescriminator.parameters()),
                                            0.1 * dlr,
                                            betas=(0.5, 0.999))

        # Placeholders for losses of disriminator and generator
        self.g_loss = None
        self.e_loss = None
        self.c_loss = None
        self.d_loss = None

    def to(self, *args, **kwargs):
        self.generator.to(*args, **kwargs)
        self.discriminator.to(*args, **kwargs)
        self.encoder.to(*args, **kwargs)
        self.codescriminator.to(*args, **kwargs)
        return self

    def forward(self, x, discriminator=True):
        if discriminator:
            return self.discriminator(x)
        else:
            return self.generator(x)

    def parallelize(self):
        self.generator = nn.DataParallel(self.generator)
        self.discriminator = nn.DataParallel(self.discriminator)
        self.encoder = nn.DataParallel(self.encoder)
        self.codescriminator = nn.DataParallel(self.codescriminator)

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
        print('Encoder:')
        model_summary, trainable_paramsE = summary(self.encoder, input_size=(1, *image_resolution),
                                                   device=self.device)
        print('Codescriminator:')
        model_summary, trainable_paramsC = summary(self.codescriminator, input_size=(self.d_z,), device=self.device)
        return trainable_paramsG + trainable_paramsE + trainable_paramsD + trainable_paramsC

    def train_on_batch(self, batch_data, epoch, num_epochs, *args, **kwargs):
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
        self.encoder.train()
        self.codescriminator.train()

        self.discriminator.zero_grad()
        self.generator.zero_grad()
        self.encoder.zero_grad()
        self.codescriminator.zero_grad()

        # Format input batch
        real_images = Variable(batch_data['image']).to(self.device)  # Real images
        fake_z = Variable(torch.randn(real_images.size(0), self.d_z)).to(self.device)  # Noise for generator

        real_labels = Variable(torch.ones((real_images.size(0)))).to(self.device)
        fake_labels = Variable(torch.zeros((real_images.size(0)))).to(self.device)

        # Encoder
        z_mean, z_logvar, _, _ = self.encoder(real_images)
        z_hat = z_mean + z_logvar * torch.randn(z_mean.size()).to(self.device)
        # Decoder (generator)
        x_rec, x_rec4, x_rec3 = self.generator(z_hat)
        x_gen, x_gen4, x_gen3 = self.generator(fake_z)

        # Discriminator
        d_real, d_real4, d_real3, _, _ = self.discriminator(real_images)
        d_rec, d_rec4, d_rec3, _, _ = self.discriminator(x_rec)
        d_gen, d_gen4, d_gen3, _, _ = self.discriminator(x_gen)

        # Codecriminator
        c_z_hat = self.codescriminator(z_hat)
        c_z = self.codescriminator(fake_z)

        # ================== Train E ================== #
        self.e_optimizer.zero_grad()
        l1_loss = 0.01 * self.l1(real_images, x_rec)
        c_hat_loss = self.bce(c_z_hat, real_labels) - self.bce(c_z_hat, fake_labels)
        e_loss = l1_loss + c_hat_loss
        e_loss.backward(retain_graph=True)
        self.e_optimizer.step()

        # ================== Train G ================== #
        self.g_optimizer.zero_grad()
        g_rec_loss = self.bce(d_rec, real_labels) - self.bce(d_rec, fake_labels)
        g_gen_loss = self.bce(d_gen, real_labels) - self.bce(d_gen, fake_labels)
        g_loss = l1_loss + g_rec_loss + g_gen_loss
        g_loss.backward(retain_graph=True)
        self.g_optimizer.step()

        # ================== Train D ================== #
        self.d_optimizer.zero_grad()
        d_real_loss = self.bce(d_real, real_labels)
        d_rec_loss = self.bce(d_rec, fake_labels)
        d_gen_loss = self.bce(d_gen, fake_labels)
        d_loss = d_real_loss + d_rec_loss + d_gen_loss
        d_loss.backward(retain_graph=True)
        self.d_optimizer.step()

        # ================== Train C ================== #
        self.c_optimizer.zero_grad()
        c_hat_loss = self.bce(c_z_hat, fake_labels)
        c_z_loss = self.bce(c_z, real_labels)
        c_loss = c_hat_loss + c_z_loss
        c_loss.backward()
        self.c_optimizer.step()

        self.d_loss = d_loss.data
        self.g_loss = g_loss.data
        self.e_loss = e_loss.data
        self.c_loss = c_loss.data

        return {'generator loss': float(self.g_loss),
                'encoder loss': float(self.g_loss),
                'discriminator loss': float(self.d_loss),
                'codiscriminator loss': float(self.c_loss)}

    def visualize_generator(self, epoch, to_mlflow=False, is_remote=False, *args, **kwargs):
        # Check how the generator is doing by saving G's output on fixed_noise
        path = TMP_IMAGES_DIR
        # Evaluation mode
        self.generator.eval()

        with torch.no_grad():
            fake = self.generator(self.fixed_noise)[0].detach().cpu()
            img = vutils.make_grid(fake, padding=20, normalize=False)
            path = f'{path}/epoch{epoch}.png'
            vutils.save_image(img, path)

            if to_mlflow:
                log_artifact(path, 'images', is_remote=is_remote)
                os.remove(path)

    def evaluate(self, loader, log_to_mlflow=False):
        """
        Evaluates discriminator on given validation test subset
        :param loader: DataLoader of validation/test
        :param type: 'validation' or 'test'
        :param log_to_mlflow: Log metrics to Mlflow
        :param val_metrics: For :param type = 'test' only. Metrcis should contain optimal threshold
        :return: Dict of calculated metrics
        """

        # Evaluation mode
        self.generator.eval()
        self.encoder.eval()
        self.discriminator.eval()
        self.codescriminator.eval()
        scores_mse = []
        scores_proba = []
        scores_coproba = []
        scores_proba_coproba = []
        scores_top_k = []
        true_labels = []

        with torch.no_grad():

            for batch_data in tqdm(loader, desc='Validation', total=len(loader)):
                # Format input batch
                inp = batch_data['image'].to(self.device)
                mask = batch_data['mask'].to(self.device)

                # Forward pass
                z_mean, z_logvar, _, _ = self.encoder(inp)
                # z_hat = z_mean + z_logvar * torch.randn(z_mean.size()).to(self.device)
                if len(z_mean.size()) == 1:
                    z_mean = z_mean.view(1, z_mean.size(0))
                # Decoder (generator)
                x_rec, _, _ = self.generator(z_mean)

                loss = self.outer_loss(x_rec, inp, mask) if self.masked_loss_on_val \
                    else self.outer_loss(x_rec, inp)

                # Scores, based on output of discriminator - Higher score must correspond to positive labeled images
                score_proba = 1 - self.discriminator(inp)[0].to('cpu').numpy().reshape(-1)
                score_coproba = 1 - self.codescriminator(z_mean).to('cpu').numpy().reshape(-1)
                score_proba_coproba = (score_proba + score_coproba) / 2

                score_mse = Mean.calculate(loss, masked_loss=self.masked_loss_on_val, mask=mask)
                score_top_k = TopK.calculate(loss, TOP_K, reduce_to_mean=True)

                scores_mse.extend(score_mse)
                scores_top_k.extend(score_top_k)
                scores_proba.extend(score_proba)
                scores_coproba.extend(score_coproba)
                scores_proba_coproba.extend(score_proba_coproba)
                true_labels.extend(batch_data['label'].numpy())

            metrics_mse = calculate_metrics(np.array(scores_mse), np.array(true_labels), 'mse')
            metrics_mse_top_k = calculate_metrics(np.array(scores_top_k), np.array(true_labels), 'mse_top_k')
            metrics_proba = calculate_metrics(np.array(scores_proba), np.array(true_labels), 'proba')
            metrics_coproba = calculate_metrics(np.array(scores_coproba), np.array(true_labels), 'coproba')
            metrics_proba_coproba = calculate_metrics(np.array(scores_proba_coproba), np.array(true_labels),
                                                      'proba_coproba')

            metrics = {**metrics_mse, **metrics_mse_top_k, **metrics_proba, **metrics_coproba, **metrics_proba_coproba}


            if log_to_mlflow:
                for (metric, value) in metrics.items():
                    mlflow.log_metric(metric, value)

            return metrics

    def save_to_mlflow(self, is_remote=False):
        save_model(self, log_to_mlflow=True, is_remote=is_remote)

    def forward_and_save_one_image(self, inp_image, label, epoch, path=TMP_IMAGES_DIR, to_mlflow=False,
                                   is_remote=False):
        """
        Reconstructs one image and writes two images (original and reconstructed) in one figure to :param path.
        :param inp_image: Image for evaluation
        :param label: Label of image
        :param epoch: Epoch
        :param path: Path to save image to
        """
        # Evaluation mode
        self.generator.eval()
        self.encoder.eval()
        with torch.no_grad():
            # Format input batch
            inp = inp_image.to(self.device)

            # Forward pass
            z_mean, z_logvar, _, _ = self.encoder(inp)
            # z_hat = z_mean + z_logvar * torch.randn(z_mean.size()).to(self.device)
            if len(z_mean.size()) == 1:
                z_mean = z_mean.view(1, z_mean.size(0))
            x_rec, _, _ = self.generator(z_mean)

            inp_image = inp_image.to('cpu')
            x_rec = x_rec.to('cpu')
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            ax[0].imshow(inp_image.numpy()[0, 0, :, :], cmap='gray', vmin=0, vmax=1)
            ax[1].imshow(x_rec.numpy()[0, 0, :, :], cmap='gray', vmin=0, vmax=1)
            path = f'{path}/epoch{epoch}_label{int(label)}.png'
            plt.savefig(path)
            plt.close(fig)

            if to_mlflow:
                log_artifact(path, 'images', is_remote=is_remote)
                os.remove(path)
