import itertools

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from torch.autograd import Variable
from tqdm import tqdm

from src import TMP_IMAGES_DIR
from src.models.autoencoders import MaskedMSELoss
from src.models.gans import SpectralNorm
from src.models.torchsummary import summary


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

        last.append(nn.Conv2d(curr_dim, z_dim, 4))
        self.last = nn.Sequential(*last)

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

        return out.squeeze(), p1, p2


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

        last.append(SpectralNorm(nn.Conv2d(curr_dim, curr_dim, 4)))
        last.append(nn.LeakyReLU(0.1))
        self.last = nn.Sequential(*last)

        if self.imsize == 128:
            self.attn1 = Self_Attn(self.imsize * 4, 'relu')
            self.attn2 = Self_Attn(self.imsize * 8, 'relu')
        if self.imsize == 64:
            self.attn1 = Self_Attn(256, 'relu')
            self.attn2 = Self_Attn(512, 'relu')

        # For inference z
        self.infer_z = nn.Sequential(
            SpectralNorm(nn.Conv2d(z_dim, 512, 1)),
            nn.LeakyReLU(0.1),
            SpectralNorm(nn.Conv2d(512, 512, 1)),
            nn.LeakyReLU(0.1),
        )

        # For inference joint x,z
        self.infer_joint = nn.Sequential(
            SpectralNorm(nn.Conv2d(1024 if self.imsize == 64 else 1536, 1024, 1)),
            nn.LeakyReLU(0.1),
            SpectralNorm(nn.Conv2d(1024, 1024, 1)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, z):
        # Inference x
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        if self.imsize == 64:
            x3, p1 = self.attn1(x)
            out = self.l4(x3)
            x4, p2 = self.attn2(out)
        if self.imsize == 128:
            x = self.l4(x)
            x3, p1 = self.attn1(x)
            out = self.l5(x3)
            x4, p2 = self.attn2(out)

        x5 = self.last(x4)

        # Inference z
        z = z.view(z.size(0), z.size(1), 1, 1)
        out_z = self.infer_z(z)

        # Inference joint
        out = self.infer_joint(torch.cat([x5, out_z], dim=1))
        return out.squeeze(), x5, x4, x3, out_z, p2, p1


class AlphaGan(nn.Module):
    def __init__(self, device, batch_normalisation=True, spectral_normalisation=True,
                 soft_labels=True, dlr=0.00005, gelr=0.001, soft_delta=0.1, z_dim=100,
                 adv_loss='hinge', masked_loss_on_val=True, image_size=(512, 512), *args, **kwargs):
        super(AlphaGan, self).__init__()

        self.hyper_parameters = locals()
        self.hyper_parameters.pop('self')
        self.device = device
        self.d_z = z_dim
        self.adv_loss = adv_loss

        self.generator = Generator(image_size=image_size[0], z_dim=self.d_z)
        self.discriminator = Discriminator(image_size=image_size[0], z_dim=self.d_z)
        self.encoder = Encoder(image_size=image_size[0], z_dim=self.d_z)
        self.hyper_parameters['discriminator'] = self.discriminator.hyper_parameters
        self.hyper_parameters['generator'] = self.generator.hyper_parameters
        self.hyper_parameters['encoder'] = self.encoder.hyper_parameters

        # Create batch of latent vectors that we will use to visualize the progression of the generator
        self.fixed_noise = torch.randn(32, self.d_z, device=self.device)

        # Losses
        # self.inner_loss = nn.BCELoss()
        self.masked_loss_on_val = masked_loss_on_val
        self.outer_loss = MaskedMSELoss(reduction='none') if self.masked_loss_on_val else nn.MSELoss(reduction='none')

        # Optimizers for discriminator and generator
        self.ge_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, itertools.chain(self.generator.parameters(), self.encoder.parameters())),
            lr=gelr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()),
                                            lr=dlr, betas=(0.5, 0.999))

        # Placeholders for losses of disriminator and generator
        self.ge_loss = None
        self.d_loss = None

    def to(self, *args, **kwargs):
        self.generator.to(*args, **kwargs)
        self.discriminator.to(*args, **kwargs)
        self.encoder.to(*args, **kwargs)
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
        # model_summary, trainable_paramsD = summary(self.discriminator, input_size=((1, *image_resolution), self.d_z),
        #                                            device=self.device)
        print('Encoder:')
        model_summary, trainable_paramsE = summary(self.encoder, input_size=(1, *image_resolution),
                                                   device=self.device)
        return trainable_paramsG + trainable_paramsE

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

        self.discriminator.zero_grad()
        self.generator.zero_grad()
        self.encoder.zero_grad()

        # Format input batch
        real_images = Variable(batch_data['image']).to(self.device)  # Real images
        fake_z = Variable(torch.randn(real_images.size(0), self.d_z)).to(self.device)  # Noise for generator

        noise1 = torch.Tensor(real_images.size()).normal_(0, 0.01 * (epoch + 1 - num_epochs) / (
                    epoch + 1)).cuda()  # Noise for real images
        noise2 = torch.Tensor(real_images.size()).normal_(0, 0.01 * (epoch + 1 - num_epochs) / (
                    epoch + 1)).cuda()  # Noise for fake images

        real_z, _, _ = self.encoder(real_images)  # Encoding real images

        fake_images, gf1, gf2 = self.generator(fake_z)  # Generating fake images

        dr, dr5, dr4, dr3, drz, dra2, dra1 = self.discriminator(real_images + noise1, real_z)
        df, df5, df4, df3, dfz, dfa2, dfa1 = self.discriminator(fake_images + noise2, fake_z)

        # Compute loss with real and fake images
        # dr1, dr2, df1, df2, gf1, gf2 are attention scores
        if self.adv_loss == 'wgan-gp':
            d_loss_real = - torch.mean(dr)
            d_loss_fake = df.mean()
            g_loss_fake = - df.mean()
            e_loss_real = - dr.mean()
        elif self.adv_loss == 'hinge1':
            d_loss_real = torch.nn.ReLU()(1.0 - dr).mean()
            d_loss_fake = torch.nn.ReLU()(1.0 + df).mean()
            g_loss_fake = - df.mean()
            e_loss_real = - dr.mean()
        elif self.adv_loss == 'hinge':
            d_loss_real = - torch.log(dr + 1e-10).mean()
            d_loss_fake = - torch.log(1.0 - df + 1e-10).mean()
            g_loss_fake = - torch.log(df + 1e-10).mean()
            e_loss_real = - torch.log(1.0 - dr + 1e-10).mean()
        elif self.adv_loss == 'inverse':
            d_loss_real = - torch.log(1.0 - dr + 1e-10).mean()
            d_loss_fake = - torch.log(df + 1e-10).mean()
            g_loss_fake = - torch.log(1.0 - df + 1e-10).mean()
            e_loss_real = - torch.log(dr + 1e-10).mean()

        # ================== Train D ================== #
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward(retain_graph=True)
        self.d_optimizer.step()

        if self.adv_loss == 'wgan-gp':
            # Compute gradient penalty
            alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
            interpolated = Variable(alpha * real_images.data + (1 - alpha) * fake_images.data, requires_grad=True)
            out, _, _ = self.discriminator(interpolated)

            grad = torch.autograd.grad(outputs=out,
                                       inputs=interpolated,
                                       grad_outputs=torch.ones(out.size()).cuda(),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]

            grad = grad.view(grad.size(0), -1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm - 1) ** 2)

            # Backward + Optimize
            d_loss = self.lambda_gp * d_loss_gp

            d_loss.backward()
            self.d_optimizer.step()

        # ================== Train G and E ================== #
        ge_loss = g_loss_fake + e_loss_real
        ge_loss.backward()
        self.ge_optimizer.step()

        self.ge_loss = ge_loss.data
        self.d_loss = d_loss.data
        return {'generator-encoder loss': float(self.ge_loss),
                'discriminator loss': float(self.d_loss)}

    def visualize_generator(self, epoch, *args, **kwargs):
        # Check how the generator is doing by saving G's output on fixed_noise
        path = TMP_IMAGES_DIR
        # Evaluation mode
        self.generator.eval()

        with torch.no_grad():
            fake = self.generator(self.fixed_noise)[0].detach().cpu()
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
        opt_threshold = val_metrics['optimal mse threshold'] if val_metrics is not None else None

        # Evaluation mode
        self.generator.eval()
        self.encoder.eval()
        self.discriminator.eval()
        with torch.no_grad():
            scores_mse = []
            scores_proba = []

            true_labels = []
            for batch_data in tqdm(loader, desc=type, total=len(loader)):
                # Format input batch
                inp = batch_data['image'].to(self.device)
                mask = batch_data['mask'].to(self.device)

                # Forward pass
                real_z, _, _ = self.encoder(inp)
                if len(real_z.size()) == 1:
                    real_z = real_z.view(1, real_z.size(0))
                reconstructed_img, _, _ = self.generator(real_z)

                loss = self.outer_loss(reconstructed_img, inp, mask) if self.masked_loss_on_val \
                    else self.outer_loss(reconstructed_img, inp)

                # Scores, based on output of discriminator - Higher score must correspond to positive labeled images
                proba = self.discriminator(inp, real_z)[0].to('cpu').numpy().reshape(-1)

                # Scores, based on MSE - higher MSE correspond to abnormal image
                if self.masked_loss_on_val:
                    sum_loss = loss.to('cpu').numpy().sum(axis=(1, 2, 3))
                    sum_mask = mask.to('cpu').numpy().sum(axis=(1, 2, 3))
                    score = sum_loss / sum_mask
                else:
                    score = loss.to('cpu').numpy().mean(axis=(1, 2, 3))

                scores_mse.extend(score)
                scores_proba.extend(proba)
                true_labels.extend(batch_data['label'].numpy())

            scores_mse = np.array(scores_mse)
            scores_proba = np.array(scores_proba)
            true_labels = np.array(true_labels)

            # ROC-AUC
            roc_auc_mse = roc_auc_score(true_labels, scores_mse)
            roc_auc_proba = roc_auc_score(true_labels, scores_proba)
            # Mean discriminator proba on validation batch
            mse = scores_mse.mean()
            proba = scores_proba.mean()
            # F1-score & optimal threshold
            if opt_threshold is None:  # validation
                precision, recall, thresholds = precision_recall_curve(y_true=true_labels, probas_pred=scores_mse)
                f1_scores = (2 * precision * recall / (precision + recall))
                f1 = np.nanmax(f1_scores)
                opt_threshold = thresholds[np.argmax(f1_scores)]
            else:  # testing
                y_pred = (scores_mse > opt_threshold).astype(int)
                f1 = f1_score(y_true=true_labels, y_pred=y_pred)

            print(f'ROC-AUC MSE on {type}: {roc_auc_mse}')
            print(f'ROC-AUC discriminator proba on {type}: {roc_auc_proba}')
            print(f'MSE on {type}: {mse}')
            print(f'F1-score on {type}: {f1}. Optimal threshold on {type}: {opt_threshold}')

            metrics = {"d_loss_train": float(self.d_loss),
                       "ge_loss_train": float(self.ge_loss),
                       "roc-auc_mse": roc_auc_mse,
                       "roc-auc_proba": roc_auc_proba,
                       "mse": mse,
                       "f1-score": f1,
                       "optimal mse threshold": opt_threshold}

            if log_to_mlflow:
                for (metric, value) in metrics.items():
                    mlflow.log_metric(metric, value)

            return metrics

    def save_to_mlflow(self):
        mlflow.pytorch.log_model(self.discriminator, f'{self.discriminator.__class__.__name__}')
        mlflow.pytorch.log_model(self.encoder, f'{self.encoder.__class__.__name__}')
        mlflow.pytorch.log_model(self.generator, f'{self.generator.__class__.__name__}')
