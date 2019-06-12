import torch.nn as nn
from typing import List
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


summary(Generator().to('cpu'), input_size=(256, 1, 1), device='cpu')
summary(Discriminator().to('cpu'), input_size=(1, 512, 512), device='cpu')
