from src.features.pixelwise_loss import PixelwiseLoss
import imgaug.augmenters as iaa
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import torch.nn as nn
from src.models.autoencoders import MaskedMSELoss
from src import DATA_PATH
from src.data import TrainValTestSplitter, MURASubset
from src.data.transforms import *
from src.features.augmentation import Augmentation

num_workers = 7
log_to_mlflow = False
device = "cpu"

# EXAMPLE RUN

# Mlflow parameters
run_params = {
    'batch_size': 32,
    'image_resolution': (512, 512),
    'num_epochs': 1000,
    'batch_normalisation': True,
    'pipeline': {
        'hist_equalisation': False,
        'data_source': 'XR_HAND_PHOTOSHOP',
    },
    'masked_loss_on_val': True,
    'masked_loss_on_train': True,
    'soft_labels': True,
    'glr': 0.001,
    'dlr': 0.00005,
    'z_dim': 1000,
    'lr': 0.0001
}

# Preprocessing pipeline

composed_transforms_val = Compose([GrayScale(),
                                   HistEqualisation(active=run_params['pipeline']['hist_equalisation']),
                                   Resize(run_params['image_resolution'], keep_aspect_ratio=True),
                                   Augmentation(iaa.Sequential([iaa.PadToFixedSize(512, 512, position='center')])),
                                   # Padding(max_shape=run_params['image_resolution']),
                                   # max_shape - max size of image after augmentation
                                   MinMaxNormalization(),
                                   ToTensor()])

# get data

data_path = f'{DATA_PATH}/{run_params["pipeline"]["data_source"]}'
print(data_path)
splitter = TrainValTestSplitter(path_to_data=data_path)

validation = MURASubset(filenames=splitter.data_val.path, true_labels=splitter.data_val.label,
                        patients=splitter.data_val.patient, transform=composed_transforms_val)

val_loader = DataLoader(validation, batch_size=run_params['batch_size'], shuffle=True, num_workers=num_workers)

# get model (change path to path to a trained model

# model = torch.load("/home/ubuntu/mlruns/1/18d0f8aa21c44bca9c7414754b6f8552/artifacts/BaselineAutoencoder/data/model.pth", map_location=device)
model = torch.load("/home/ubuntu/mlruns/1/5ca7f67c33674926a00590752c877fe5/artifacts/BaselineAutoencoder.pth", map_location=device)
# model = torch.load("/home/diana/xray/models/VAE.pth", map_location=device)


# set loss function
outer_loss = nn.MSELoss(reduction='none')
model.eval().to(device)

# evaluation = PixelwiseLoss(model=model, model_class='CAE', device=device, loss_function=outer_loss, masked_loss_on_val=False)
evaluation = PixelwiseLoss(model=model, model_class='CAE', device=device, loss_function=MaskedMSELoss(reduction="none"), masked_loss_on_val=True)
loss_dict = evaluation.get_loss(data=val_loader)
