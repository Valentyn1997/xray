from pprint import pprint

import imgaug.augmenters as iaa
import mlflow.pytorch
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from src import MLFLOW_TRACKING_URI, DATA_PATH
from src.data import TrainValTestSplitter, MURASubset
from src.data.transforms import GrayScale, Resize, HistEqualisation, MinMaxNormalization, ToTensor
from src.data.transforms import OtsuFilter, AdaptiveHistogramEqualization
from src.features.augmentation import Augmentation
from src.models.alphagan import AlphaGan
from src.models.autoencoders import BottleneckAutoencoder, BaselineAutoencoder, SkipConnection, Bottleneck
from src.models.gans import DCGAN
from src.models.sagan import SAGAN
from src.models.vaetorch import VAE
from src.utils import query_yes_no, save_model

# ---------------------------------------  Parameters setups ---------------------------------------
# Ignoring numpy warnings and setting seeds
np.seterr(divide='ignore', invalid='ignore')
torch.manual_seed(42)

# set model type
model_class = AlphaGan
# set device
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
# set number of cpu kernels for data processing
num_workers = 12
# set
log_to_mlflow = query_yes_no('Log this run to mlflow?', 'no')

# Mlflow settings
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(model_class.__name__)

# Mlflow parameters
if model_class in [AlphaGan, SAGAN, DCGAN, VAE]:
    run_params = {
        'batch_size': 16,
        'image_resolution': (128, 128),
        'num_epochs': 300,
        'batch_normalisation': True,
        'spectral_normalisation': True,
        'pipeline': {
            'hist_equalisation': False,
            'otsu_filter': False,
            'adaptive_hist_equilization': False,
            'data_source': 'XR_HAND',
        },
        'masked_loss_on_val': True,
        'masked_loss_on_train': True,
        'soft_labels': True,
        'glr': 0.001,
        'dlr': 0.00001,
        'z_dim': 200,
        'lr': 0.0001,
        'soft_delta': 0.05,
        'adv_loss': 'hinge',
    }
elif model_class in [BaselineAutoencoder, BottleneckAutoencoder, SkipConnection, Bottleneck]:
    run_params = {
        'batch_size': 32,
        'image_resolution': (512, 512),
        'num_epochs': 500,
        'batch_normalisation': True,
        'pipeline': {
            'hist_equalisation': True,
            'otsu_filter': False,
            'adaptive_hist_equilization': False,
            'data_source': 'XR_HAND_PHOTOSHOP',
        },
        'masked_loss_on_val': True,
        'masked_loss_on_train': True,
        'lr': 0.0001,
    }

# Augmentation
augmentation_seq = iaa.Sequential([iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                                   iaa.Flipud(0.5),  # vertically flip 50% of all images,
                                   # iaa.Sometimes(0.5, iaa.Affine(fit_output=True,  # not crop corners by rotation
                                   #                               rotate=(-20, 20),  # rotate by -45 to +45 degrees
                                   #                               order=[0, 1])),
                                   #  iaa.Sometimes(0.5, iaa.Multiply((0.8, 1.2))),
                                   #  iaa.Sometimes(0.5, iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})),
                                   #  iaa.Sometimes(0.5, iaa.Affine(rotate=(-20,20))),
                                   # use nearest neighbour or bilinear interpolation (fast)
                                   # iaa.Resize(),
                                   iaa.PadToFixedSize(*run_params['image_resolution'], position='center')
                                   ])
run_params['augmentation'] = augmentation_seq.get_all_children()

# ----------------------------- Data, preprocessing and model initialization ------------------------------------
# Preprocessing pipeline
# transformation for training
composed_transforms = Compose([GrayScale(),
                               HistEqualisation(active=run_params['pipeline']['hist_equalisation']),
                               OtsuFilter(active=run_params['pipeline']['otsu_filter']),
                               AdaptiveHistogramEqualization(
                                   active=run_params['pipeline']['adaptive_hist_equilization']),
                               Resize(run_params['image_resolution'], keep_aspect_ratio=True),
                               Augmentation(augmentation_seq),
                               # Padding(max_shape=run_params['image_resolution']),
                               # max_shape - max size of image after augmentation
                               MinMaxNormalization(),
                               ToTensor()])

# transformation for validation and test
composed_transforms_val = Compose([GrayScale(),
                                   HistEqualisation(active=run_params['pipeline']['hist_equalisation']),
                                   OtsuFilter(active=run_params['pipeline']['otsu_filter']),
                                   AdaptiveHistogramEqualization(
                                       active=run_params['pipeline']['adaptive_hist_equilization']),
                                   Resize(run_params['image_resolution'], keep_aspect_ratio=True),
                                   Augmentation(
                                       iaa.Sequential([iaa.PadToFixedSize(*run_params['image_resolution'], position='center')])),
                                   # Padding(max_shape=run_params['image_resolution']),
                                   # max_shape - max size of image after augmentation
                                   MinMaxNormalization(),
                                   ToTensor()])

# Dataset loaders
print(f'\nDATA SPLIT:')
data_path = f'{DATA_PATH}/{run_params["pipeline"]["data_source"]}'
print(data_path)
# Split data
splitter = TrainValTestSplitter(path_to_data=data_path)
train = MURASubset(filenames=splitter.data_train.path, patients=splitter.data_train.patient,
                   transform=composed_transforms, true_labels=np.zeros(len(splitter.data_train.path)))
validation = MURASubset(filenames=splitter.data_val.path, true_labels=splitter.data_val.label,
                        patients=splitter.data_val.patient, transform=composed_transforms_val)
test = MURASubset(filenames=splitter.data_test.path, true_labels=splitter.data_test.label,
                  patients=splitter.data_test.patient, transform=composed_transforms_val)

train_loader = DataLoader(train, batch_size=run_params['batch_size'], shuffle=True, num_workers=num_workers)
val_loader = DataLoader(validation, batch_size=run_params['batch_size'], shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test, batch_size=run_params['batch_size'], shuffle=True, num_workers=num_workers)

# Model initialization
model = model_class(device=device,
                    image_size=run_params['image_resolution'],
                    **run_params).to(device)
# model = torch.load(f'{MODELS_DIR}/{model_class.__name__}.pth')
# model.eval().to(device)
print(f'\nMODEL ARCHITECTURE:')
trainable_params = model.summary(image_resolution=run_params['image_resolution'])
run_params['trainable_params'] = trainable_params
run_params['other_hyperparams'] = model.hyper_parameters

# -------------------------------- Logging ------------------------------------
# Logging
print('\nRUN PARAMETERS:')
pprint(run_params, width=-1)

if log_to_mlflow:
    for (param, value) in run_params.items():
        mlflow.log_param(param, value)

# -------------------------------- Training and evaluation -----------------------------------
val_metrics = None
for epoch in range(1, run_params['num_epochs'] + 1):

    print('===========Epoch [{}/{}]============'.format(epoch, run_params['num_epochs']))

    for batch_data in tqdm(train_loader, desc='Training', total=len(train_loader)):
        losses_dict = model.train_on_batch(batch_data, epoch=epoch, num_epochs=run_params['num_epochs'])

    # log
    print(f'Loss on last train batch: {losses_dict}')
    if log_to_mlflow:
        for loss in losses_dict:
            mlflow.log_metric(f'train_{loss}', losses_dict[loss])

    # validation
    val_metrics = model.evaluate(val_loader, 'validation', log_to_mlflow=log_to_mlflow)

    if model_class in [BottleneckAutoencoder, BaselineAutoencoder, VAE, SkipConnection, Bottleneck]:
        # forward pass for the random validation image
        index = np.random.randint(0, len(validation), 1)[0]
        model.forward_and_save_one_image(validation[index]['image'].unsqueeze(0), validation[index]['label'], epoch)
    elif model_class in [DCGAN, SAGAN] and epoch % 3 == 0:
        # evaluate performance of generator
        model.visualize_generator(epoch)
    elif model_class in [AlphaGan] and epoch % 2 == 0:
        # evaluate performance of generator
        model.visualize_generator(epoch)
        # evaluate performance of encoder / generator
        index = np.random.randint(0, len(validation), 1)[0]
        model.forward_and_save_one_image(validation[index]['image'].unsqueeze(0), validation[index]['label'], epoch)
    if epoch % 50 == 0:
        save_model(model, log_to_mlflow=log_to_mlflow)

print('=========Training ended==========')

# Test performance
model.evaluate(test_loader, 'test', log_to_mlflow=log_to_mlflow, val_metrics=val_metrics)

# Saving
save_model(model, log_to_mlflow=log_to_mlflow)
