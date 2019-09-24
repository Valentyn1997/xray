import random
from pprint import pprint

import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from src import MLFLOW_TRACKING_URI, DATA_PATH
from src.data import TrainValTestSplitter, MURASubset
from src.data.transforms import GrayScale, Resize, HistEqualisation, MinMaxNormalization, ToTensor
from src.data.transforms import OtsuFilter, AdaptiveHistogramEqualization, Padding
from src.features.augmentation import Augmentation
from src.models.alphagan import AlphaGan
from src.models.autoencoders import BottleneckAutoencoder, BaselineAutoencoder, SkipConnection, Bottleneck
from src.models.gans import DCGAN
from src.models.run_params import COMMON_PARAMS, MODEL_SPECIFIC_PARAMS
from src.models.sagan import SAGAN
from src.models.vaetorch import VAE
from src.utils import query_yes_no, save_model

# ---------------------------------------  Parameters setups ---------------------------------------
# set model type
model_class = VAE

# set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device: {device}')
torch.cuda.empty_cache()
# torch.cuda.set_device(1)
# device = 'cpu'
# set number of cpu kernels for data processing
num_workers = 12
log_to_mlflow = query_yes_no('Log this run to mlflow?', 'no')
remote_run = query_yes_no('Is this run remote?', 'no')

# Mlflow settings
if log_to_mlflow:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(model_class.__name__ + 'New')
    mlflow.start_run()

# Ignoring numpy warnings
np.seterr(divide='ignore', invalid='ignore')

run_params = {**COMMON_PARAMS, **MODEL_SPECIFIC_PARAMS[model_class.__name__]}
test_metrics = []

for random_seed in run_params['random_seed']:

    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    loader_init_fn = lambda worker_id: np.random.seed(random_seed + worker_id)

    # ----------------------------- Data, preprocessing and model initialization ------------------------------------
    # Preprocessing pipeline
    # transformation for training
    composed_transforms = Compose([GrayScale(),
                                   HistEqualisation(active=run_params['pipeline']['hist_equalisation']),
                                   OtsuFilter(active=run_params['pipeline']['otsu_filter']),
                                   AdaptiveHistogramEqualization(
                                       active=run_params['pipeline']['adaptive_hist_equilization']),
                                   Augmentation(run_params['augmentation'], random_state=random_seed),
                                   Resize(run_params['image_resolution'], keep_aspect_ratio=True),
                                   Padding(max_shape=run_params['image_resolution']),
                                   # max_shape - max size of image after augmentation
                                   MinMaxNormalization(*run_params['pipeline']['normalisation']),
                                   ToTensor()])

    # transformation for validation and test
    composed_transforms_val = Compose([GrayScale(),
                                       HistEqualisation(active=run_params['pipeline']['hist_equalisation']),
                                       OtsuFilter(active=run_params['pipeline']['otsu_filter']),
                                       AdaptiveHistogramEqualization(
                                           active=run_params['pipeline']['adaptive_hist_equilization']),
                                       Resize(run_params['image_resolution'], keep_aspect_ratio=True),
                                       Padding(max_shape=run_params['image_resolution']),
                                       # max_shape - max size of image after augmentation
                                       MinMaxNormalization(*run_params['pipeline']['normalisation']),
                                       ToTensor()])

    # Dataset loaders
    print(f'\nDATA SPLIT:')
    data_path = f'{DATA_PATH}/{run_params["data_source"]}'
    print(data_path)

    # Split data
    splitter = TrainValTestSplitter(path_to_data=data_path, random_state=random_seed)
    train = MURASubset(filenames=splitter.data_train.path, patients=splitter.data_train.patient,
                       transform=composed_transforms, true_labels=np.zeros(len(splitter.data_train.path)))
    validation = MURASubset(filenames=splitter.data_val.path, true_labels=splitter.data_val.label,
                            patients=splitter.data_val.patient, transform=composed_transforms_val)
    test = MURASubset(filenames=splitter.data_test.path, true_labels=splitter.data_test.label,
                      patients=splitter.data_test.patient, transform=composed_transforms_val)

    train_loader = DataLoader(train, batch_size=run_params['batch_size'], shuffle=True, num_workers=num_workers,
                              worker_init_fn=loader_init_fn, drop_last=model_class in [DCGAN])
    val_loader = DataLoader(validation, batch_size=run_params['batch_size'], shuffle=True, num_workers=num_workers,
                            drop_last=model_class in [DCGAN])
    test_loader = DataLoader(test, batch_size=run_params['batch_size'], shuffle=True, num_workers=num_workers,
                             drop_last=model_class in [DCGAN])

    # Model initialization
    model = model_class(device=device, **run_params)
    model = model.to(device)

    # model = torch.load(f'{MODELS_DIR}/{model_class.__name__}.pth')
    # model.eval().to(device)
    print(f'\nMODEL ARCHITECTURE:')
    trainable_params = model.summary(image_resolution=run_params['image_resolution'])
    run_params['trainable_params'] = trainable_params
    run_params['other_hyperparams'] = model.hyper_parameters

    # Parallelism
    if torch.cuda.device_count() > 1:
        model.parallelize()

    # -------------------------------- Logging ------------------------------------
    # Logging
    print('\nRUN PARAMETERS:')
    pprint(run_params, width=-1)

    if log_to_mlflow:
        for (param, value) in run_params.items():
            if param == 'augmentation':
                mlflow.log_param(param, value.get_all_children())
            elif param != 'random_seed':
                mlflow.log_param(param, value)

        mlflow.start_run(nested=True)
        mlflow.log_param('random_seed', random_seed)

    # -------------------------------- Training and evaluation -----------------------------------
    val_metrics = None
    losses_dict = None
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
        val_metrics = model.evaluate(val_loader, log_to_mlflow=log_to_mlflow)

        # Vizualisations
        index = np.random.randint(0, len(validation), 1)[0]
        if model_class in [BottleneckAutoencoder, BaselineAutoencoder, VAE, SkipConnection,
                           Bottleneck] and epoch % 2 == 0:
            # forward pass for the random validation image
            model.forward_and_save_one_image(validation[index]['image'].unsqueeze(0),
                                             validation[index]['label'], epoch, to_mlflow=log_to_mlflow,
                                             is_remote=remote_run)
        elif model_class in [DCGAN] and epoch % 3 == 0:
            # evaluate performance of generator
            model.visualize_generator(epoch, to_mlflow=log_to_mlflow, is_remote=remote_run,
                                      vmin=run_params['pipeline']['normalisation'][0],
                                      vmax=run_params['pipeline']['normalisation'][1])
        elif model_class in [AlphaGan, SAGAN] and epoch % 2 == 0:
            # evaluate performance of generator
            model.visualize_generator(epoch, to_mlflow=log_to_mlflow, is_remote=remote_run,
                                      vmin=run_params['pipeline']['normalisation'][0],
                                      vmax=run_params['pipeline']['normalisation'][1])
            # evaluate performance of encoder / generator
            model.forward_and_save_one_image(validation[index]['image'].unsqueeze(0), validation[index]['label'], epoch,
                                             to_mlflow=log_to_mlflow, is_remote=remote_run,
                                             vmin=run_params['pipeline']['normalisation'][0],
                                             vmax=run_params['pipeline']['normalisation'][1])

        # Checkpoints
        if 'checkpoint_frequency' in run_params and epoch % run_params['checkpoint_frequency'] == 0:
            save_model(model, log_to_mlflow=log_to_mlflow, epoch=epoch, is_remote=remote_run)

    print('=========Training ended==========')

    # Test performance
    test_metric = model.evaluate(test_loader, log_to_mlflow=log_to_mlflow)
    test_metrics.append(test_metric)

    # Saving
    save_model(model, log_to_mlflow=log_to_mlflow, is_remote=remote_run)
    mlflow.end_run()

# Averaging metrics for different random-seeds
avg_metric = dict(pd.DataFrame(test_metrics).mean())
if log_to_mlflow:
    for (metric, value) in avg_metric.items():
        mlflow.log_metric(metric, value)

mlflow.end_run()
