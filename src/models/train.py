import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm
import mlflow.pytorch
import imgaug.augmenters as iaa
from pprint import pprint

from src.data import TrainValTestSplitter, MURASubset
from src.data.transforms import GrayScale, Padding, Resize, HistEqualisation, MinMaxNormalization, ToTensor
from src.features.augmentation import Augmentation
from src.models import BottleneckAutoencoder
from src.models.torchsummary import summary
from src import XR_HAND_CROPPED_PATH, MODELS_DIR, MLFLOW_TRACKING_URI, XR_HAND_PATH
from src.utils import query_yes_no


# ---------------------------------------  Parameters setups ---------------------------------------
# Ignoring numpy warnings and setting seeds
np.seterr(divide='ignore', invalid='ignore')
torch.manual_seed(42)

model_class = BottleneckAutoencoder
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
num_workers = 6
log_to_mlflow = query_yes_no('Log this run to mlflow?', 'no')

# Mlflow settings
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(model_class.__name__)

# Mlflow parameters
run_params = {
    'batch_size': 64,
    'image_resolution': (512, 512),
    'num_epochs': 500,
    'batch_normalisation': True,
    'pipeline': {
        'hist_equalisation': False,
        'cropped': True,
    }
}

# Data source
data_path = XR_HAND_CROPPED_PATH if run_params['pipeline']['cropped'] else XR_HAND_PATH

# Augmentation
augmentation_seq = iaa.Sequential([iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                                   iaa.Flipud(0.5),  # vertically flip 50% of all images
                                   iaa.Sometimes(0.5, iaa.Affine(fit_output=True,  # not crop corners by rotation
                                                                 rotate=(-45, 45),  # rotate by -45 to +45 degrees
                                                                 order=[0, 1])),
                                   # use nearest neighbour or bilinear interpolation (fast)
                                   ])
run_params['augmentation'] = augmentation_seq.get_all_children()


# ----------------------------- Data, preprocessing and model initialization ------------------------------------
# Preprocessing pipeline
composed_transforms = Compose([GrayScale(),
                               Augmentation(augmentation_seq),
                               Padding(max_shape=(750, 750)),  # max_shape - max size of image after augmentation
                               Resize(run_params['image_resolution']),
                               HistEqualisation(active=run_params['pipeline']['hist_equalisation']),
                               MinMaxNormalization(),
                               ToTensor()])

# Dataset loaders
print(f'\nDATA SPLIT:')
splitter = TrainValTestSplitter(path_to_data=data_path)
train = MURASubset(filenames=splitter.data_train.path, patients=splitter.data_train.patient,
                   transform=composed_transforms, true_labels=np.zeros(len(splitter.data_train.path)))
validation = MURASubset(filenames=splitter.data_val.path, true_labels=splitter.data_val.label,
                        patients=splitter.data_val.patient, transform=composed_transforms)
test = MURASubset(filenames=splitter.data_test.path, true_labels=splitter.data_test.label,
                  patients=splitter.data_test.patient, transform=composed_transforms)

train_loader = DataLoader(train, batch_size=run_params['batch_size'], shuffle=True, num_workers=num_workers)
val_loader = DataLoader(validation, batch_size=run_params['batch_size'], shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test, batch_size=run_params['batch_size'], shuffle=True, num_workers=num_workers)

# Model initialization
model = model_class(use_batchnorm=run_params['batch_normalisation']).to(device)
# model = torch.load(f'{MODELS_DIR}/{current_model.__name__}.pt')
# model.eval().to(device)
print(f'\nMODEL ARCHITECTURE:')
model_summary, trainable_params = summary(model, input_size=(1, *run_params['image_resolution']), device=device)
run_params['trainable_params'] = trainable_params

# Losses
inner_loss = nn.MSELoss()
outer_loss = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# -------------------------------- Logging ------------------------------------
# Logging
print('\nRUN PARAMETERS:')
pprint(run_params, width=-1)

if log_to_mlflow:
    for (param, value) in run_params.items():
        mlflow.log_param(param, value)


# -------------------------------- Training and evaluation -----------------------------------
val_metrics = None
for epoch in range(run_params['num_epochs']):

    print('===========Epoch [{}/{}]============'.format(epoch + 1, run_params['num_epochs']))

    for batch_data in tqdm(train_loader, desc='Training', total=len(train_loader)):
        model.train()
        inp = Variable(batch_data['image']).to(device)

        # forward pass
        output = model(inp)
        loss = inner_loss(output, inp)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # model.forward_and_save_one_image(train[0]['image'].unsqueeze(0), train[0]['label'], epoch, device)

    # log
    print(f'Loss on last train batch: {loss.data}')

    # validation
    val_metrics = model.evaluate(val_loader, 'validation', outer_loss, device, log_to_mlflow=log_to_mlflow)

    # forward pass for the random validation image
    index = np.random.randint(0, len(validation), 1)[0]
    model.forward_and_save_one_image(validation[index]['image'].unsqueeze(0), validation[index]['label'], epoch, device)

print('=========Training ended==========')

# Test performance
model.evaluate(test_loader, 'test', outer_loss, device, log_to_mlflow=log_to_mlflow,
               opt_threshold=val_metrics['optimal mse threshold'])

# Saving
mlflow.pytorch.log_model(model, 'current_model.__name__')
torch.save(model, f'{MODELS_DIR}/{model_class.__name__}.pth')
