import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm
import mlflow.pytorch

from src.data import TrainValTestSplitter, MURASubset
from src.data.transforms import GrayScale, Padding, Resize, HistEqualisation, MinMaxNormalization, ToTensor
from src.models.vaetorch import VAE
from src.models.torchsummary import summary
from src import XR_HAND_CROPPED_PATH, MODELS_DIR, MLFLOW_TRACKING_URI, XR_HAND_PATH

# Ignoring numpy warnings and setting seeds
np.seterr(divide='ignore', invalid='ignore')
torch.manual_seed(42)

# General Setup
model_class = VAE
device = "cuda" if torch.cuda.is_available() else "cpu"
num_workers = 6

# Mlflow settings
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(model_class.__name__)

# Mlflow parameters
run_params = {
    'batch_size': 64,
    'image_resolution': (512, 512),
    'num_epochs': 100,
    'batch_normalisation': True,
    'pipeline': {
        'hist_equalisation': True,
        'cropped': True,
    }
}

# Initialization
data_path = XR_HAND_CROPPED_PATH if run_params['pipeline']['cropped'] else XR_HAND_PATH

print(f'\nDATA SPLIT:')
splitter = TrainValTestSplitter(path_to_data=data_path)

composed_transforms = Compose([GrayScale(),
                               HistEqualisation(active=run_params['pipeline']['hist_equalisation']),
                               Padding(),
                               Resize(run_params['image_resolution']),
                               MinMaxNormalization(),
                               ToTensor()])

train = MURASubset(filenames=splitter.data_train.path, patients=splitter.data_train.patient,
                   transform=composed_transforms)
validation = MURASubset(filenames=splitter.data_val.path, true_labels=splitter.data_val.label,
                        patients=splitter.data_val.patient, transform=composed_transforms)
test = MURASubset(filenames=splitter.data_test.path, true_labels=splitter.data_test.label,
                  patients=splitter.data_test.patient, transform=composed_transforms)

train_loader = DataLoader(train, batch_size=run_params['batch_size'], shuffle=True, num_workers=num_workers)
val_loader = DataLoader(validation, batch_size=run_params['batch_size'], shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test, batch_size=run_params['batch_size'], shuffle=True, num_workers=num_workers)

model = model_class(device=device).to(device)
# model = torch.load(f'{MODELS_DIR}/{current_model.__name__}.pt')
# model.eval().to(device)
print(f'\nMODEL ARCHITECTURE:')
model_summary, trainable_params = summary(model, input_size=(1, *run_params['image_resolution']), device=device)
run_params['trainable_params'] = trainable_params

# Logging
print(f'\nRUN PARAMETERS: {run_params}')
for (param, value) in run_params.items():
    mlflow.log_param(param, value)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
val_metrics = None
for epoch in range(run_params['num_epochs']):

    print('===========Epoch [{}/{}]============'.format(epoch + 1, run_params['num_epochs']))

    for batch_data in tqdm(train_loader, desc='Training', total=len(train_loader)):
        model.train()
        inp = Variable(batch_data['image']).to(device)

        # forward pass
        output, mu, logvar = model(inp)
        loss = VAE.loss(output, inp, mu, logvar)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # log
    print(f'Loss on last train batch: {loss.data}')

    # validation
    val_metrics = model.evaluate(val_loader, 'validation', device, log_to_mlflow=True)

    # forward pass for the random validation image
    index = np.random.randint(0, len(validation), 1)[0]
    model.forward_and_save_one_image(validation[index]['image'].unsqueeze(0), validation[index]['label'], epoch, device)

print('=========Training ended==========')

# Test performance
model.evaluate(test_loader, 'test', device, log_to_mlflow=True, opt_threshold=val_metrics['optimal mse threshold'])

# Saving
mlflow.pytorch.log_model(model, 'current_model.__name__')
torch.save(model, f'{MODELS_DIR}/{model_class.__name__}.pth')
