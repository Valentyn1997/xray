import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from torchsummary import summary
import mlflow.pytorch

import matplotlib.pyplot as plt

from src.data import DataGenerator, TrainValTestSplitter
from src.models import BaselineAutoencoder

from src import mkdir_p

np.seterr(divide='ignore', invalid='ignore')

# Parameters
torch.manual_seed(42)
batch_size = 32
image_resolution = (512, 512)
num_epochs = 4

# Initialization
splitter = TrainValTestSplitter()
train_generator = DataGenerator(filenames=splitter.data_train.path, batch_size=batch_size, dim=image_resolution)
val_generator = DataGenerator(filenames=splitter.data_val.path, batch_size=batch_size, dim=image_resolution,
                              true_labels=splitter.data_val.label)
test_generator = DataGenerator(filenames=splitter.data_test.path, batch_size=batch_size, dim=image_resolution,
                               true_labels=splitter.data_test.label)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Available device: {device}')
model = BaselineAutoencoder().to(device)
# model = torch.load('../../models/baseline_autoencoder.pt')
# model.eval().to(device)

inner_loss = nn.MSELoss()
outer_loss = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(summary(model, input_size=(train_generator.n_channels, *train_generator.dim), device='cuda'))
mkdir_p('tmp')  # For saving intermediate pictures

# Logging
mlflow.log_param("batch_size", batch_size)
mlflow.log_param("image_resolution", image_resolution)
mlflow.log_param("num_epochs", num_epochs)

# Training
for epoch in range(num_epochs):

    print('===========Epoch [{}/{}]============'.format(epoch + 1, num_epochs))

    for batch in tqdm(range(len(train_generator)), desc='Training'):
        inp = Variable(torch.from_numpy(train_generator[batch]).float()).to(device)

        # forward pass
        output = model(inp)
        loss = inner_loss(output, inp)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # log
    print(f'Loss on last train batch: {loss.data}')

    # shuffle
    train_generator.on_epoch_end()

    # forward pass for the last train image
    with torch.no_grad():
        inp_image = train_generator[-1][-1:]
        inp = torch.from_numpy(inp_image).float().to(device)
        output = model(inp)
        output_img = output.to('cpu').numpy()[0][0]

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        ax[0].imshow(inp_image[0][0], cmap='gray', vmin=0, vmax=1)
        ax[1].imshow(output_img, cmap='gray', vmin=0, vmax=1)
        plt.savefig(f'tmp/epoch{epoch}.png')
        plt.close(fig)

    # validation
    model.evaluate(val_generator, 'validation', outer_loss, device)

print('=========Training ended==========')

# Test performance
model.evaluate(test_generator, 'test', outer_loss, device, log_to_mlflow=True)

# Saving
mlflow.pytorch.log_model(model, "baseline_autoencoder")
torch.save(model, '../../models/baseline_autoencoder.pt')
