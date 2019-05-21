import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from torchsummary import summary

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

from src.data import DataGenerator, TrainValTestSplitter
from src.models import BaselineAutoencoder

from src import mkdir_p

np.seterr(divide='ignore', invalid='ignore')

splitter = TrainValTestSplitter()
train_generator = DataGenerator(filenames=splitter.data_train.path[0:128],
                                batch_size=32,
                                dim=(64, 64))
val_generator = DataGenerator(filenames=splitter.data_val.path[0:64],
                              batch_size=32,
                              dim=(64, 64),
                              true_labels=splitter.data_val.label)

# Initialization
model = BaselineAutoencoder().cpu()
# model = torch.load('..\\..\\models\\baseline_autoencoder.pt')
# model.eval()

inner_loss = nn.MSELoss()
outer_loss = nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(summary(model,
              input_size=(train_generator.n_channels, *train_generator.dim)))
mkdir_p('tmp') #For saving intermediate pictures

# Training
num_epochs = 2
for epoch in range(num_epochs):

    print('===========Epoch [{}/{}]============'.format(epoch + 1, num_epochs))

    for batch in tqdm(range(len(train_generator)), desc='Training'):
        inp = Variable(torch.from_numpy(train_generator[batch]).float()).cpu()

        # forward pass
        output = model(inp)
        loss = inner_loss(output, inp)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # log
    print('Loss on last train batch:{:.4f}'.format(epoch + 1,
                                                   num_epochs, loss.data))

    # shuffle
    train_generator.on_epoch_end()

    # forward pass for the last train image
    with torch.no_grad():
        inp_image = train_generator[-1][-1:]
        inp = torch.from_numpy(inp_image).float()
        output = model(inp)
        output_img = output.numpy()[0][0]

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        ax[0].imshow(inp_image[0][0])
        ax[1].imshow(output_img)
        plt.savefig(f'tmp\\epoch{epoch}.png')

    # validation
    with torch.no_grad():
        losses = []
        for batch in tqdm(range(len(val_generator)), desc='Validation'):
            inp = Variable(torch.from_numpy(val_generator[batch]).float()).\
                cpu()

            # forward pass
            output = model(inp)
            losses.extend(outer_loss(output, inp).numpy().mean(axis=(1, 2, 3)))
        losses = np.array(losses)
        true_labels = val_generator.get_true_labels()

        # ROC-AUC
        print(f'ROC-AUC on val: {roc_auc_score(true_labels, losses)}')

        # MSE
        print(f'MSE on val: {losses.mean()}')

        # F1-score
        precision, recall, thresholds = \
            precision_recall_curve(y_true=true_labels, probas_pred=losses)
        f1_scores = (2 * precision * recall / (precision + recall))
        opt_treshold = thresholds[np.argmax(f1_scores)]
        print(f'F1-score: {np.max(f1_scores)}. '
              f'Optimal threshold: {opt_treshold}')

torch.save(model, '..\\..\\models\\baseline_autoencoder.pt')