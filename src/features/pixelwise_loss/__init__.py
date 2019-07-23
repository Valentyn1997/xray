from os.path import basename, dirname
from src import TMP_IMAGES_DIR
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter


class PixelwiseLoss:

    def __init__(self, model, model_class, device, loss_function, masked_loss_on_val):
        """
        Initialize evluation of pixelwise loss

        :param model: trained pytorch model
        :param model_class: string, which model class: CAE, VAE
        :param device: string, which device usually cpu or cuda
        :param loss_function: function, which loss should be used example: nn.MSELoss
        :param masked_loss_on_val: bool, should masked loss should be used
        """

        self.model = model
        self.device = device
        self.loss_function = loss_function
        self.model_class = model_class
        self.masked_loss_on_val = masked_loss_on_val

    def get_loss(self, data):
        """
        Apply model on data and get pixelwise loss

        :param data: data to be evaluated, example from dataloader
        :return: dictionary with pixelwise loss, label, patient, filepath
        """

        # set model to evaluation, so no weights will be updated
        model = self.model
        model.eval()
        with torch.no_grad():
            # Initiate list to save results
            pixelwise_loss = []
            true_labels = []
            patient = []
            path = []

            # iterate over data with batch
            for batch_data in tqdm(data, desc='evaluation', total=len(data)):
                # load data from batch
                inp = batch_data['image'].to(self.device)
                mask = batch_data['mask'].to(self.device)

                if self.model_class == 'CAE':
                    # apply model on image
                    output = model(inp)
                    # calculate loss per pixel
                    loss = self.loss_function(output, inp, mask) if self.masked_loss_on_val else self.loss_function(
                        output, inp)
                    loss = loss.cpu().numpy()
                elif self.model_class == 'VAE':
                    # apply model on image
                    output, mu, var = model(inp)
                    # calculate loss per pixel
                    loss = self.loss_function(output, inp)
                    loss = loss.cpu().numpy()

                # get the first image from and save heatmap
                self.add_heatmap(inp[0].data[0, :, :], batch_data['label'].numpy()[0],
                                 batch_data['patient'].numpy()[0],
                                 loss[0][0, :, :],
                                 batch_data['filename'][0])

                # append values to list
                pixelwise_loss.extend(loss)
                true_labels.extend(batch_data['label'])
                patient.extend(batch_data['patient'])
                path.extend(batch_data['filename'])

            # create dictionary with results
            out = {'loss': pixelwise_loss, 'label': true_labels, 'patient': patient, 'path': path}
            return out

    def add_heatmap(self, inp_image, label, patient, loss, original_path, max_loss=0.002,
                    sigma=5, path=TMP_IMAGES_DIR, save=True, display=False):
        """
        Add heatmap layer on top of the image
        :param inp_image: imput image array
        :param: current patient
        :param label: true label
        :param loss: current loss from the model
        :param original_path: path of original image
        :param max_loss: max_loss for heatmap. Adjust this for different models
        :param sigma: gaussian blur parameter
        :param path: path to save
        :param save: flag to save an image with heatmap
        """

        loss = gaussian_filter(loss, sigma)

        mycmap = self._transparent_cmap(plt.cm.Reds)
        inp_image = inp_image.numpy()
        w, h = inp_image.shape
        y, x = np.mgrid[0:h, 0:w]
        # Plot image and overlay colormap
        fig, ax = plt.subplots(1, 1)
        ax.imshow(inp_image, cmap='gray')
        ax.set_title(label)
        ax.contourf(x, y, loss, 50, cmap=mycmap, vmin=0, vmax=max_loss)

        if save:
            image_name = basename(original_path)
            study_name = basename(dirname(original_path))
            name_to_save = f'{path}/heatmap_patient_{patient}_label{int(label)}_' + study_name + '_' + image_name
            plt.savefig(name_to_save)
        if display:
            plt.show()
            plt.close(fig)

    def _transparent_cmap(self, cmap, N=255):
        "Copy colormap and set alpha values"
        mycmap = cmap
        mycmap._init()
        mycmap._lut[:, -1] = np.linspace(0, 0.8, N + 4)
        return mycmap
