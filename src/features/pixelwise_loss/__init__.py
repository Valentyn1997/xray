import torch
from tqdm import tqdm


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
                    output, mu, var = self(inp)
                    # calculate loss per pixel
                    loss = self.loss_function(output, inp, mu, var, reduction='none')
                    loss = loss.numpy()

                # append values to list
                pixelwise_loss.extend(loss)
                true_labels.extend(batch_data['label'])
                patient.extend(batch_data['patient'])
                path.extend(batch_data['filename'])

            # create dictionary with results
            out = {'loss': pixelwise_loss, 'label': true_labels, 'patient': patient, 'path': path}
            return out
