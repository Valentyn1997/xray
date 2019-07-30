import plotly.figure_factory as ff
import torch
from tqdm import tqdm
import numpy as np


class PlotLossLabel:
    """
    Class to plot CAE evaluation
    """
    def __init__(self, model, data, device, masked_loss_on_val, outer_loss, bin_size):
        """

        :param model: pretrained model
        :param data: data from data loader
        :param device: device e.g. "cpu"
        :param masked_loss_on_val: bool, True means calculate loss on mask
        :param outer_loss: function to calculate loss
        :param bin_size: step size for histogram
        """
        self.model = model
        self.data = data
        self.masked_loss_on_val = masked_loss_on_val
        self.outer_loss = outer_loss
        self.scores = []
        self.true_labels = []
        self.bin_size = bin_size
        self.device = device

    def evaluation_cae(self):
        """
        get metrics, currently only works for CAE
        :return: loss and label
        """

        # Evaluation mode
        self.model.eval()
        with torch.no_grad():

            for batch_data in tqdm(self.data, desc='type', total=len(self.data)):
                # Format input batch
                inp = batch_data['image'].to(self.device)
                mask = batch_data['mask'].to(self.device)

                # Forward pass
                output = self.model(inp)
                loss = self.outer_loss(output, inp, mask) if self.masked_loss_on_val else self.outer_loss(output, inp)

                # Scores, based on MSE - higher MSE correspond to abnormal image
                if self.masked_loss_on_val:
                    sum_loss = loss.to('cpu').numpy().sum(axis=(1, 2, 3))
                    sum_mask = mask.to('cpu').numpy().sum(axis=(1, 2, 3))
                    score = sum_loss / sum_mask
                else:
                    score = loss.to('cpu').numpy().mean(axis=(1, 2, 3))
                # saves score and label
                self.scores.extend(score)
                self.true_labels.extend(batch_data['label'].numpy())

            self.scores = np.array(self.scores)
            self.true_labels = np.array(self.true_labels)


    def get_plot(self):
        """
        generate the loss vs label plot
        :return: Loss vs label plot
        """

        # get scores and true labels for plot
        scores = self.scores
        true_labels = self.true_labels

        # filter for normal and not normal loss
        normal = np.asarray([scores[i] for i in range(len(true_labels)) if true_labels[i] == 0])
        not_normal = np.asarray([scores[i] for i in range(len(true_labels)) if true_labels[i] == 1])
        # combine normal and not normal to one list
        data = [normal, not_normal]
        # assign labels and color
        group_labels = ['normal', 'not normal']
        colors = ['#A56CC1', '#A6ACEC']

        # Create distplot with curve_type set to 'normal'
        fig = ff.create_distplot(data, group_labels, colors=colors,
                                 bin_size=self.bin_size, show_rug=False,
                                 histnorm='probability')

        # Add title
        fig.update_layout(title_text='Distribution of Losses')

        return fig
