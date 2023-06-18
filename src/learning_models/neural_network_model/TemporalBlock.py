import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.nn import BatchNorm1d, Conv1d, Dropout, Flatten, Linear, Sequential, Softmax, Tanh, MaxPool1d, ReLU, Sigmoid

# custom support files
from src.utils.gesture_data_related.read_data import read_data
from src.utils.neural_network_related.format_data_for_nn import format_batch_data
from src.utils.neural_network_related.task_generator import HandGestureTask, HandGestureDataSet, get_data_loader

num_classes = 10

class TemporalBlock(nn.Module):
    def __init__(self, input_size, feature_dim):
        super(TemporalBlock, self).__init__()

        layers = []
        #self.input_size = input_size

        # Adding 2 cnn-1d layer to the network
        cnn_layer1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm1d(128),
            #nn.MaxPool1d(kernel_size=2),
            nn.ReLU(),)
        layers.append(cnn_layer1)

        cnn_layer2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3,
                      stride=1, padding=1),
            nn.BatchNorm1d(128),
            #nn.MaxPool1d(kernel_size=2),
            nn.ReLU(),)
        layers.append(cnn_layer2)

        # cnn_layer = nn.Sequential(
        #     nn.Conv2d(in_channels=self.hidden_layers[i - 1], out_channels=self.hidden_layers[i],
        #               kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(self.hidden_layers[i]),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.ReLU(),
        #     nn.Dropout(drop_out), )
        # layers.append(cnn_layer)

        # Adding 2 FC layer to the network
        FC_layer1 = nn.Sequential(
            nn.Linear(3840, 256),      #  cnn_layer2_out_channels(128) x seq_len + 63
            nn.ReLU())
        layers.append(FC_layer1)

        FC_layer2 = nn.Sequential(
            nn.Linear(256, feature_dim),)
        layers.append(FC_layer2)

        self.network = nn.ModuleList(layers)

        #print("printing layers \n", self.layers)

    def forward(self, x):
        cnn_input1 = x.float()
        cnn_output1 = 0
        cnn_output1 = self.network[0](cnn_input1)
        cnn_input2 = cnn_output1
        cnn_output2 = 0
        cnn_output2 = self.network[1](cnn_input2)

        #cnn_output2 = torch.cat((x, cnn_output2), 1)

        #FC_input1 = torch.reshape(cnn_output2, (cnn_output2.shape[0], cnn_output2.shape[1]))
        FC_input1 = cnn_output2.view(cnn_output2.size(0), -1)
        #print(FC_input1.shape)
        FC_output1 = self.network[2](FC_input1.float())
        FC_input2 = FC_output1
        FC_output2 = self.network[3](FC_input2)

        return FC_output2

# model = TemporalBlock(input_size, hidden_size, num_classes, norm_layer=norm_layer).to(device)
# inp = torch.randn(1, 3, 32, 32).to (device)
# # out = model.forward(inp)
# print(model.forward(inp).shape)
# model.apply(weights_init)
# # for name, param in model.named_parameters():
# #     print(name, param)

# # Print the model
# print(model)