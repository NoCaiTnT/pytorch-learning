import torch
import torch.nn as nn

net = nn.RNN(10, 20, 2)

net = nn.GRU(10, 20, 2)

net = nn.LSTM(10, 20, 2)

net = nn.Transformer(nhead=4)