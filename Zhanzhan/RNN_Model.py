"""
Author: Zhan Yi
Date: 2023/4/14
Description: RNN Model, we need to pass the config to the model
"""

import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.embedding = nn.Embedding(config.embed, config.hidden_size1)

        self.lstm = nn.LSTM(config.hidden_size1, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        embed = self.embedding(x)

        out, _ = self.lstm(embed)

        aa = out[:, -1, :]
        out = self.fc(aa)  # 句子最后时刻的 hidden state
        return out