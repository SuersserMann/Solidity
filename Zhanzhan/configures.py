"""
Author: Zhan Yi
Date: 2023/4/14
Description: configures
"""

import os
import torch
from tap import Tap
from typing import List


class DataParser(Tap):
    path: str = "data/input_bytecode.pkl"
    num_classes: int = 6
    avg_num_nodes: int = 150
    data_split_ratio: List = [0.6, 0.2, 0.2]  # the ratio of training, validation and testing set for random split
    batch_size: int = 5
    seed: int = 1

class TrainParser(Tap):
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    max_epochs: int = 100
    save_epoch: int = 10
    early_stopping: int = 100


class RNN_ModelParser(Tap):
    device: str = "cpu"
    model_name = 'RNN'
    embed: int = 24428   # 768 字向量维度, 若使用了预训练词向量，则维度统一
    hidden_size1: int = 256  # emb隐藏层
    hidden_size: int = 128  # lstm隐藏层
    num_layers: int = 2
    dropout: float = 0.5
    num_classes: int = 6



data_args = DataParser().parse_args(known_only=True)
train_args = TrainParser().parse_args(known_only=True)
model_args = RNN_ModelParser().parse_args(known_only=True)
