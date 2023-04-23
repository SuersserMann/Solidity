"""
Author: Zhan Yi
Date: 2023/4/14
Description: dataloader
"""

import pandas as pd
from torch.utils.data import random_split, Subset, ConcatDataset
from torch.utils.data import Dataset, DataLoader
import torch
from torch_geometric.loader import DataLoader
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence


class MyDataset(Dataset):
    def __init__(self, code_list, slither_list):
        self.code_list = code_list
        self.slither_list = slither_list
    def __len__(self):
        return len(self.code_list)
    def __getitem__(self, index):
        return torch.tensor(self.code_list[index]), self.slither_list[index]


def get_dataloader(data_args):

    dataset = pd.read_pickle(data_args.path)
    dataloader = dict()

    if data_args.path == "data/input_tensor_one_label.pkl":
        source_code_list = []
        slither_list = []

        for index in range(len(dataset)):
            data = dataset.iloc[index]
            source_code = torch.tensor(data['source_code'])
            slither = torch.tensor(data['slither'])

            source_code_list.append(source_code)
            slither_list.append(slither)

        # padding
        source_code_list = pad_sequence(source_code_list,batch_first=True).tolist()

        dataset = MyDataset(source_code_list, slither_list)
        data_split_ratio = data_args.data_split_ratio
        seed = data_args.seed
        batch_size = data_args.batch_size

        train, valid, test = random_split(dataset, lengths=[int(data_split_ratio[0] * len(dataset)),
                                                               int(data_split_ratio[1] * len(dataset)),
                                                               len(dataset) - int(data_split_ratio[0] * len(dataset)) - int(
                                                                   data_split_ratio[1] * len(dataset))],
                                                generator=torch.Generator().manual_seed(seed))

        dataloader['train'] = DataLoader(train, batch_size= batch_size, shuffle=True)
        dataloader['valid'] = DataLoader(valid, batch_size= batch_size, shuffle=True)
        dataloader['test'] = DataLoader(test, batch_size= batch_size, shuffle=True)

    elif data_args.path == "data/input_bytecode.pkl":
        bytecode_list = []
        slither_list = []

        for index in range(len(dataset)):
            data = dataset.iloc[index]
            source_code = torch.tensor(data['bytecode'])
            slither = torch.tensor(data['slither'])

            bytecode_list.append(source_code)
            slither_list.append(slither)

        # padding
        bytecode_list = pad_sequence(bytecode_list, batch_first=True).tolist()

        dataset = MyDataset(bytecode_list, slither_list)
        data_split_ratio = data_args.data_split_ratio
        seed = data_args.seed
        batch_size = data_args.batch_size

        train, valid, test = random_split(dataset, lengths=[int(data_split_ratio[0] * len(dataset)),
                                                            int(data_split_ratio[1] * len(dataset)),
                                                            len(dataset) - int(
                                                                data_split_ratio[0] * len(dataset)) - int(
                                                                data_split_ratio[1] * len(dataset))],
                                          generator=torch.Generator().manual_seed(seed))

        dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
        dataloader['valid'] = DataLoader(valid, batch_size=batch_size, shuffle=True)
        dataloader['test'] = DataLoader(test, batch_size=batch_size, shuffle=True)



    return dataloader