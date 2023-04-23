"""
Author: Zhan Yi
Date: 2023/4/14
Description: there are two main functions: get_tensor_dataset() and get_raw_dataset()
get_raw_dataset() is used to get the raw dataset from the slither-audited-smart-contracts and saved as pkl
get_tensor_dataset() uses the transformers model process the code data into tensor
"""

from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch



def write_pkl(data_frame: pd.DataFrame, path, file_name):
    data_frame.to_pickle(path + file_name)

def get_raw_dataset():
    train_set = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split='train', verification_mode='no_checks')

    address_list = []
    source_code_list = []
    bytecode_list = []
    slither_list = []

    #只保留slither第一个标签
    for i in range(len(train_set)):
        byte_info = dict()
        source_code_list.append(train_set[i]['source_code'])
        slither_list.append(train_set[i]['slither'])
        bytecode_list.append(train_set[i]['bytecode'])
        address_list.append(train_set[i]['address'])

    pkl_data = {'address': address_list, 'source_code': source_code_list, 'bytecode': bytecode_list, 'slither': slither_list}
    our_dataset = pd.DataFrame(pkl_data)

    write_pkl(our_dataset[['address', 'source_code', 'bytecode', 'slither']], "", "data/input.pkl")

def cutToken(tokens, token_list):
    """
    Cut tokens which are too long
    """
    if len(tokens) > 500:
        token_list.append(tokens[0:500])
        tokens = tokens[500: len(tokens)]
        cutToken(tokens, token_list)
    else:
        token_list.append(tokens)
    return token_list


def get_code_tensor_dataset():
    dataset = pd.read_pickle("data/input.pkl")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")

    source_code_list = []
    slither_list = []
    token_size = []

    for index in range(len(dataset)):
        data = dataset.iloc[index]
        source_code = data['source_code']
        slither = data['slither'][0]

        source_code = source_code.replace("\t", "").split("\n")
        source_code = list(filter(None, source_code))

        nl_list = []
        code_list = []

        for line in source_code:
            line = line.strip()
            if len(line) > 2:
                if "//" == line[0:2] or "/*" == line[0:2] or "*" == line[0]:
                    nl_list.append(line)
                else:
                    code_list.append(line)
            else:
                code_list.append(line)

        code = ""
        nl = ""
        for str in code_list:
            code = code + " " + str

        for str in nl_list:
            nl = nl + " " + str

        code_tokens = tokenizer.tokenize(code)
        nl_tokens = tokenizer.tokenize(nl)

        token_list = []
        token_embeddings = []
        # tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token] + code_tokens + [tokenizer.sep_token]
        tokens = code_tokens
        token_list = cutToken(tokens, token_list)

        for token in token_list:
            token_id = tokenizer.convert_tokens_to_ids(token)
            context_embeddings = model(torch.tensor(token_id)[None, :])[0]
            token_embeddings.append(context_embeddings)

        torch_tensor = torch.cat(token_embeddings, dim=1)
        torch_tensor = torch.tensor(torch_tensor.tolist()[0])
        token_size.append(torch_tensor.size(0))
        source_code_list.append(torch_tensor)
        slither_list.append(slither)

        print("the ", index, " is processed")

        if index == 50:
            break

    # max_size = max(token_size)
    # for source_code in source_code_list:
    #     x_zero = torch.zeros(max_size, 840).float()
    #     x_zero[:source_code.size(0), :] = source_code

    pkl_data = {'source_code': source_code_list,'slither': slither_list, 'token_size': token_size}
    our_dataset = pd.DataFrame(pkl_data)
    write_pkl(our_dataset[['source_code', 'slither', 'token_size']], "", "data/input_tensor_one_label.pkl")

def get_bytecode_dataset():
    dataset = pd.read_pickle("data/input.pkl")
    bytecode_list = []
    slither_list = []

    for index in range(len(dataset)):
        data = dataset.iloc[index]
        bytecode = data['bytecode']
        slither = data['slither'][0]
        bytecode = bytecode[2:]
        prefixed_byte = ""
        byte_list = []

        for i in range(0, len(bytecode), 2):
            prefixed_byte += '0x' + bytecode[i:i+2] + ""

        for j in range(0, len(prefixed_byte), 4):
            byte = int(prefixed_byte[j:j + 4], 16)
            byte_list.append(byte)

        bytecode_list.append(torch.tensor(byte_list))
        slither_list.append(slither)

        if index == 50:
            break

    pkl_data = {'bytecode': bytecode_list, 'slither': slither_list}
    our_dataset = pd.DataFrame(pkl_data)
    write_pkl(our_dataset[['bytecode', 'slither']], "", "data/input_bytecode.pkl")



if __name__ == "__main__":
    get_code_tensor_dataset()
    # get_raw_dataset()
    # get_bytecode_dataset()
