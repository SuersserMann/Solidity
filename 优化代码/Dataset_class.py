from datasets import load_dataset
import torch
class Dataset(torch.utils.data.Dataset):
    def __init__(self, split):
        self.dataset = load_dataset("mwritescode/slither-audited-smart-contracts", 'big-multilabel', split=split,
                                    verification_mode='no_checks')
        # 从huggingface导入数据集

    def __len__(self):
        return len(self.dataset)
        # 计算数据集长度，方便后面进行一个批量的操作

    def __getitem__(self, i):
        source_code = self.dataset[i]['source_code']
        bytecode = self.dataset[i]['bytecode']
        label = self.dataset[i]['slither']
        # 从数据集遍历数据，比如第一批是16个，那么第二批就可以从17-32
        return source_code, bytecode, label