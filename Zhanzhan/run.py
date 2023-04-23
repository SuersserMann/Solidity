"""
Author: Zhan Yi
Date: 2023/4/14
Description: train model
"""

from configures import data_args, train_args, model_args
from data_loader import get_dataloader
from RNN_Model import Model
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import os
import torch
import shutil
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, accuracy_score
from sklearn.preprocessing import label_binarize
from time import process_time
import pandas as pd


def train():
    print('start loading data====================')
    dataloader = get_dataloader(data_args)

    print('start training model==================')
    model = Model(model_args)
    model.to('cpu')

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=train_args.learning_rate, weight_decay=train_args.weight_decay)

    best_acc = 0.0

    # save path for model
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if not os.path.isdir(os.path.join('checkpoint', model_args.model_name)):
        os.mkdir(os.path.join('checkpoint', f"{model_args.model_name}"))
    ckpt_dir = f"./checkpoint/{model_args.model_name}/"

    early_stop_count = 0
    for epoch in range(train_args.max_epochs):
        acc = []
        loss_list = []
        model.train()

        for data, label in dataloader['train']:

            data = data.to(model_args.device)

            pre = model(data)
            loss = criterion(pre, label)

            # optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
            optimizer.step()

            ## record
            _, prediction = torch.max(pre, -1)
            acc.append(prediction.eq(label).cpu().numpy())
            loss_list.append(loss.item())

        # report train msg
        epoch_acc = np.concatenate(acc, axis=0).mean()
        epoch_loss = np.average(loss_list)
        print(f"Train Epoch:{epoch}  |Loss: {epoch_loss:.3f} | Train Acc: {epoch_acc:.3f}")

        # report valid msg
        valid_state = evaluate(dataloader['valid'], model, criterion)
        print(f"Eval Epoch: {epoch} | Loss: {valid_state['loss']:.3f} | Eval Acc: {valid_state['acc']:.3f}")

        # report test msg
        test_state = evaluate(dataloader['test'], model, criterion)
        print(f"Test Epoch: {epoch} | Loss: {test_state['loss']:.3f} | Test Acc: {test_state['acc']:.3f}")

        # only save the best model
        is_best = (valid_state['acc'] > best_acc)

        if valid_state['acc'] > best_acc:
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count > train_args.early_stopping:
            break

        if is_best:
            best_acc = valid_state['acc']
            early_stop_count = 0
        if is_best or epoch % train_args.save_epoch == 0:
            save_best(ckpt_dir, epoch, model, model_args.model_name, valid_state['acc'], is_best, 666)

    print(f"The best validation accuracy is {best_acc}.")

    # report test msg
    checkpoint = torch.load(os.path.join(ckpt_dir, f'{model_args.model_name}_best_666.pth'))
    model.update_state_dict(checkpoint['net'])
    test_state = test(dataloader['test'], model, criterion)
    print(
        f"Test Acc: {test_state['acc']:.3f}   | Test F1: {test_state['f1']:.3f}   | Test AUC: {test_state['auc']:.3f}   | Test MCC: {test_state['mcc']:.3f}")


def test(valid_dataloader, model, criterion):
    acc = []

    prob_all = []
    label_all = []

    loss_list = []
    model.eval()

    with torch.no_grad():
        for data, label in valid_dataloader:
            data = data.to(model_args.device)
            pre = model(data)
            loss = criterion(pre, label)

            ## record
            _, prediction = torch.max(pre, -1)
            prob_all.extend(prediction.cpu())
            label_all.extend(label.cpu())

            loss_list.append(loss.item())
            acc.append(prediction.eq(label).cpu().numpy())

        F1 = f1_score(list(label_all), list(prob_all), average='macro')
        # AUC
        labels = [0, 1, 2]
        label_all_for_auc = label_binarize(label_all, classes=labels)
        prob_all_for_auc = label_binarize(prob_all, classes=labels)
        AUC = roc_auc_score(label_all_for_auc, prob_all_for_auc, average='macro')
        # MCC
        MCC = matthews_corrcoef(list(label_all), list(prob_all))

        state = {'loss': np.average(loss_list),
                 'acc': np.concatenate(acc, axis=0).mean(),
                 'f1': F1,
                 'auc': AUC,
                 'mcc': MCC}

    return state


def evaluate(valid_dataloader, model, criterion):
    acc = []
    loss_list = []
    model.eval()

    with torch.no_grad():
        for data, label in valid_dataloader:
            data = data.to(model_args.device)
            pre = model(data)
            loss = criterion(pre, label)

            ## record
            _, prediction = torch.max(pre, -1)
            loss_list.append(loss.item())
            acc.append(prediction.eq(label).cpu().numpy())

        valid_state = {'loss': np.average(loss_list),
                       'acc': np.concatenate(acc, axis=0).mean()}

    return valid_state


def save_best(ckpt_dir, epoch, model, model_name, valid_acc, is_best, k_fold):
    if is_best:
        print('saving best....')
    else:
        print('saving last....')

    model.to('cpu')
    state = {
        'net': model.state_dict(),
        'epoch': epoch,
        'acc': valid_acc
    }
    pth_name = f"{model_name}_latest_{str(k_fold)}.pth"
    best_pth_name = f'{model_name}_best_{str(k_fold)}.pth'
    ckpt_path = os.path.join(ckpt_dir, pth_name)
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copy(ckpt_path, os.path.join(ckpt_dir, best_pth_name))
    model.to('cpu')

if __name__ == "__main__":

    train()

