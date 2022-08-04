from torch.autograd import Variable
from torch import nn, numel
import torch
import numpy as np


def run_epoch(epoch, model, dataloader, cuda, training=False, optimizer=None):
    if training:
        model.train()
    else:
        model.eval()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if cuda: inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets.long())
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        if cuda:
            correct += predicted.eq(targets.data).cpu().sum().item()
        else:
            correct += predicted.eq(targets.data).sum().item()
    acc = 100 * correct / total
    avg_loss = total_loss / total
    return acc, avg_loss


def get_predictions(model, dataloader, cuda):#, get_probs=False):
    probs_list = []
    preds = []
    correction_count = 0
    data_count = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if cuda: inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets.long())
        outputs = model(inputs)
        '''
        if get_probs:
            probs = torch.nn.functional.softmax(outputs, dim=1)
            if cuda: probs = probs.data.cpu().numpy()
            else: probs = probs.data.numpy()
            preds.append(probs)
        else:
            _, predicted = torch.max(outputs.data, 1)
            corrections = (predicted-targets)
            correction_count = correction_count + numel(corrections[corrections==0])
            #print(correction_count)
        if cuda: predicted = predicted.cpu()
            preds += list(predicted.numpy().ravel())
        '''
        probs = torch.nn.functional.softmax(outputs, dim=1)
        if cuda: probs = probs.data.cpu().numpy()
        else: probs = probs.data.numpy()
        probs_list.append(probs)

        _, predicted = torch.max(outputs.data, 1)
        corrections = (predicted-targets)
        correction_count = correction_count + numel(corrections[corrections==0])
        data_count = data_count + 1
        #print(correction_count)
        if cuda: predicted = predicted.cpu()
        preds += list(predicted.numpy().ravel())
    return np.vstack(probs_list), np.array(preds), correction_count/data_count*100
'''
    if get_probs:
        return np.vstack(preds)
    else:
        return np.array(preds), correction_count
'''
