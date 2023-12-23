import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from collections import OrderedDict

import torch
import torch.nn.functional as F

from timm.utils import AverageMeter


def train_slides(model, device, train_loader, optimizer, epoch, criterion, _logger):
    model.train()

    losses_m = AverageMeter()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 100 == 0:
            _logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        losses_m.update(loss.item(), data.size(0))

    return OrderedDict([('loss', losses_m.avg)])


def eval_slides(model, device, test_loader, criterion, _logger):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    label = []
    p = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            test_loss += criterion(logits, target).item()  # sum up batch loss

            _, predicted = torch.max(logits, 1)
            logits = F.softmax(logits, dim=1)

            total += target.size(0)
            correct += (predicted == target).sum().item()

            label.extend(target.cpu().numpy())
            p.extend(logits.cpu().numpy())

        acc = 100. * correct / total
        if len(np.unique(label)) > 2:
            auc = roc_auc_score(label, p, average='macro', multi_class='ovr')
        else:
            auc = roc_auc_score(label, p[:, 1])
        test_loss /= total
        _logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%), AUC: {:.3f}%\n'.format(
            test_loss, correct, total, acc, auc))
        cm = confusion_matrix(label, np.round(p))

        metrics = OrderedDict([('acc', acc), ('auc', auc), ('cm', cm), ('predict', p), ('label', label)])
    return metrics
