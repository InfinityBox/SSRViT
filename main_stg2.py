import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import sys
import os
import argparse
import logging
from pathlib import Path
import pandas as pd
import yaml
import numpy as np
import csv
from collections import OrderedDict

from timm.scheduler import create_scheduler
from timm.utils import *
from timm.models import resume_checkpoint

import torch
from torch.utils.data import Dataset
import torch.optim as optim
from data.dataset import WSIDataset
from models.SSRViT import SViT
from engine_stg2 import train_slides, eval_slides
from utils.utils import init_distributed_mode, get_rank
from utils.losses import focal_loss


_logger = logging.getLogger('train')

sys.path.append(os.path.dirname(os.path.abspath(__file__) + '/../../'))
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='Train and test the wsi classification', add_help=False)
parser.add_argument('--folds', default=5, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--max_patience', default=20, type=int)
parser.add_argument('--img_classes', default=2, type=int)
parser.add_argument('--device', default='cuda', help='device to use for training / testing')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--h5', default=True, type=int, help='feature file is .h5 or .pkl')
parser.add_argument('--eval', action='store_true', default=True, help='test process')
parser.add_argument('--model', default='SViT', type=str, help='name of training model')

parser.add_argument('--data_path', default='', type=str, help='dataset path')
parser.add_argument('--csv_path', default='', type=str, help='train/val/test csv path')
parser.add_argument('--output_dir', default='', type=str, help='model & log save path')

parser.add_argument('--focal', type=bool, default=False,  help='use focal loss or not')
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
parser.add_argument('--warmup-lr', type=float, default=5e-6,  help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min-lr', type=float, default=1e-5, help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

parser.add_argument('--GPU', default='1', type=str, help='which GPU to use, default 0')
parser.add_argument('--warmup-epochs', type=int, default=10, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=5, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')

def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    setup_default_logging()
    args, args_text = _parse_args()
    device = init_distributed_mode(args)

    if args.seed is not None:
        random_seed(args.seed, get_rank())

    val_auc = []
    val_acc = []

    for fold in range(args.folds):
        _logger.info("************** FOLD (%d/%d) **************" % (fold + 1, args.folds))
        csv_path = os.path.join(args.csv_path, 'fold_{}.csv'.format(fold))
        df = pd.read_csv(csv_path)

        data_path = os.path.join(args.data_path, str(fold))

        dataset_train = WSIDataset(data_path, df, h5=args.h5, splits='train')
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, num_workers=12, shuffle=True)

        dataset_valid = WSIDataset(data_path, df, h5=args.h5, splits='val')
        dataloader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=1, num_workers=12, shuffle=False)

        _logger.info(f"Start training for {args.epochs} epochs in fold {fold}")

        model = SViT(dim=384, depth=3, num_heads=4, n_classes=args.img_classes)

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _logger.info('number of params: %d', n_parameters)

        model.to(device)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        lr_scheduler, num_epochs = create_scheduler(args, optimizer)

        if args.focal:
            criterion = focal_loss(alpha=[1, 0.75, 0.75], gamma=2, num_classes=3).to(device)
        else:
            criterion = torch.nn.CrossEntropyLoss().to(device)

        output_dir = os.path.join(args.output_dir, str(fold))
        if not os.path.isdir(output_dir):
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        if args.eval:
            dataset_test = WSIDataset(data_path, df, h5=args.h5, splits='test')
            dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, num_workers=10, shuffle=False)

            eval_path = os.path.join(output_dir, 'model_best.pth.tar')
            _ = resume_checkpoint(
                model, eval_path,
                log_info=args.local_rank == 0)

            eval_stats = eval_slides(model, device, dataloader_test, criterion, _logger)

            rowd = OrderedDict(epoch=fold)
            rowd.update([('eval_' + k, v) for k, v in eval_stats.items()])
            with open(os.path.join(args.output_dir, 'summary.csv'), mode='a') as cf:
                dw = csv.DictWriter(cf, fieldnames=rowd.keys())
                if fold == 0:
                    dw.writeheader()
                dw.writerow(rowd)

            val_auc.append(eval_stats["auc"])
            val_acc.append(eval_stats["acc"])

            continue

        early_stop_counter = 0
        eval_best = 0.0
        max_auc = 0.0
        max_acc = 0.0
        early_stop = False
        best_metric = None

        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, checkpoint_dir=output_dir,
            recovery_dir=output_dir, max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

        for epoch in range(0, args.epochs):
            train_stats = train_slides(model, device, dataloader_train, optimizer, epoch, criterion, _logger)
            eval_stats = eval_slides(model, device, dataloader_valid, criterion, _logger)

            eval_best = max(eval_best, eval_stats['acc'] + eval_stats['auc'])

            lr_scheduler.step(epoch + 1)

            if output_dir is not None:
                update_summary(
                    epoch, train_stats, eval_stats, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_stats['acc'] + eval_stats['auc']
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

            if not eval_best == eval_stats['acc'] + eval_stats['auc']:
                early_stop_counter += 1
                if early_stop_counter > args.max_patience:
                    early_stop = True
            else:
                max_acc = eval_stats['acc']
                max_auc = eval_stats['auc']
                early_stop_counter = 0

            if early_stop:
                val_acc.append(max_acc)
                val_auc.append(max_auc)
                break

    val_accmean, val_accstd = np.mean(val_acc), np.std(val_acc)
    val_aucmean, val_aucstd = np.mean(val_auc), np.std(val_auc)
    _logger.info(f"acc and auc validation mean(std): {val_accmean:.2f} +/- {val_accstd:.2f}"
                 f" and {val_aucmean * 100:.2f} +/- {val_aucstd * 100:.2f}")


if __name__ == '__main__':
    main()
