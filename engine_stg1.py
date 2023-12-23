from collections import OrderedDict
from contextlib import suppress
import pandas as pd
import time
from tqdm import tqdm
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import sklearn
import numpy as np
import pickle

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from data.dataset import ValideImageDataset, TestImageDataset, ImageDataset
from data.samplers import OrderedDistributedSampler
import torch.distributed as dist

from timm.models import model_parameters
from timm.utils import accuracy, AverageMeter, dispatch_clip_grad, reduce_tensor


def train_one_epoch(args, model, criterion, data_loader, saver, lr_scheduler, optimizer, device, epoch, loss_scaler,
                    _logger, model_ema=None, mask=False, amp_autocast=suppress):
    model.train()

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()

    end = time.time()
    last_idx = len(data_loader) - 1
    num_updates = epoch * len(data_loader)

    for batch_idx, (samples, targets, mask_token, coarse_anno) in enumerate(data_loader):
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        samples = samples.to(device)
        targets = targets.to(device)
        mask_token = mask_token.to(device)
        coarse_anno = coarse_anno.to(device)
        coarse_anno = coarse_anno.float()

        with amp_autocast():
            outputs = model(samples)
            if mask:
                loss = criterion(outputs, targets, mask_token, coarse_anno, epoch)
            else:
                loss = criterion(outputs, targets)

        if not args.distributed:
            losses_m.update(loss.item(), samples.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad, mode=args.clip_mode)
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()
        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), samples.size(0))

            if args.local_rank == 0:
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(data_loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=samples.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=samples.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m))

        if saver is not None and args.recovery_interval and (
                last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        end = time.time()

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])


def dist_gather_tensor(tensor, names, features, slide_names, world_size=None):
    if world_size is None:
        world_size = dist.get_world_size()
    gathered_tensor = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensor, tensor)
    slide_name = list(map(lambda x: x.strip(), names))
    gathered_slide_names = [[] for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered_slide_names, slide_name)

    features.extend([output.cpu() for sublist in gathered_tensor for output in sublist])
    slide_names.extend([name for sublist in gathered_slide_names for name in sublist])


def get_all(features, names, df, split='train'):
    all = []
    labels = []
    name_feature_dict = {}
    for feature, name in zip(features, names):
        if name not in name_feature_dict:
            name_feature_dict[name] = []
        name_feature_dict[name].append(feature.cpu())
    for name, f in name_feature_dict.items():
        all.append(torch.stack(f).mean(dim=0))
        selected_row = df.loc[df[split] == name]
        labels.append(int(selected_row[split + '_label'].values[0]))

    all = np.vstack(all)
    labels = np.vstack(labels)
    return all, labels.squeeze()


def evalknn(model, csv_path, device, epoch, args, _logger, amp_autocast=suppress, log_suffix=''):
    sample_ratio = args.sample_ratio
    knn_train_dataset = ValideImageDataset(args, csv_path, sample_ratio, epoch, split='train')
    split_value = 'val' if not args.eval else 'test'
    knn_val_dataset = ValideImageDataset(args, csv_path, sample_ratio, epoch, split=split_value)

    if args.distributed:
        sampler_knn_train = OrderedDistributedSampler(knn_train_dataset)
        sampler_knn_val = OrderedDistributedSampler(knn_val_dataset)
    else:
        sampler_knn_train = torch.utils.data.SequentialSampler(knn_train_dataset)
        sampler_knn_val = torch.utils.data.SequentialSampler(knn_val_dataset)

    train_loader = torch.utils.data.DataLoader(knn_train_dataset,
                                                   sampler=sampler_knn_train,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   pin_memory=args.pin_mem,
                                                   drop_last=False)

    val_loader = torch.utils.data.DataLoader(knn_val_dataset,
                                                 sampler=sampler_knn_val,
                                                 batch_size=args.batch_size,
                                                 num_workers=args.num_workers,
                                                 pin_memory=args.pin_mem,
                                                 drop_last=False)
    batch_time_m = AverageMeter()

    train_tensors = []
    train_names = []
    df = pd.read_csv(csv_path)

    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (train_input, train_name) in enumerate(tqdm(train_loader)):
            train_input = train_input.to(device)

            with amp_autocast():
                train_outputs = model(train_input)
            if isinstance(train_outputs, (tuple, list)):
                train_outputs = train_outputs[-1]

            if args.distributed:
                dist.barrier()
                dist_gather_tensor(train_outputs, train_name, train_tensors, train_names)
            else:
                train_tensors.extend(train_outputs)
                train_names.extend(train_name)

        train_features, train_labels = get_all(train_tensors, train_names, df, split='train')

        val_tensors = []
        val_names = []
        last_idx = len(val_loader) - 1
        for batch_idx, (val_input, val_name) in enumerate(tqdm(val_loader)):
            last_batch = batch_idx == last_idx
            val_input = val_input.to(device)

            with amp_autocast():
                val_outputs = model(val_input)
            if isinstance(val_outputs, (tuple, list)):
                val_outputs = val_outputs[-1]

            if args.distributed:
                dist.barrier()
                dist_gather_tensor(val_outputs, val_name, val_tensors, val_names)
            else:
                val_tensors.extend(val_outputs)
                val_names.extend(val_name)

        val_features, val_labels = get_all(val_tensors, val_names, df, split='val')

        label_enc = LabelEncoder().fit(train_labels)
        train_labels = label_enc.transform(train_labels)
        val_labels = label_enc.transform(val_labels)

        knn = KNeighborsClassifier().fit(train_features, train_labels)
        y_score = knn.predict_proba(val_features)

        if len(np.unique(val_labels)) > 2:
            aucs = sklearn.metrics.roc_auc_score(val_labels, y_score, average='macro', multi_class='ovr')
        else:
            aucs = sklearn.metrics.roc_auc_score(val_labels, y_score[:, 1])

        torch.cuda.synchronize()

        batch_time_m.update(time.time() - end)

        if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
            log_name = 'Test' + log_suffix
            _logger.info(
                '{0}: [{1:>4d}/{2}]  '
                'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                'AUC: {auc:>7.4f}'.format(
                    log_name, batch_idx, last_idx, batch_time=batch_time_m, auc=aucs))

    metrics = OrderedDict([('acc1', aucs)])

    return metrics


def extract_features(model, csv_path, device, args, save_dir, _logger, amp_autocast=suppress):
    df = pd.read_csv(csv_path)
    s1 = df['train'].dropna()
    s2 = df['val'].dropna()
    s3 = df['test'].dropna()
    slide_names = pd.concat([s1, s2, s3]).reset_index(drop=True)
    for slide_name in tqdm(slide_names.values):
        slide_folder = os.path.join(args.data_path, slide_name)
        test_dataset = TestImageDataset(slide_folder)

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                     batch_size=1,
                                                     num_workers=args.num_workers,
                                                     pin_memory=args.pin_mem,
                                                     drop_last=False)
        all_topk = []
        coords = []
        token_indx = []

        model.eval()

        with torch.no_grad():
            for batch_idx, (input, coord) in enumerate(test_loader):
                input = input.to(device)

                with amp_autocast():
                    outputs = model(input)

                tile_cls, token_cls, token_feat, cls_feat = outputs
                token_cls = F.softmax(token_cls, dim=2)
                tile_cls = F.softmax(tile_cls, dim=1)
                cls = torch.argmax(tile_cls, dim=1)
                B, N, C = token_cls.shape

                token_l = token_cls[range(B), :, cls+1]
                _, topk_indices = token_l.topk(args.num_topk, dim=1)
                batch_indices = torch.arange(B).view(B, 1).expand_as(topk_indices)
                topk_feats = token_feat[batch_indices, topk_indices]
                cls_feat = cls_feat.unsqueeze(1)
                tile_feats = torch.concatenate([cls_feat, topk_feats], dim=1)

                all_topk.extend(tile_feats.cpu())
                coords.append([t.item() for t in coord])
                token_indx.extend(topk_indices.cpu())

        name_feature_dict = {'feature': torch.stack(all_topk).numpy(), 'tile_coord': coords,
                             'token_coord': torch.stack(token_indx).numpy()}
        save_path = os.path.join(save_dir, slide_name + '.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(name_feature_dict, f)
    return


def validate(model, csv_path, device, criterion, args, _logger, amp_autocast=suppress, log_suffix=''):
    split_value = 'val' if not args.eval else 'test'
    val_dataset = ImageDataset(args.data_path, img_size=args.input_size, mask_path=args.mask_path, patch_size=args.patch_size,
            df_path=csv_path, mil=args.mil, sp=split_value)

    if args.distributed:
        sampler_knn_val = OrderedDistributedSampler(val_dataset)
    else:
        sampler_knn_val = torch.utils.data.SequentialSampler(val_dataset)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 sampler=sampler_knn_val,
                                                 batch_size=args.batch_size,
                                                 num_workers=args.num_workers,
                                                 pin_memory=args.pin_mem,
                                                 drop_last=False)
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top1_t = AverageMeter()

    model.eval()

    end = time.time()
    with torch.no_grad():
        last_idx = len(val_loader) - 1
        for batch_idx, (val_input, target, mask_token, _) in enumerate(val_loader):
            last_batch = batch_idx == last_idx
            val_input = val_input.to(device)
            target = target.to(device)
            mask_token = mask_token.view(-1).to(device)

            with amp_autocast():
                val_outputs = model(val_input)
            if isinstance(val_outputs, (tuple, list)):
                output = val_outputs[0]
                token = val_outputs[1].view(-1, 3)

            loss = criterion(output, target) + criterion(token, mask_token)
            acc1 = accuracy(output, target, topk=(1,))[0]
            acct = accuracy(token, mask_token, topk=(1,))[0]

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acct = reduce_tensor(acct, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), val_input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top1_t.update(acct.item(), token.size(0))

            batch_time_m.update(time.time() - end)

            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'ACC1: {top1.val:>7.4f} ({top1.avg:>7.4f})'
                    'ACCT: {topt.val:>7.4f} ({topt.avg:>7.4f})'        .format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m, top1=top1_m, topt=top1_t))

    metrics = OrderedDict([('acc1', top1_m.avg), ('acct', top1_t.avg)])

    return metrics
