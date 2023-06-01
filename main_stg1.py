import argparse
import datetime
import numpy as np
import os
import time
import logging
from pathlib import Path
from contextlib import suppress
import yaml
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.utils.data as udata

from timm.models import create_model, load_checkpoint, convert_splitbn_model, resume_checkpoint
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.utils import *

from data.dataset import ImageDataset
from models.SSRViT import SRViT
from utils import utils
from utils.losses import CERCELoss
from engine_stg1 import train_one_epoch, evalknn, extract_features, validate

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
    use_amp = 'apex'
except ImportError:
    has_apex = False
    use_amp = 'native'

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

_logger = logging.getLogger('train')

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser('SSRViT training and evaluation script', add_help=False)
# Global & Device parameters
parser.add_argument('--folds', default=5, type=int)
parser.add_argument('--batch-size', default=4, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--num_classes_img', default=3, type=int, help='number of classes for image classification')
parser.add_argument('--num_classes_tk', default=3, type=int, help='number of classes for token classification')
parser.add_argument('--mask', default=True, type=bool, help='use shrink mask for tokens in training loop')
parser.add_argument('--mil', default=True, type=int, help='if the dataset only contains slide-level label, set True')
parser.add_argument('--seed', default=0, type=int)

parser.add_argument('--device', default='cuda', help='device to use for training / testing')

# Dataset parameters
parser.add_argument('--data_path', default='', type=str, help='dataset path')
parser.add_argument('--csv_path', default='', type=str, help='csv path')
parser.add_argument('--input-size', default=1024, type=int, help='input image size')
parser.add_argument('--patch-size', default=32, type=int, help='size of every patch in an input image')
parser.add_argument('--mask_path', default='', type=str, help='path of shrink mask for tokens')
parser.add_argument('--output_dir', default='', type=str, help='model & log save path')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--repeated_aug', default=False, type=bool)
parser.add_argument('--sample_ratio', default=0.5, type=int, help='sample ratio for faster speed during KNN validation')

parser.set_defaults(pin_mem=True)

# eval & extarct features settings
parser.add_argument('--knn_eval', action='store_true', default=True,
                    help='Perform knn evaluation (slide-level) in train&test process')
parser.add_argument('--eval', action='store_true', default=False, help='Perform evaluation only')
parser.add_argument('--num_topk', type=int, default=8, help='number of topk for extract features')
parser.add_argument('--feat_path', default='', type=str, help='feature path')


# Model parameters
parser.add_argument('--model', default='SRViT', type=str, help='name of training model')
parser.add_argument('--pretrained', default=False, type=bool, help='use pre_trained model')

parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: 0.1)')

parser.add_argument('--model-ema', default=True, type=bool, help='exponential moving average of model weights')
parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
parser.add_argument('--model-ema-force-cpu', action='store_true', default=True, help='')

# Optimizer parameters
parser.add_argument('--amp', action='store_true', default=True,
                    help='using automatic mixed precision, is loss is nan, please set False!')
parser.add_argument('--sync-bn', action='store_true', default=True,
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
parser.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
parser.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "adamw"')
parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: 1e-8)')
parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=0.05,
                    help='weight decay (default: 0.05)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "cosine"')
parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                    help='learning rate (default: 5e-4)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 1e-6)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=20, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=5, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# some parameters for debug or other actions
parser.add_argument('--resume', default=None, help='resume from checkpoint')
parser.add_argument('--start_epoch', default=None, type=int, metavar='N', help='start epoch')

parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
parser.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
parser.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')

# distributed training parameters
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--distill', action='store_true', default=False, help='Enabling distributed evaluation')
parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-eval', action='store_true', default=False, help='Enabling distributed evaluation')


def _parse_args():
    # Do we have a config file to parse?
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

def main_folds():
    setup_default_logging()
    args, args_text = _parse_args()

    device = utils.init_distributed_mode(args)

    if args.rank == 0 and args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")

    # fix the random seed
    if args.seed is not None:
        random_seed(args.seed, utils.get_rank())

    val_auc = []

    # cross validation
    for fold in range(1, args.folds):
        csv_path = os.path.join(args.csv_path, 'fold_{}.csv'.format(fold))
        output_path = os.path.join(args.output_dir, str(fold))
        if output_path:
            Path(output_path).mkdir(parents=True, exist_ok=True)
            if args.rank == 0:
                _logger.info("************** FOLD (%d/%d) **************" % (fold + 1, args.folds))

        # Model settings
        if args.model == 'SRViT':
            model = SRViT(img_size=args.input_size, num_classes_i=args.num_classes_img, num_classes_t=args.num_classes_tk,
                           patch_size=args.patch_size, embed_dim=384, depth=9, num_heads=8, mlp_ratio=4., p_emb='4_2',
                           drop_path_rate=0., skip_lam=2., mix_token=True, return_dense=True, patch_shuffle=True)
        else:
            model = create_model(
                    args.model,
                    pretrained=args.pretrained,
                    num_classes=args.num_classes_img,
                    drop_rate=args.drop,
                    drop_path_rate=args.drop_path)

        if args.local_rank == 0:
            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            _logger.info('number of params: %d', n_parameters)

        # enable split bn (separate bn stats per batch-portion)
        num_aug_splits = 0
        if args.split_bn:
            model = convert_splitbn_model(model, max(num_aug_splits, 2))

        model.cuda()
        # setup synchronized BatchNorm for distributed training
        if args.distributed and args.sync_bn:
            assert not args.split_bn
            if has_apex and use_amp == 'apex':
                # Apex SyncBN preferred unless native amp is activated
                model = convert_syncbn_model(model)
            else:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

        # setup automatic mixed-precision (AMP) loss scaling and op casting
        amp_autocast = suppress
        if use_amp == 'apex':
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
            loss_scaler = ApexScaler()
        else:
            amp_autocast = torch.cuda.amp.autocast
            loss_scaler = NativeScaler()

        # resume checkpoint
        resume_epoch = None
        resume_path = os.path.join(args.resume, 'last.pth.tar')
        if args.resume:
            checkpoint = torch.load(os.path.join(args.resume, 'model_best.pth.tar'), map_location='cpu')
            max_accuracy = checkpoint['metric']
            best_epoch = checkpoint['epoch']
            resume_epoch = resume_checkpoint(
                model, resume_path,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                log_info=args.local_rank == 0)
        else:
            max_accuracy = 0.0
            best_epoch = 0

            # exponential moving average of model weights
        if args.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            model_ema = ModelEmaV2(
                model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
            if args.resume:
                load_checkpoint(model_ema.module, resume_path, use_ema=True)
        else:
            model_ema = None

        # distributed training
        if args.distributed:
            if use_amp == 'apex':
                model = ApexDDP(model, delay_allreduce=True)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

        # setup learning rate schedule and starting epoch
        lr_scheduler, num_epochs = create_scheduler(args, optimizer)
        start_epoch = 0
        if args.start_epoch is not None:
            # a specified start_epoch will always override the resume epoch
            start_epoch = args.start_epoch
        elif resume_epoch is not None:
            start_epoch = resume_epoch

        if lr_scheduler is not None and start_epoch > 0:
            lr_scheduler.step(start_epoch)

        # Dataset
        dataset_train = ImageDataset(args.data_path, img_size=args.input_size, mask_path=args.mask_path, patch_size=args.patch_size,
            df_path=csv_path, mil=args.mil, sp='train')

        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            sampler_train = udata.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)

        data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                        sampler=sampler_train,
                                                        batch_size=args.batch_size,
                                                        num_workers=args.num_workers,
                                                        pin_memory=args.pin_mem,
                                                        drop_last=False)

        criterion = CERCELoss(img_cls=args.num_classes_img, tk_cls=args.num_classes_tk).to(device=device)
        CEcriterion = torch.nn.CrossEntropyLoss().to(device=device)

        if args.eval:
            eval_path = os.path.join(output_path, 'model_best.pth.tar')
            _ = resume_checkpoint(
                model, eval_path,
                log_info=args.local_rank == 0)
            feat_path = os.path.join(args.feat_path, str(fold))
            if args.knn_eval:
                eval_stats = evalknn(model, csv_path, device, 0, args, _logger, amp_autocast=amp_autocast)
                auc = eval_stats["acc1"]
            else:
                auc = 0
            val_auc.append(auc)
            if feat_path:
                Path(feat_path).mkdir(parents=True, exist_ok=True)
            extract_features(model, csv_path, device, args, feat_path, _logger, amp_autocast=amp_autocast)

            if args.local_rank == 0:
                _logger.info(f"AUC of the network on the fold {fold} test dataset: {auc:.1f}% ")
            continue

        output_dir = None
        saver = None
        best_metric = None
        if args.rank == 0:
            output_dir = Path(output_path)
            saver = CheckpointSaver(
                model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
                checkpoint_dir=output_dir, recovery_dir=output_dir, max_history=args.checkpoint_hist)
            with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
                f.write(args_text)

        start_time = time.time()

        # early_stop seetings
        early_stop_counter = start_epoch - best_epoch
        patience = 40
        early_stop = False

        for epoch in range(start_epoch, args.epochs):
            if args.distributed and hasattr(data_loader_train.sampler, 'set_epoch'):
                data_loader_train.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(args, model, criterion, data_loader_train, saver, lr_scheduler,
                optimizer, device, epoch, loss_scaler, _logger, model_ema, mask=args.mask, amp_autocast=amp_autocast)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            if args.knn_eval:
                eval_stats = evalknn(model, csv_path, device, epoch, args, _logger, amp_autocast=amp_autocast)
            else:
                eval_stats = validate(model, csv_path, device, CEcriterion, args, _logger, amp_autocast=amp_autocast)

            max_accuracy = max(max_accuracy, eval_stats["acc1"])
            _logger.info(f"AUC in epoch {epoch} on the validation dataset: {eval_stats['acc1']:.1f}%, "
                         f"and the best AUC is {max_accuracy:.1f}%")

            lr_scheduler.step(epoch + 1)
            if output_dir is not None:
                update_summary(
                    epoch, train_stats, eval_stats, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_stats["acc1"]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

            if not max_accuracy == eval_stats["acc1"]:
                early_stop_counter += 1
                if early_stop_counter > patience:
                    early_stop = True
            else:
                early_stop_counter = 0

            if early_stop:
                val_auc.append(max_accuracy)
                break

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        del model
        torch.cuda.empty_cache()
        if args.local_rank == 0:
            _logger.info('Fold {} done, Training time {}'.format(fold+1, total_time_str))

    val_mean, val_std = np.mean(val_auc), np.std(val_auc)
    if args.local_rank == 0:
        _logger.info(f"validation mean(std): {val_mean:.2f} +/- {val_std:.2f}")


if __name__ == '__main__':
    main_folds()
