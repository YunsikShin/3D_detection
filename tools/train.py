import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt
import pdb

import torch
import torch.distributed as dist
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/nuscenes_models/cbgs_pp_multihead.yaml', help='specify the config for training')

    parser.add_argument('--modality', type=str, default='radar')
    parser.add_argument('--sweep_version', type=str, default='version2')
    parser.add_argument('--max_sweeps', type=int, default=13)
    parser.add_argument('--class_names', type=list, 
                        default=['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                                 'barrier', 'motorcycle', 'bicycle', 'pedestrain', 'traffic_cone'])

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=8, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    flags = parser.parse_args()

    cfg_from_yaml_file(flags.cfg_file, cfg)
    cfg.TAG = Path(flags.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(flags.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if flags.set_cfgs is not None:
        cfg_from_list(flags.set_cfgs, cfg)

    return flags, cfg


def cfg_update(flags, cfg):
    class_names = flags.class_names
    num_class = len(class_names)

    modality = flags.modality
    max_sweeps = flags.max_sweeps
    sweep_version = flags.sweep_version

    dir_info_train = f'{modality}_infos_{sweep_version}_{max_sweeps}sweeps_train.pkl'
    dir_info_val = f'{modality}_infos_{sweep_version}_{max_sweeps}sweeps_val.pkl'
    dir_dbinfo = f'nuscenes_dbinfos_{modality}_{sweep_version}_{max_sweeps}sweeps_withvelo.pkl'

    cfg.DATA_CONFIG.MODALITY = modality
    cfg.DATA_CONFIG.MAX_SWEEPS = max_sweeps
    cfg.DATA_CONFIG.SWEEP_VERSION = sweep_version
    cfg.DATA_CONFIG.INFO_PATH = {'train': dir_info_train, 'test': dir_info_val}
    cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].DB_INFO_PATH = dir_dbinfo

    ## Re-define CFG w.r.t. num_class
    PREPARE = cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0]['PREPARE']
    #SAMPLE_GROUPS = cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0]['SAMPLE_GROUPS']
    #DENSE_HEAD = cfg.MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG
    PREPARE_list = []
    #SAMPLE_GROUPS_list = []
    #DENSE_HEAD_list = []
    for i in range(num_class):
        # DATA_AUGMENTOR
        item = PREPARE['filter_by_min_points'][i]
        name, num = item.split(':')
        if modality == 'radar':
            num = 1
        PREPARE_list.append
        PREPARE_list.append(':'.join([name, str(num)]))

    #    SAMPLE_GROUPS_list.append(SAMPLE_GROUPS[i])
    #    DENSE_HEAD_list.append(DENSE_HEAD[i])
    #cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0]['PREPARE'] = PREPARE_list
    #cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0]['SAMPLE_GROUPS'] = SAMPLE_GROUPS_list
    #cfg.MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG = DENSE_HEAD_list
    return cfg


def main():
    flags, cfg = parse_config()
    cfg = cfg_update(flags, cfg)
    if flags.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % flags.launcher)(
            flags.tcp_port, flags.local_rank, backend='nccl'
        )
        dist_train = True

    if flags.batch_size is None:
        flags.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert flags.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        flags.batch_size = flags.batch_size // total_gpus

    flags.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if flags.epochs is None else flags.epochs

    if flags.fix_random_seed:
        common_utils.set_random_seed(666)

    output_dir = Path('/mnt/mnt/ysshin/nuscenes') / 'output' / cfg.TAG / flags.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * flags.batch_size))
    for key, val in vars(flags).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (flags.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=flags.batch_size,
        dist=dist_train, workers=flags.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=flags.merge_all_iters_to_one_epoch,
        total_epochs=flags.epochs
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if flags.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
    if flags.pretrained_model is not None:
        model.load_params_from_file(filename=flags.pretrained_model, to_cpu=dist, logger=logger)

    if flags.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(flags.ckpt, to_cpu=dist, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=flags.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, flags.extra_tag))
    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=flags.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=flags.ckpt_save_interval,
        max_ckpt_save_num=flags.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=flags.merge_all_iters_to_one_epoch
    )

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, flags.extra_tag))

    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, flags.extra_tag))
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=flags.batch_size,
        dist=dist_train, workers=flags.workers, logger=logger, training=False
    )
    eval_output_dir = output_dir / 'eval' / 'eval_with_train'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    flags.start_epoch = max(flags.epochs - 10, 0)  # Only evaluate the last 10 epochs

    repeat_eval_ckpt(
        model.module if dist_train else model,
        test_loader, flags, eval_output_dir, logger, ckpt_dir,
        dist_test=dist_train
    )
    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, flags.extra_tag))


if __name__ == '__main__':
    main()
