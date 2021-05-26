import argparse
import pdb
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/nuscenes_models/cbgs_pp_multihead.yaml', help='specify the config for training')

    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--modality', type=str, default='radar')
    parser.add_argument('--sweep_version', type=str, default='version2')
    parser.add_argument('--max_sweeps', type=int, default=13)
    parser.add_argument('--class_names', type=list, 
                        default=['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                                 'barrier', 'motorcycle', 'bicycle', 'pedestrain', 'traffic_cone'])

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    flags = parser.parse_args()

    cfg_from_yaml_file(flags.cfg_file, cfg)
    cfg.TAG = Path(flags.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(flags.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if flags.set_cfgs is not None:
        cfg_from_list(flags.set_cfgs, cfg)

    return flags, cfg


def eval_single_ckpt(model, test_loader, flags, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    model.load_params_from_file(filename=flags.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()

    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=flags.save_to_file
    )


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, flags):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= flags.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def repeat_eval_ckpt(model, test_loader, flags, eval_output_dir, logger, ckpt_dir, dist_test=False):
    # evaluated ckpt record
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    # tensorboard log
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
    total_time = 0
    first_eval = True

    while True:
        # check whether there is checkpoint which is not evaluated
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, flags)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < flags.start_epoch:
            wait_second = 30
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, flags.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > flags.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()

        # start evaluation
        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        tb_dict = eval_utils.eval_one_epoch(
            cfg, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir, save_to_file=flags.save_to_file
        )

        if cfg.LOCAL_RANK == 0:
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, cur_epoch_id)

        # record this epoch which has been evaluated
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)

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

    # Re-define CFG w.r.t. num_class
    PREPARE = cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0]['PREPARE']
    SAMPLE_GROUPS = cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0]['SAMPLE_GROUPS']
    DENSE_HEAD = cfg.MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG
    PREPARE_list = []
    SAMPLE_GROUPS_list = []
    DENSE_HEAD_list = []
    for i in range(num_class):
        # DATA_AUGMENTOR
        item = PREPARE['filter_by_min_points'][i]
        name, num = item.split(':')
        if modality == 'radar':
            num = 1
        PREPARE_list.append
        PREPARE_list.append(':'.join([name, str(num)]))

        SAMPLE_GROUPS_list.append(SAMPLE_GROUPS[i])
        DENSE_HEAD_list.append(DENSE_HEAD[i])
    cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0]['PREPARE'] = {'filter_by_min_points': PREPARE_list}
    cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0]['SAMPLE_GROUPS'] = SAMPLE_GROUPS_list
    cfg.MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG = DENSE_HEAD_list

    if modality == 'radar':
        cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].NUM_POINT_FEATURES = 6
    return cfg


def main():
    flags, cfg = parse_config()
    cfg = cfg_update(flags, cfg)

    if flags.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % flags.launcher)(
            flags.tcp_port, flags.local_rank, backend='nccl'
        )
        dist_test = True

    if flags.batch_size is None:
        flags.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert flags.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        flags.batch_size = flags.batch_size // total_gpus

    output_dir = Path('/mnt/mnt/sdd/ysshin/nuscenes') / 'output' / cfg.TAG / flags.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not flags.eval_all:
        num_list = re.findall(r'\d+', flags.ckpt) if flags.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if flags.eval_tag is not None:
        eval_output_dir = eval_output_dir / flags.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    os.environ['CUDA_VISIBLE_DEVICES']="%d"%flags.gpu

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * flags.batch_size))
    for key, val in vars(flags).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = flags.ckpt_dir if flags.ckpt_dir is not None else output_dir / 'ckpt'

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=flags.batch_size,
        dist=dist_test, workers=flags.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    flags.eval_all = True
    with torch.no_grad():
        if flags.eval_all:
            repeat_eval_ckpt(model, test_loader, flags, eval_output_dir, logger, ckpt_dir, dist_test=dist_test)
        else:
            eval_single_ckpt(model, test_loader, flags, eval_output_dir, logger, epoch_id, dist_test=dist_test)


if __name__ == '__main__':
    main()
