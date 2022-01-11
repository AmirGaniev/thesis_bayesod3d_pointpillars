import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
import pickle
from torch.utils.data import DataLoader

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader, DistributedSampler
from pcdet.models import build_network
from pcdet.utils import common_utils

from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

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
    parser.add_argument('--test_set', action='store_true', default=False, help='Turn off test set eval')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    parser.add_argument('--pickle_file', type=str, default=None, help='pickle file to evaluate')
    parser.add_argument('--save_path', type=str, default=None, help='save path for pred.bin')

    args = parser.parse_args()

    if args.pickle_file is None:
        cfg_from_yaml_file(args.cfg_file, cfg)
        cfg.TAG = Path(args.cfg_file).stem
        cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg

def create_pred_bin(time_name_infos, det_annos, save_path):
    def limit_period(val, offset=0.5, period=np.pi):
        return val - np.floor(val / period + offset) * period
    
    str_to_class = {
        'Vehicle': label_pb2.Label.TYPE_VEHICLE,
        'Pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
        'Cyclist': label_pb2.Label.TYPE_CYCLIST
    }

    objects = metrics_pb2.Objects()

    for i in range(len(time_name_infos)):
        assert time_name_infos[i]['frame_id'] == det_annos[i]['frame_id']

        for j in range(len(det_annos[i]['boxes_lidar'])):
            o = metrics_pb2.Object()
            o.context_name = time_name_infos[i]['context_name']
            o.frame_timestamp_micros = time_name_infos[i]['timestamp_micros']

            pred_boxes = det_annos[i]['boxes_lidar']
            pred_boxes[:, -1] = limit_period(pred_boxes[:, -1], offset=0.5, period=np.pi * 2)

            # Populating box and score.
            box = label_pb2.Label.Box()
            box.center_x = pred_boxes[j][0]
            box.center_y = pred_boxes[j][1]
            box.center_z = pred_boxes[j][2]
            box.length = pred_boxes[j][3]
            box.width = pred_boxes[j][4]
            box.height = pred_boxes[j][5]
            box.heading = pred_boxes[j][6]
            o.object.box.CopyFrom(box)
            # This must be within [0.0, 1.0]. It is better to filter those boxes with
            # small scores to speed up metrics computation.
            o.score = det_annos[i]['score'][j]

            # Use correct type.
            o.object.type = str_to_class[det_annos[i]['name'][j]]

            objects.objects.append(o)

    # Write objects to a file.
    f = open(save_path, 'wb')
    f.write(objects.SerializeToString())
    f.close()
    print('the prediction bin file is saved at',save_path)


def main():
    args, cfg = parse_config()
    if args.test_set:
        time_name_infos_file = '../data/waymo/waymo_time_name_infos_test.pkl'
    else:
        time_name_infos_file = '../data/waymo/waymo_time_name_infos_val.pkl'

    time_name_infos = pickle.load(open(time_name_infos_file, 'rb'))

    if args.pickle_file:
        det_annos = pickle.load(open(args.pickle_file, 'rb'))
        create_pred_bin(time_name_infos, det_annos, args.save_path)
    else:
        if args.launcher == 'none':
            dist_test = False
            total_gpus = 1
        else:
            total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
                args.tcp_port, args.local_rank, backend='nccl'
            )
            dist_test = True

        if args.batch_size is None:
            args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
        else:
            assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
            args.batch_size = args.batch_size // total_gpus

        output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
        output_dir.mkdir(parents=True, exist_ok=True)

        eval_output_dir = output_dir / 'eval'
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / 'test'

        if args.eval_tag is not None:
            eval_output_dir = eval_output_dir / args.eval_tag

        eval_output_dir.mkdir(parents=True, exist_ok=True)
        log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

        # log to file
        logger.info('**********************Start logging**********************')
        gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
        logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

        if dist_test:
            logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
        for key, val in vars(args).items():
            logger.info('{:16} {}'.format(key, val))
        log_config_to_file(cfg, logger=logger)

        test_set, test_loader, sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=dist_test, workers=args.workers, logger=logger, training=False
        )
        if args.test_set:
            # Modify data path, sequence_list etc. to test set
            set_split_file = '../data/waymo/ImageSets/test.txt'
            test_set.sample_sequence_list = [x.strip() for x in open(set_split_file).readlines()]
            processed_data_tag = cfg.DATA_CONFIG.PROCESSED_DATA_TAG_TEST_SET if cfg.DATA_CONFIG.get('PROCESSED_DATA_TAG_TEST_SET', None) else 'waymo_processed_data_test_set'
            test_set.data_path = Path('../data/waymo/') / processed_data_tag
            test_set.infos = []
            test_set.include_waymo_data('test')

            if dist_test:
                rank, world_size = common_utils.get_dist_info()
                sampler = DistributedSampler(test_set, world_size, rank, shuffle=False)
            else:
                sampler = None

            test_loader = DataLoader(
                test_set, batch_size=args.batch_size, pin_memory=True, num_workers=args.workers,
                shuffle=False, collate_fn=test_set.collate_batch,
                drop_last=False, sampler=sampler, timeout=0
            )
            model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
            with torch.no_grad():
                # load checkpoint
                model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
                model.cuda()
                # start evaluation
                eval_dict = eval_utils.eval_one_epoch_test_set(
                    cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
                    result_dir=eval_output_dir, save_to_file=args.save_to_file)

            create_pred_bin(time_name_infos, eval_dict, args.save_path)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    main()