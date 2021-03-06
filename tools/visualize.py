import argparse
import glob
from pathlib import Path
from functools import partial

import mayavi.mlab as mlab
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate, build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

hooked = {}


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--testing', action='store_true', default=False, help='')
    parser.add_argument('--show_voxel_layers', nargs='+', default=[], help='Show overlapping voxels')
    parser.add_argument('--show_rois', action='store_true', default=False,
                        help='Visualize initial bounding box proposals. Only works for two-stage networks')
    parser.add_argument('--show_keypoints', action='store_true', default=False, help='Visualize keypoints. Only works for PV-RCNN')
    parser.add_argument('--show_iou', action='store_true', default=False, help='Visualize IoU between prediction and ground truth boxes')
    parser.add_argument('--index', type=int, default=None, help='Choose the index')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def hook_voxels_fn(m, i, o, layer=None):
    xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features = i
    new_features, ball_idxs = o
    voxel_idxs = ball_idxs.unique()
    voxels = xyz
    voxels_used = xyz[voxel_idxs.type(torch.int64)]

    if 'voxels' not in hooked:
        hooked['voxels'] = {}

    hooked['voxels'][layer] = {
        'all': voxels,
        'used': voxels_used
    }


def hook_rois_fn(m, i, o):
    batch_dict, = i
    hooked['rois'] = batch_dict['rois'][0]


def hook_keypoints_fn(m, i, o):
    xyz, xyz_batch_cnt, new_xyz, new_xyz_batch_cnt, features = i
    new_features, ball_idxs = o
    keypoints_idxs = ball_idxs.unique()
    keypoints = xyz[keypoints_idxs.type(torch.int64)]
    hooked['keypoints'] = {
        'all': xyz,
        'used': keypoints
    }


def find_voxels_in_bbox(voxels, ref_boxes):
    voxel_ids = roiaware_pool3d_utils.points_in_boxes_gpu(voxels.unsqueeze(0), ref_boxes.unsqueeze(0))
    return voxels[voxel_ids.squeeze() != -1]


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    # torch.manual_seed(0)
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=8,
        logger=logger,
        training=not args.testing
    )
    if args.ckpt:
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()

        # Register hooks
        if len(args.show_voxel_layers) > 0:
            # Convert list of strings to list of ints
            args.show_voxel_layers = [int(i) for i in args.show_voxel_layers]
            print('Showing Voxels')
            assert model.__class__.__name__ == 'PVRCNN'
            for layer in args.show_voxel_layers:
                model.pfe.SA_layers[layer].groupers[0].register_forward_hook(partial(hook_voxels_fn, layer=layer))
        if args.show_rois:
            print('Showing ROIs')
            model.roi_head.register_forward_hook(hook_rois_fn)
        if args.show_keypoints:
            print('Showing keypoints')
            assert model.__class__.__name__ == 'PVRCNN'
            model.roi_head.roi_grid_pool_layer.groupers[0].register_forward_hook(hook_keypoints_fn)

        with torch.no_grad():
            index = args.index if args.index is not None else np.random.choice(len(train_set))
            data_dict = train_set[index]
            data_dict = train_set.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            if len(args.show_voxel_layers) > 0:
                for layer_name, voxels in hooked['voxels'].items():
                    hooked['voxels'][layer_name]['in_bbox'] = find_voxels_in_bbox(voxels['all'], pred_dicts[0]['pred_boxes'])

            # Print out batch info
            print('index: {}, frame_id: {}, num_boxes: {}'.format(index, data_dict['frame_id'], len(data_dict['gt_boxes'][0])))

            # Default IoU value
            ious = None
            if args.show_iou:
                ious = V.calculate_iou(pred_dicts[0]['pred_boxes'],data_dict['gt_boxes'])

            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ious=ious, ref_labels=pred_dicts[0]['pred_labels'],
                gt_boxes=data_dict['gt_boxes'][0], voxels=hooked.get('voxels'),
                roi_boxes=hooked.get('rois'), keypoints=hooked.get('keypoints'),
                frustums=data_dict['frustums'][0] if 'frustums' in data_dict else []
            )
            mlab.show(stop=True)
    else:
        index = args.index if args.index is not None else np.random.choice(len(train_set))
        data_dict = train_set[index]
        data_dict = train_set.collate_batch([data_dict])

        print('index: {}, frame_id: {}'.format(index, data_dict['frame_id']))

        V.draw_scenes(points=data_dict['points'][:, 1:], gt_boxes=data_dict['gt_boxes'][0], frustums=data_dict['frustums'][0] if 'frustums' in data_dict else [])
        mlab.show(stop=True)

    logger.info('Vis done.')


if __name__ == '__main__':
    main()
