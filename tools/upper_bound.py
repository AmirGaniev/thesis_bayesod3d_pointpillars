import numpy as np
import torch
import argparse
import pickle
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

def main():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--pred_infos', type=str, default=None, help='pickle file to evaluate')
    parser.add_argument('--new_file', type=str, default=None, help='new pickle file to save')
    parser.add_argument('--angle', action='store_true', default=False, help='Turn off setting perfect angles')
    parser.add_argument('--centroid', action='store_true', default=False, help='Turn off setting perfect centroids')
    parser.add_argument('--score', action='store_true', default=False, help='Turn off setting perfect scores')

    args = parser.parse_args()

    result_pkl = pickle.load(open(args.pred_infos, 'rb'))
    gt_pkl = pickle.load(open('../data/waymo/waymo_infos_val.pkl', 'rb'))
    if args.angle:
        new_pkl = gt_orientations(result_pkl,gt_pkl)
    elif args.centroid:
        new_pkl = gt_centroids(result_pkl,gt_pkl)
    elif args.score:
        new_pkl = perfect_scores(result_pkl,gt_pkl)
    else:
        raise NotImplemented
    
    # with open(args.new_file, 'wb') as f:
    #     pickle.dump(new_pkl, f)


def gt_orientations(result_pkl, gt_pkl):
    print(len(result_pkl))
    print(len(gt_pkl))
    assert len(result_pkl) == len(gt_pkl)
    new_pkl = []
    print(result_pkl[0].keys())
    print(gt_pkl[0].keys())
    return new_pkl


def gt_centroids(result_pkl, gt_pkl):
    assert len(result_pkl) == len(gt_pkl)
    new_pkl = []
    return new_pkl

def perfect_scores(result_pkl, gt_pkl):
    assert len(result_pkl) == len(gt_pkl)
    new_pkl = []
    return new_pkl


def calculate_iou(pred_boxes, gt_boxes):
    """
    Calculate IoU matrix (currently gt boxes have an additional dimension from data_dict, will fixed later)
    Args:
        pred_boxes (K, 7)
        gt_boxes (1, M, 7)

    Returns:
        iou_matrix (K, 1)
    """
    if not isinstance(pred_boxes, torch.Tensor):
        pred_boxes = torch.Tensor(pred_boxes).cuda()
    if not isinstance(gt_boxes, torch.Tensor):
        gt_boxes = torch.Tensor(gt_boxes).cuda()
    
    gt_boxes = gt_boxes[:,:,0:7].view(-1,7)
    pred_boxes = pred_boxes.view(-1,7)

    iou_matrix = boxes_iou3d_gpu(pred_boxes,gt_boxes).cpu().numpy()
    # TODO: add option for one-to-one prediction assignment
    # assign by max iou (many-to-one)
    max_ious = np.amax(iou_matrix, axis=1)
    max_iou_idx = np.argmax(iou_matrix, axis=1)
    return max_ious, max_iou_idx


if __name__ == '__main__':
    main()

