import numpy as np
import torch
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingle(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.VAR_OUTPUT_REG:
            #number of elements requires to describe an N*N covariance matrix is computes as (N*(N+1)) / 2
            #in our case N = 7 (seven parameters to describe the bounding box)
            # we will only implement diagonal here (not full covariance)
            self.conv_var = nn.Conv2d(
                input_channels, self.num_anchors_per_location * self.box_coder.code_size,
                kernel_size=1
            )

            self.logvar_max = self.model_cfg.get("LOGVAR_MAX", None)

            ## for full covariance approach (does not work in pytorch)
            # self.conv_var = nn.Conv2d(
            #     input_channels, self.num_anchors_per_location * 28,
            #     kernel_size=1
            # )
        if self.model_cfg.VAR_OUTPUT_CLS:
            self.conv_cls_var = nn.Conv2d(
                input_channels, self.num_anchors_per_location * self.num_class,
                kernel_size=1
            )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
        if self.model_cfg.VAR_OUTPUT_REG:
            nn.init.normal_(self.conv_var.weight, mean=0, std=0.001)
        if self.model_cfg.VAR_OUTPUT_CLS:
            nn.init.normal_(self.conv_cls_var.weight, mean=0, std=0.01)
            nn.init.constant_(self.conv_cls_var.bias, -10.0)


    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
 
        # variance probablities
        if self.model_cfg.VAR_OUTPUT_REG:
            box_var_preds = self.conv_var(spatial_features_2d)
            box_var_preds = box_var_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
            if self.logvar_max is not None:
                box_var_preds = torch.clamp(box_var_preds, max=self.logvar_max)
            box_var_preds = torch.exp(box_var_preds)
            self.forward_ret_dict['box_var_preds'] = box_var_preds
        else:
            box_var_preds = None

        # classification probabilities
        if self.model_cfg.VAR_OUTPUT_CLS:
            cls_var_preds = self.conv_cls_var(spatial_features_2d)
            cls_var_preds = cls_var_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
            self.forward_ret_dict['cls_var_preds'] = cls_var_preds
    

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
