import numpy as np
import torch
import torch.nn as nn

from ...utils import box_coder_utils, common_utils, loss_utils
from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner


class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)

        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.anchors = [x.cuda() for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        if losses_cfg.get('REG_VAR_LOSS_TYPE') == "NLL":
            reg_loss_name = 'NegativeLogLikelihood'
            kwargs = losses_cfg.get('REG_LOSS_ARGS', {})
            self.add_module(
                'reg_loss_func',
                getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'], **kwargs))
        elif losses_cfg.get('REG_VAR_LOSS_TYPE') == "EnergyScore":
            reg_loss_name = 'EnergyScore'
            kwargs = losses_cfg.get('REG_LOSS_ARGS', {})
            self.add_module(
                'reg_loss_func',
                getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'], **kwargs))
        else:
            reg_loss_name = 'WeightedSmoothL1Loss'
            self.add_module(
                'reg_loss_func',
                getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
            )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes
        )
        return targets_dict

    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]

        if self.model_cfg.VAR_OUTPUT_CLS == True:
            #get predictions generated by the model
            cls_var_preds = self.forward_ret_dict['cls_var_preds']
            cls_var_preds = cls_var_preds.view(batch_size, -1, self.num_class)

            num_samples = self.model_cfg.LOSS_CONFIG.NUM_SAMPLES_CLS_VAR_LOSS

            ## ---
            ## CODE BELOW IS TAKEN FROM ALI'S WORK on BAYESOD
            # Compute standard deviation
            pred_class_logits_var = torch.sqrt(torch.exp(cls_var_preds))

            # Produce normal samples using logits as the mean and the standard deviation computed above
            # Scales with GPU memory. 12 GB ---> 3 Samples per anchor for
            # COCO dataset.
            univariate_normal_dists = torch.distributions.normal.Normal(
                cls_preds, scale=pred_class_logits_var)

            pred_class_stochastic_logits = univariate_normal_dists.rsample(
                (num_samples,))
            pred_class_stochastic_logits = pred_class_stochastic_logits.view(
                (pred_class_stochastic_logits.shape[1] * num_samples, pred_class_stochastic_logits.shape[2], -1))
            pred_class_stochastic_logits = pred_class_stochastic_logits.squeeze(
                2)

            # Produce copies of the target classes to match the number of
            # stochastic samples.
            gt_classes_target = torch.unsqueeze(one_hot_targets, 0)
            gt_classes_target = torch.repeat_interleave(
                gt_classes_target, num_samples, dim=0).view(
                (gt_classes_target.shape[1] * num_samples, gt_classes_target.shape[2], -1))
            gt_classes_target = gt_classes_target.squeeze(2)

            # --- my code
             ##change the weight to reflect number of samples of stochastic run
            cls_weights = torch.unsqueeze(cls_weights, 0)
            cls_weights = torch.repeat_interleave(
                cls_weights, num_samples, dim=0).view(
                (cls_weights.shape[1] * num_samples, cls_weights.shape[2], -1))
            cls_weights = cls_weights.squeeze(2)

            # now can take the cross entropy
            cls_loss_src = self.cls_loss_func(pred_class_stochastic_logits, gt_classes_target, weights=cls_weights)  # [N, M]
            cls_loss = cls_loss_src.sum() / (batch_size * num_samples)

            cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        
        else:
            cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
            cls_loss = cls_loss_src.sum() / batch_size

            cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        ## variance 
        box_var_preds = self.forward_ret_dict.get('box_var_preds', None) # [N, H, W, C]
        batch_size = int(box_preds.shape[0])
        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])

        if box_var_preds is not None:
            # will be loglikelighood loss
            if self.model_cfg.LOSS_CONFIG.FULL_COVARIANCE:
                covariance_matrix_predictions = box_var_preds.view(batch_size, -1,
                                    box_var_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                    box_var_preds.shape[-1])
                #FULL COVARIANCE DOES NOT WORK
                print('FULL COVARIANCE DOES NOT WORK IN PYTHON')
                log_D = covariance_matrix_predictions

                # HERE I TRIED TO CREATE LOWER TRIANGULAR COVARIANCE MATRIX from a 28 (7*8/2) value
                # variance array in pytorch. However, I could not find a way to do it using 
                # pytorch (keras does have function implemented for this task)
                # that is probably why Ali Harakeh did not use a full covariance matrix in his review
                # of probabilistic object detectors  which utitlized pytorch (not keras)

                # # new matrix is used to make a covariance matrix from an array of variances network head returns
                # new_matrix_temp = torch.zeros((covariance_matrix_predictions.shape[0], covariance_matrix_predictions.shape[1], self.box_coder.code_size, self.box_coder.code_size)).to(covariance_matrix_predictions.get_device())

                # # we will convert an array consisting of variances (length of 28 for 7 parameters (N*(N+1)/2)) 
                # # into the lower triangular covariance matrix
                # for i in range(0, new_matrix_temp.shape[0]):
                #     for k in range(0, new_matrix_temp.shape[1]):
                #         temp_matrix = new_matrix_temp[i][k]
                #         temp_array = covariance_matrix_predictions[i][k]
                #         tril_indices = torch.tril_indices(row=7, col=7, offset=0)
                #         temp_matrix[tril_indices[0], tril_indices[1]] = temp_array
                #         new_matrix_temp[i][k] = temp_matrix

                # log_covariance = new_matrix_temp
                # #get cholensky decomposition
                # log_D = torch.diagonal(log_covariance, dim1=2, dim2=3)
                # print(log_D)

                #since full covariance does not work, we will stick with diagonal variance
            

                #multivariate nnl equation
                # loc_loss_src = 0.5 * torch.exp(-log_D) * self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)
                # loc_loss_src += 0.5 * log_D 
                # #normalize by batch size and multply by loss weight
                # loc_loss = loc_loss_src.sum() / batch_size
                # loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
                # box_loss = loc_loss
                # tb_dict = {
                #     'rpn_loss_loc': loc_loss.item()
                # }

            else:
                covariance_matrix_predictions = box_var_preds.view(batch_size, -1,
                                    box_var_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                    box_var_preds.shape[-1])

                loc_loss_src = self.reg_loss_func(mean=box_preds,
                                                var=covariance_matrix_predictions,
                                                target=box_reg_targets,
                                                weights=reg_weights)  # [N, M]
                loc_loss = loc_loss_src.sum() / batch_size
                
                loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
                box_loss = loc_loss
                tb_dict = {
                    'rpn_loss_loc': loc_loss.item()
                }
                
        else:
            # sin(a - b) = sinacosb-cosasinb
            box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
            loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
            loc_loss = loc_loss_src.sum() / batch_size

            loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
            box_loss = loc_loss
            tb_dict = {
                'rpn_loss_loc': loc_loss.item()
            }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None, box_var_preds=None, cls_var_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)
            box_var_preds: (N, H, W, C2)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)


        if box_var_preds == None:
            # no box variance
            batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
                if not isinstance(cls_preds, list) else cls_preds
            batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
                else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
            batch_box_preds, batch_box_var_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        else:
            # yes to box variance
            batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
                if not isinstance(cls_preds, list) else cls_preds
            batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
                else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
            

            # Attempt below to decode the box values and their variance by creating a distribution of means does not work
            # if we make a distribution, the tensopr becomes too large for gpu to handle (memory issues)
            # Thus we need to make a box decoder that will accept the variances and decode them as well instead of passing many distribution values to the decode
            # ------------------------------------
            # here we incoporate variance into results
            # batch_box_var_preds = box_var_preds.view(batch_size, num_anchors, -1) if not isinstance(box_var_preds, list) \
            #     else torch.cat(box_var_preds, dim=1).view(batch_size, num_anchors, -1)
            # diag_batch_box_var_preds = torch.diag_embed(batch_box_var_preds + 1e-4)
            # cholesky = torch.sqrt(diag_batch_box_var_preds)
            # mutivariate_normal_distribution = torch.distributions.multivariate_normal.MultivariateNormal(
            #     batch_box_preds,
            #     scale_tril=cholesky
            # )
            # num_samples = self.model_cfg.LOSS_CONFIG.LOSS_ARGS.num_samples
            # batch_box_preds = mutivariate_normal_distribution.rsample((num_samples, ))
            # # now we decode with variance incoroparated
            # batch_box_preds = self.box_coder.decode_torch(batch_box_preds,
            #     batch_anchors).view(num_samples, -1, self.box_coder.code_size)

            # batch_box_preds, batch_box_var_pred = compute_mean_covariance_torch(batch_box_preds)
            # batch_box_preds = batch_box_preds.view(batch_size, -1, self.box_coder.code_size)
            # batch_box_var_pred = batch_box_var_pred.view(batch_size, -1, self.box_coder.code_size, self.box_coder.code_size)
            # ------------------------------------
            batch_box_var_preds = box_var_preds.view(batch_size, num_anchors, -1) if not isinstance(box_var_preds, list) \
                else torch.cat(box_var_preds, dim=1).view(batch_size, num_anchors, -1)
            # batch_box_var_preds = torch.log(batch_box_var_preds) #maybe, not sure
            
            batch_box_preds, batch_box_var_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors, batch_box_var_preds)
            batch_box_var_preds = torch.diag_embed(batch_box_var_preds)

        if cls_var_preds == None:
            batch_cls_var_preds = None
        else: 
            batch_cls_var_preds = cls_var_preds.view(batch_size, num_anchors, -1).float() \
                if not isinstance(cls_var_preds, list) else cls_var_preds
            
        # ---------------------
        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )
        # -----------------

        # let's incoporate gaussian prior
        if box_var_preds != None and self.model_cfg.BAYESOD_CONFIG.use_gaussian_prior:
            if  self.model_cfg.BAYESOD_CONFIG.gaussian_prior.type == 'isotropic':
                gaussian_likelihood_precisions = torch.inverse(batch_box_var_preds)

                # ----------------------- initialize prior variables    
                # let's create a gaussian prior
                gaussian_prior_variance = torch.tensor([
                    self.model_cfg.BAYESOD_CONFIG.gaussian_prior.isotropic_variance]).to(box_var_preds.device)
                gaussian_prior_variance = gaussian_prior_variance.repeat(gaussian_likelihood_precisions.shape[2])
                gaussian_prior_variance = torch.diag_embed(gaussian_prior_variance)
                gaussian_prior_variance = gaussian_prior_variance.unsqueeze(0)
                gaussian_prior_variance = gaussian_prior_variance.repeat(gaussian_likelihood_precisions.shape[1], 1, 1)
                gaussian_prior_variance = gaussian_prior_variance.unsqueeze(0)
                gaussian_prior_variance = gaussian_prior_variance.repeat(gaussian_likelihood_precisions.shape[0], 1, 1, 1)
                # put in the dimensions we need for calculations B * enchors * C * 1 
                gaussian_prior_means = batch_box_preds[0].unsqueeze(0).unsqueeze(-1)
                #------------------------

                # Here gaussian posterior calculations from the paper begin
                # below dimensions B * enchors * C * C
                gaussian_prior_precision = torch.inverse(gaussian_prior_variance)

                gaussian_posterior_precisions = gaussian_likelihood_precisions + gaussian_prior_precision
                gaussian_posterior_covs = torch.inverse(gaussian_posterior_precisions)

                # calculate posterior means
                gaussian_likelihood_mean_weights = torch.matmul(gaussian_likelihood_precisions,
                                                                batch_box_preds.unsqueeze(-1))
                
                prior_mean_weights = torch.matmul(gaussian_prior_precision,
                                                gaussian_prior_means)
                
                intermediate_value = prior_mean_weights + gaussian_likelihood_mean_weights

                # calculate and put in dimestions batch_size, anchors, 

                gaussian_posterior_means = torch.matmul(gaussian_posterior_covs,
                                                        intermediate_value).squeeze(-1)

                # done with posterior calculations
                batch_box_preds = gaussian_posterior_means
                batch_box_var_preds = gaussian_posterior_covs

        return batch_cls_preds, batch_box_preds, batch_box_var_preds, batch_cls_var_preds

    def forward(self, **kwargs):
        raise NotImplementedError


# borrowed from Ali's original work on comparing probabilistic object detectors
def compute_mean_covariance_torch(input_samples):
    """
    Function for efficient computation of mean and covariance matrix in pytorch.
    Args:
        input_samples(list): list of tensors from M stochastic monte-carlo sampling runs, each containing
        N x k tensors.
    Returns:
        predicted_mean(Tensor): an Nxk tensor containing the predicted mean.
        predicted_covariance(Tensor): an Nxkxk tensor containing the predicted covariance matrix.
    """
    if isinstance(input_samples, torch.Tensor):
        num_samples = input_samples.shape[0]
    else:
        num_samples = len(input_samples)
        input_samples = torch.stack(input_samples, 2)

    # Compute Mean
    predicted_mean = torch.mean(input_samples, 0, keepdim=True)

    # Compute Covariance
    residuals = torch.transpose(
        torch.unsqueeze(
            input_samples -
            predicted_mean,
            2),
        2,
        3)
    predicted_covariance = torch.matmul(residuals, torch.transpose(residuals, 3, 2))
    predicted_covariance = torch.sum(predicted_covariance, 0) / (num_samples - 1)

    return predicted_mean.squeeze(0), predicted_covariance
