# Copyright (c) OpenMMLab. All rights reserved.
import copy
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from mmcv.utils import ConfigDict
from mmdet.core import bbox2result, bbox2roi
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import StandardRoIHead
from torch import Tensor

from mmfewshot.detection.models.utils.aggregation_layer import build_aggregator
from mmdet.models.builder import build_head, build_neck
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as trans
from mmfewshot.detection.datasets.coco import COCO_SPLIT
from mmfewshot.detection.datasets.voc import VOC_SPLIT
from torchvision.transforms import InterpolationMode
from mmfewshot.detection.models.roi_heads.meta_rcnn_roi_head import MetaRCNNRoIHead
from embeddings.embeddings_utils import load_obj, map_indices_to_new_classes_clip, map_indices_to_new_classes
from torch.distributions import uniform, normal
import random

import pdb

class MCVAE(nn.Module):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dim: int) -> None:
        super(MCVAE, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_channels + 512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)

        self.decoder_input = nn.Sequential(nn.Linear(latent_dim + 512, hidden_dim))

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.Sigmoid()
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:

        z = self.decoder_input(z)
        z_out = self.decoder(z)
        return z_out

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu, std + mu
    
    def random_feature_masking(self, input, mask_ratio=0.5):
        """
        Randomly mask the input features by selecting a certain ratio of feature dimensions and setting them to zero.

        Parameters:
            input: (n, 2048) or (2048), input feature tensor.
            mask_ratio: Masking ratio between 0.0 and 1.0, representing the proportion of features to be masked.

        Returns:
            The feature tensor after masking.
        """
        if input.dim() == 1:
            N, C = 1, input.shape[0]  # n=1，2048
            input = input.unsqueeze(0)  # (1, 2048)
        else:
            # (n, 2048)
            N, C = input.shape
        mask_features = torch.rand(N, C).to(input.device) < mask_ratio 
        input[mask_features] = 0
        if N == 1:
            input = input.squeeze(0)
        return input

    def forward(self, input: Tensor, embdeing, **kwargs) -> List[Tensor]:    

        input = self.random_feature_masking(input)
        if input.dim() == 1:
            input = torch.cat((input, embdeing), dim=0)
        else:
            input = torch.cat((input, embdeing), dim=1)
        mu, log_var = self.encode(input)
        z, z_inv = self.reparameterize(mu, log_var)
        
        if input.dim() == 1:
            z = torch.cat((z, embdeing), dim=0)
        else:
            z = torch.cat((z, embdeing), dim=1)
        z_out = self.decode(z)
        return [z_out, z_inv, input, mu, log_var]

    def loss_function(self, input, rec, mu, log_var, kld_weight=0.00025) -> dict:
        recons_loss = F.mse_loss(rec, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss_vae': loss}


@HEADS.register_module()
class CPLRoIHead(StandardRoIHead):
    """
    Args:
        aggregation_layer (ConfigDict): Config of `aggregation_layer`.
    """

    def __init__(self,
                 aggregation_layer: Optional[ConfigDict] = None,
                 feats_pooling: Optional[ConfigDict] = None,
                 aware_esnhancement: Optional[ConfigDict] = None,
                 rpn_head_: Optional[ConfigDict] = None,
                 namuda=0.2,
                 all_classes=None,
                 **kwargs) -> None:
 
        super().__init__(**kwargs)

        assert feats_pooling is not None, "missing config of `feats_pooling`"
        self.feats_pooling = build_aggregator(copy.deepcopy(feats_pooling)) 
        assert aware_esnhancement is not None, "missing config of `aware_esnhancement`"
        self.aware_esnhancement = build_aggregator(copy.deepcopy(aware_esnhancement))       
        assert aggregation_layer is not None, 'missing config of `aggregation_layer`.'
        self.aggregation_layer = build_neck(copy.deepcopy(aggregation_layer))
        
        self.namuda = namuda
        self.all_classes = all_classes
        
        # vec
        if self.all_classes == 'ALL_CLASSES':
            self.embedings = load_obj('./clip_coco')
        else:
            self.embedings = load_obj('./clip_pascal')

        self.mcvae = MCVAE(2048, 2048, 2048)
        
        self.with_rpn = False
        if rpn_head_ is not None:
            self.with_rpn = True
            self.rpn_with_support = False
            self.rpn_head = build_head(rpn_head_)  #  build RPN head
            self.rpn_head_ = rpn_head_
        
        

    def forward_train(self,
                      query_feats: List[Tensor],
                      support_feats: List[Tensor],
                      proposals: List[Tensor],
                      query_img_metas: List[Dict],
                      query_gt_bboxes: List[Tensor],
                      query_gt_labels: List[Tensor],
                      support_gt_labels: List[Tensor],
                      query_gt_bboxes_ignore: Optional[List[Tensor]] = None,
                      **kwargs) -> Dict:
        """Forward function for training.
        Args:
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            support_feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
            proposals (list[Tensor]): List of region proposals with positive
                and negative pairs.
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            query_gt_bboxes (list[Tensor]): Ground truth bboxes for each
                query image, each item with shape (num_gts, 4)
                in [tl_x, tl_y, br_x, br_y] format.
            query_gt_labels (list[Tensor]): Class indices corresponding to
                each box of query images, each item with shape (num_gts).
            support_gt_labels (list[Tensor]): Class indices corresponding to
                each box of support images, each item with shape (1).
            query_gt_bboxes_ignore (list[Tensor] | None): Specify which
                bounding boxes can be ignored when computing the loss.
                Default: None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        # ([20, 1024, 14, 14])
        # ([6, 1024, 48, 68])

        # assign gts and sample proposals
        sampling_results = []   
        if not self.with_rpn:  
            if self.with_bbox:
                num_imgs = len(query_img_metas)
                if query_gt_bboxes_ignore is None:
                    query_gt_bboxes_ignore = [None for _ in range(num_imgs)]
                for i in range(num_imgs):  # dense detector, bbox assign task
                    assign_result = self.bbox_assigner.assign(
                        proposals[i], query_gt_bboxes[i],
                        query_gt_bboxes_ignore[i], query_gt_labels[i])
                    sampling_result = self.bbox_sampler.sample(
                        assign_result,
                        proposals[i],
                        query_gt_bboxes[i],
                        query_gt_labels[i],
                        feats=[lvl_feat[i][None] for lvl_feat in query_feats])
                    sampling_results.append(sampling_result)

        losses = dict()
        
        if self.with_bbox:
            bbox_results = self._optimized_bbox_forward_train(
                query_feats, support_feats, sampling_results, query_img_metas,
                query_gt_bboxes, query_gt_labels, support_gt_labels, query_gt_bboxes_ignore)
            if bbox_results is not None:
                losses.update(bbox_results['loss_bbox'])

        return losses

    def _optimized_bbox_forward_train(self, query_feats: List[Tensor],
                                      support_feats: List[Tensor],
                                      sampling_results: object,
                                      query_img_metas: List[Dict],
                                      query_gt_bboxes: List[Tensor],
                                      query_gt_labels: List[Tensor],
                                      support_gt_labels: List[Tensor],
                                      query_gt_bboxes_ignore: Optional[List[Tensor]] = None, ) -> Dict:
        """Forward function and calculate loss for box head in training.
        Args:
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            support_feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
            sampling_results (obj:`SamplingResult`): Sampling results.
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            query_gt_bboxes (list[Tensor]): Ground truth bboxes for each query
                image with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y]
                format.
            query_gt_labels (list[Tensor]): Class indices corresponding to
                each box of query images.
            support_gt_labels (list[Tensor]): Class indices corresponding to
                each box of support images.
        Returns:
            dict: Predicted results and losses.
        """

        if not self.with_rpn:
            query_rois = bbox2roi([res.bboxes for res in sampling_results])
            bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                      query_gt_bboxes,
                                                      query_gt_labels,
                                                      self.train_cfg)
            (labels, label_weights, bbox_targets, bbox_weights) = bbox_targets

        '''FFA'''
        support_gt_labels_ = torch.cat(support_gt_labels)
        unique = set(support_gt_labels_.cpu().numpy())
        num_classes = len(unique)
        # num_supp_shots = support_feats[0].size(0) // num_classes # 1
        support_feat = self.extract_support_feats(support_feats)[0]    # torch.Size([20, 2048])

        '''word_embedings'''
        #coco/voc vec
        if self.all_classes == 'ALL_CLASSES':
            emm_support_gt_labels_ = map_indices_to_new_classes_clip(COCO_SPLIT[self.all_classes], support_gt_labels_, self.all_classes)
        else:
            emm_support_gt_labels_ = map_indices_to_new_classes_clip(VOC_SPLIT[self.all_classes], support_gt_labels_, self.all_classes)
        emmbedings_voc = self.embedings[emm_support_gt_labels_].detach().to('cuda')
        '''mcvae'''
        support_feat_rec, support_feat_inv, _, mu, log_var = self.mcvae(support_feat, emmbedings_voc)
        # pdb.set_trace()

        pooling_feats = self.feats_pooling(support_feats[0], support_gt_labels=support_gt_labels_)

        fused_feats = self.aware_esnhancement(query_feats[0], pooling_feats)
      
        # ************ POST RPN *****************
        # RPN forward and loss 
        if self.with_rpn:
            if self.with_rpn:
                proposal_cfg = self.train_cfg.get('rpn_proposal', self.rpn_head_.test_cfg)
                if self.rpn_with_support:
                    raise NotImplementedError
                else:
                    
                    # stop gradient at RPN
                    query_feats_rpn = [x.detach() for x in [fused_feats]]
                    
                    rpn_losses, proposal_list = self.rpn_head.forward_train(
                        query_feats_rpn,
                        copy.deepcopy(query_img_metas),
                        copy.deepcopy(query_gt_bboxes),
                        gt_labels=None,
                        gt_bboxes_ignore=query_gt_bboxes_ignore,
                        proposal_cfg=proposal_cfg)
                proposals = proposal_list

                if self.with_bbox:
                    num_imgs = len(query_img_metas) # batchsize
                    if query_gt_bboxes_ignore is None:
                        query_gt_bboxes_ignore = [None for _ in range(num_imgs)]
                    for i in range(num_imgs):
                        assign_result = self.bbox_assigner.assign(
                            proposals[i], query_gt_bboxes[i],
                            query_gt_bboxes_ignore[i], query_gt_labels[i])
                        sampling_result = self.bbox_sampler.sample(
                            assign_result,
                            proposals[i],
                            query_gt_bboxes[i],
                            query_gt_labels[i],
                            feats=[lvl_feat[i][None] for lvl_feat in query_feats])
                        sampling_results.append(sampling_result)
            
            query_rois = bbox2roi([res.bboxes for res in sampling_results])  # 742 5
            bbox_targets = self.bbox_head.get_targets(sampling_results,
                                                      query_gt_bboxes,
                                                      query_gt_labels,
                                                      self.train_cfg)
            (labels, label_weights, bbox_targets, bbox_weights) = bbox_targets
        

        query_roi_feats_fuse = self.extract_query_roi_feat([fused_feats], query_rois) 
        query_roi_feats = self.extract_query_roi_feat(query_feats, query_rois)
        
        
        loss_bbox = {'loss_cls': [], 'loss_bbox': [], 'acc': [], 'loss_cls_sam': [], 'acc_sam': []}

        
        batch_size = len(query_img_metas)
        num_sample_per_imge = query_roi_feats.size(0) // batch_size #   [512, 2048] batch_size 4
        bbox_results = None
   
        for img_id in range(batch_size):
            start = img_id * num_sample_per_imge
            end = (img_id + 1) * num_sample_per_imge
            
            # random_index = np.random.choice(
            #     range(query_gt_labels[img_id].size(0)))
            # random_query_label = query_gt_labels[img_id][random_index]
            
            random_index = np.random.choice(range(len(support_gt_labels)))
            random_query_label = support_gt_labels[random_index]
            
            for i in range(support_feat.size(0)):
                # Following the official code, each query image only sample
                # one support class for training. Also the official code
                # only use the first class in `query_gt_labels` as support
                # class, while this code use random one sampled from
                # `query_gt_labels` instead. 
                if support_gt_labels[i] == random_query_label:
                    # bbox_results['cls_score'].shape torch.Size([122, 21])
                    # bbox_results['bbox_pred'].shape torch.Size([122, 80]) 4 x 20
                    
                    bbox_results = self._bbox_forward(
                        query_roi_feats[start:end],
                        support_feat_inv[i].sigmoid().unsqueeze(0)) 
                    
                    bbox_results_fuse = self._bbox_forward(
                        query_roi_feats_fuse[start:end],
                        support_feat_inv[i].sigmoid().unsqueeze(0)) 
                    
                    single_loss_bbox = self.bbox_head.loss(
                        bbox_results['cls_score'], bbox_results_fuse['bbox_pred'],
                        query_rois[start:end], labels[start:end],
                        label_weights[start:end], bbox_targets[start:end],
                        bbox_weights[start:end])
                    if 1 == 1:
                        valid_indices = (labels[start:end] != 20).nonzero(as_tuple=True)[0]
                        if len(valid_indices) > 1:
                            # pdb.set_trace()
                            # label_idxs =  labels[start + valid_indices]
                            label_idxs = torch.randint(0, support_feat.size(0), (max(len(valid_indices), 2),)).to('cuda')
                            support_sam = support_feat[label_idxs]
                            label_idxs = support_gt_labels_[label_idxs] # idx -> label
                            #coco/voc vec
                            if self.all_classes == 'ALL_CLASSES':
                                label_idxs_emmbed = map_indices_to_new_classes_clip(COCO_SPLIT[self.all_classes], label_idxs, self.all_classes)
                            else:
                                label_idxs_emmbed = map_indices_to_new_classes_clip(VOC_SPLIT[self.all_classes], label_idxs, self.all_classes)

                            emmbeding_sam = self.embedings[label_idxs_emmbed].to('cuda')
                            support_feat_rec_sam, _, _, _, _ = self.mcvae(support_sam, emmbeding_sam)
                            

                            # no gradient
                            # support_feat_rec_sam_no = support_feat_rec_sam.detach()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
                            # support_feat_inv_no = support_feat_inv.detach()
                            # pdb.set_trace()
                            
                            bbox_results_sam = self._bbox_forward(
                                support_feat_rec_sam,
                                support_feat_inv[i].sigmoid().unsqueeze(0))
                            
                            single_loss_bbox_sam = self.bbox_head.loss(
                                bbox_results_sam['cls_score'], None,
                                None, label_idxs,
                                torch.ones_like(label_idxs) * self.namuda, None,
                                None)  

                            single_loss_bbox_sam['loss_cls_sam'] = single_loss_bbox_sam.pop('loss_cls')
                            single_loss_bbox_sam['acc_sam'] = single_loss_bbox_sam.pop('acc')
                            
                            for key in single_loss_bbox_sam.keys():
                                loss_bbox[key].append(single_loss_bbox_sam[key])
                    
                    
                    for key in single_loss_bbox.keys():
                        loss_bbox[key].append(single_loss_bbox[key])
        
        # delete null key
        loss_bbox = {key: value for key, value in loss_bbox.items() if value}   
                     
        # pdb.set_trace()    
        if bbox_results is not None:
            for key in loss_bbox.keys():
                if key == 'acc':
                    loss_bbox[key] = torch.cat(loss_bbox['acc']).mean()
                elif key == 'acc_sam':
                    loss_bbox[key] = torch.cat(loss_bbox['acc_sam']).mean()
                else:
                    loss_bbox[key] = torch.stack(
                        loss_bbox[key]).sum() / batch_size
        
        # meta classification loss
        if self.bbox_head.with_meta_cls_loss:
            meta_cls_score = self.bbox_head.forward_meta_cls(support_feat_rec)  # support_feat [20,2048] -> meta_cls_score[20, 20]
            meta_cls_labels = torch.cat(support_gt_labels)  #[20]
            loss_meta_cls = self.bbox_head.loss_meta(
                meta_cls_score, meta_cls_labels,
                torch.ones_like(meta_cls_labels))
            loss_meta_cls['loss_meta_cls'] = loss_meta_cls['loss_meta_cls'] * 0
            loss_bbox.update(loss_meta_cls)
            

        loss_vae = self.mcvae.loss_function(
            support_feat, support_feat_rec, mu, log_var)
        loss_bbox.update(loss_vae)

        bbox_results.update(loss_bbox=loss_bbox)
        if self.with_rpn:
            bbox_results['loss_bbox'].update(rpn_losses)
        return bbox_results

    def extract_query_roi_feat(self, feats: List[Tensor],
                               rois: Tensor) -> Tensor:
        """Extracting query BBOX features, which is used in both training and
        testing.
        Args:
            feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W).
            rois (Tensor): shape with (bs*128, 5).
        Returns:
            Tensor: RoI features with shape (N, C).
        """
        roi_feats = self.bbox_roi_extractor(
            feats[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        return roi_feats

    def extract_support_feats(self, feats: List[Tensor]) -> List[Tensor]:
        """Forward support features through shared layers.
        Args:
            feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
        Returns:
            list[Tensor]: List of support features, each item
                with shape (N, C).
        """
        out = []
        if self.with_shared_head:
            for lvl in range(len(feats)):
                out.append(self.shared_head.forward_support(feats[lvl]))
        else:
            out = feats
        return out

    def _bbox_forward(self, query_roi_feats: Tensor,
                      support_roi_feats: Tensor) -> Dict:
        """Box head forward function used in both training and testing.

        Args:
            query_roi_feats (Tensor): Query roi features with shape (N, C).
            support_roi_feats (Tensor): Support features with shape (1, C).

        Returns:
             dict: A dictionary of predicted results.
        """
        
        # meta rcnn
        # feature aggregation
        # roi_feats = self.aggregation_layer(
        #     query_feat=query_roi_feats.unsqueeze(-1).unsqueeze(-1),
        #     support_feat=support_roi_feats.view(1, -1, 1, 1))[0]
        # cls_score, bbox_pred = self.bbox_head(roi_feats.squeeze(-1).squeeze(-1))
        # bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        # return bbox_results

        # feature aggregation
        roi_feats = self.aggregation_layer(
            query_feat=query_roi_feats.unsqueeze(-1).unsqueeze(-1),
            support_feat=support_roi_feats.view(1, -1, 1, 1))[0]
        cls_score, bbox_pred = self.bbox_head(
            roi_feats.squeeze(-1).squeeze(-1), query_roi_feats)
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results

    def _bbox_forward_without_agg(self, query_roi_feats: Tensor) -> Dict:
        """Box head forward function used in both training and testing.
        Args:
            query_roi_feats (Tensor): Query roi features with shape (N, C).
        Returns:
             dict: A dictionary of predicted results.
        """
        cls_score, bbox_pred = self.bbox_head(query_roi_feats)
        bbox_results = dict(cls_score=cls_score, bbox_pred=bbox_pred)
        return bbox_results

    def simple_test(self,
                    query_feats: List[Tensor],
                    support_feats_dict: Dict,
                    new_support_feats_dict: Dict,
                    proposal_list: List[Tensor],
                    query_img_metas: List[Dict],
                    rescale: bool = False) -> List[List[np.ndarray]]:
        """Test without augmentation.
        Args:
            query_feats (list[Tensor]): Features of query image,
                each item with shape (N, C, H, W).
            support_feats_dict (dict[int, Tensor]) Dict of support features
                used for inference only, each key is the class id and value is
                the support template features with shape (1, C).
            new_support_feats_dict: {'prototype': {cls_id: }, ..}.
            proposal_list (list[Tensors]): list of region proposals.
            query_img_metas (list[dict]): list of image info dict where each
                dict has: `img_shape`, `scale_factor`, `flip`, and may also
                contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            rescale (bool): Whether to rescale the results. Default: False.
        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'
        det_bboxes, det_labels = self.simple_test_bboxes(
            query_feats,
            support_feats_dict,
            new_support_feats_dict,
            query_img_metas,
            proposal_list,
            self.test_cfg,
            rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        return bbox_results

    def simple_test_bboxes(
            self,
            query_feats: List[Tensor],
            support_feats_dict: Dict,
            new_support_feats_dict: Dict,
            query_img_metas: List[Dict],
            proposals: List[Tensor],
            rcnn_test_cfg: ConfigDict,
            rescale: bool = False) -> Tuple[List[Tensor], List[Tensor]]:
        """Test only det bboxes without augmentation.
        Args:
            query_feats (list[Tensor]): Features of query image,
                each item with shape (N, C, H, W).
            support_feats_dict (dict[int, Tensor]) Dict of support features
                used for inference only, each key is the class id and value is
                the support template features with shape (1, C).
            new_support_feats_dict:
            query_img_metas (list[dict]): list of image info dict where each
                dict has: `img_shape`, `scale_factor`, `flip`, and may also
                contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
        Returns:
            tuple[list[Tensor], list[Tensor]]: Each tensor in first list
                with shape (num_boxes, 4) and with shape (num_boxes, )
                in second list. The length of both lists should be equal
                to batch_size.
        """
        img_shapes = tuple(meta['img_shape'] for meta in query_img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in query_img_metas)
       
        if not self.with_rpn:
            rois = bbox2roi(proposals)
            num_rois = rois.size(0)

        '''FFA'''
        num_classes = len(support_feats_dict)
        # num_classes = self.bbox_head.num_classes 
        support_feat = torch.cat([support_feats_dict[i] for i in range(num_classes)])   # [15 + 5, 2048]
        pooling_feats = new_support_feats_dict['pooling_feats'][0]      # [15, 5, 1024] 
        
        ''' word emmbeding'''
        if self.all_classes == 'ALL_CLASSES':
            emm_support_gt_labels_ = map_indices_to_new_classes_clip(COCO_SPLIT[self.all_classes], list(range(num_classes)), self.all_classes)
        else:
            emm_support_gt_labels_ = map_indices_to_new_classes_clip(VOC_SPLIT[self.all_classes], list(range(num_classes)), self.all_classes)
        # emmbedings_voc = torch.tensor(emmbedings[emm_support_gt_labels_], device='cuda')
        emmbedings_voc = self.embedings[emm_support_gt_labels_].detach().to('cuda')
        # pdb.set_trace()
        support_feat_rec, support_feat_inv, _, mu, log_var = self.mcvae(support_feat, emmbedings_voc)
        fused_feats = self.aware_esnhancement(query_feats[0], pooling_feats, query_img_metas)

        # **** POST RPN ****
        if self.with_rpn:
            proposals = self.rpn_head.simple_test([fused_feats], query_img_metas)
            rois = bbox2roi(proposals)                                             
            num_rois = rois.size(0)                                                
        # *************
            
        query_roi_feats_fuse = self.extract_query_roi_feat([fused_feats], rois) 
        query_roi_feats = self.extract_query_roi_feat(query_feats, rois)


        
        cls_scores_dict, bbox_preds_dict = {}, {}
        for class_id in support_feats_dict.keys():
            support_feat = support_feats_dict[class_id]
            
            bbox_results = self._bbox_forward(query_roi_feats, support_feat_inv[class_id].sigmoid().unsqueeze(0))

            bbox_results_fuse = self._bbox_forward(query_roi_feats_fuse, support_feat_inv[class_id].sigmoid().unsqueeze(0))
            bbox_results['bbox_pred'] = bbox_results_fuse['bbox_pred']
            
            cls_scores_dict[class_id] = \
                bbox_results['cls_score'][:, class_id:class_id + 1]
            bbox_preds_dict[class_id] = \
                bbox_results['bbox_pred'][:, class_id * 4:(class_id + 1) * 4]
            # the official code use the first class background score as final
            # background score, while this code use average of all classes'
            # background scores instead.
            if cls_scores_dict.get(num_classes, None) is None:
                cls_scores_dict[num_classes] = \
                    bbox_results['cls_score'][:, -1:]
            else:
                cls_scores_dict[num_classes] += \
                    bbox_results['cls_score'][:, -1:]

        cls_scores_dict[num_classes] /= len(support_feats_dict.keys())  #
        cls_scores = [
            cls_scores_dict[i] if i in cls_scores_dict.keys() else
            torch.zeros_like(cls_scores_dict[list(cls_scores_dict.keys())[0]])
            for i in range(num_classes + 1)
        ]
        bbox_preds = [
            bbox_preds_dict[i] if i in bbox_preds_dict.keys() else
            torch.zeros_like(bbox_preds_dict[list(bbox_preds_dict.keys())[0]])
            for i in range(num_classes)
        ]
        
        cls_score = torch.cat(cls_scores, dim=1)  # tensor(141,21)    [87,16]
        bbox_pred = torch.cat(bbox_preds, dim=1)  # tensor(141,80)    [87, 60]

        # split batch bbox prediction back to each image
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
        
        # apply bbox post-processing to each image individually
        # after nms？
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels
