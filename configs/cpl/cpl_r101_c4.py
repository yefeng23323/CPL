_base_ = [
    './meta-rcnn_r50_c4.py',
]
pretrained = 'open-mmlab://detectron2/resnet101_caffe'
# model settings
model = dict(
    type='CPL',
    post_rpn=True,
    pretrained=pretrained,
    backbone=dict(depth=101),
    roi_head=dict(
        type='CPLRoIHead',
        shared_head=dict(pretrained=pretrained),
        # bbox_head=dict(num_classes=20, num_meta_classes=20),
        feats_pooling=dict(
            type='FeatsPooling'),
        aware_esnhancement=dict(
            type='AwareEnhancement',
            dim=1024),
        bbox_head=dict(type='CPLBBoxHead', num_classes=20, num_meta_classes=20),
    )
)
