_base_ = [
    '../../../_base_/datasets/nway_kshot/base_voc_ms.py',
    '../../../_base_/schedules/schedule.py', '../../cpl_r101_c4.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        save_dataset=False,
        dataset=dict(classes='BASE_CLASSES_SPLIT3'),
        support_dataset=dict(classes='BASE_CLASSES_SPLIT3')),
    val=dict(classes='BASE_CLASSES_SPLIT3'),
    test=dict(classes='BASE_CLASSES_SPLIT3'),
    model_init=dict(classes='BASE_CLASSES_SPLIT3'))

lr_config = dict(warmup_iters=500, step=[17000])
evaluation = dict(interval=30000)
checkpoint_config = dict(interval=30000)
runner = dict(max_iters=30000)
optimizer = dict(lr=0.005)

# model settings
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=15, num_meta_classes=15),
                  namuda=0.5, all_classes='ALL_CLASSES_SPLIT3',))