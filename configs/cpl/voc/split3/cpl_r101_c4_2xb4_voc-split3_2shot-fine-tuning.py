_base_ = [
    '../../../_base_/datasets/nway_kshot/few_shot_voc_ms.py',
    '../../../_base_/schedules/schedule.py', '../../cpl_r101_c4.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        save_dataset=True,
        dataset=dict(
            type='FewShotVOCDefaultDataset',
            ann_cfg=[dict(method='MetaRCNN', setting='SPLIT3_2SHOT')],
            num_novel_shots=2,
            num_base_shots=2,
            classes='ALL_CLASSES_SPLIT3',
        )),
    val=dict(classes='ALL_CLASSES_SPLIT3'),
    test=dict(classes='ALL_CLASSES_SPLIT3'),
    model_init=dict(classes='ALL_CLASSES_SPLIT3'))
evaluation = dict(
    interval=300, class_splits=['BASE_CLASSES_SPLIT3', 'NOVEL_CLASSES_SPLIT3'])
checkpoint_config = dict(interval=300)
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None)
runner = dict(max_iters=1200)
# load_from = 'path of base training model'
load_from = \
    'work_dirs/cpl_r101_c4_2xb4_voc-split3_base-training/latest.pth'
# model settings
model = dict(
    frozen_parameters=['backbone', 'shared_head'],
    roi_head=dict(namuda=0.2, all_classes='ALL_CLASSES_SPLIT3')
)
