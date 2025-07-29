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
            ann_cfg=[dict(method='MetaRCNN', setting='SPLIT2_5SHOT')],
            num_novel_shots=5,
            num_base_shots=5,
            classes='ALL_CLASSES_SPLIT2',
        )),
    val=dict(classes='ALL_CLASSES_SPLIT2'),
    test=dict(classes='ALL_CLASSES_SPLIT2'),
    model_init=dict(classes='ALL_CLASSES_SPLIT2'))

evaluation = dict(
    interval=500, class_splits=['BASE_CLASSES_SPLIT2', 'NOVEL_CLASSES_SPLIT2'])
checkpoint_config = dict(interval=500)
optimizer = dict(lr=0.001)
lr_config = dict(warmup=None)
runner = dict(max_iters=2000)

# load_from = 'path of base training model'
load_from = \
    'work_dirs/cpl_r101_c4_2xb4_voc-split2_base-training/latest.pth'

# model settings
model = dict(
    frozen_parameters=['backbone', 'shared_head'],
    roi_head=dict(namuda=0.2, all_classes='ALL_CLASSES_SPLIT2')
)
