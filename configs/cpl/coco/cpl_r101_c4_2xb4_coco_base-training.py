_base_ = [
    '../../_base_/datasets/nway_kshot/base_coco_ms.py',
    '../../_base_/schedules/schedule.py', '../cpl_r101_c4.py',
    '../../_base_/default_runtime.py'
]

lr_config = dict(warmup_iters=1000, step=[92000])
evaluation = dict(interval=165000)
checkpoint_config = dict(interval=165000)
runner = dict(max_iters=165000)
optimizer = dict(lr=0.005)

# model settings
model = dict(
    roi_head=dict(bbox_head=dict(num_classes=60, num_meta_classes=60),
                  namuda=0.2, all_classes='ALL_CLASSES'),
)
