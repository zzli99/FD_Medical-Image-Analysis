# dataset settings
dataset_type = 'CocoDataset'
data_root = '/cpfs01/projects-HDD/cfff-95eb48b12daa_HDD/lzz_24110240047/vim_maskrcnn/vim_data/v0.3/'
img_norm_cfg = dict(
    # old1
    mean=[160.699, 150.677, 176.46],
    std=[32.927, 41.64, 18.29],
    # old2
    # mean=[134.881, 128.656, 138.36],
    # std=[30.981, 36.487, 27.453],
    to_rgb=True)

classes = ('cell',)
# pretrained='/cpfs01/projects-HDD/cfff-95eb48b12daa_HDD/lzz_24110240047/vim_maskrcnn/source_pth/vim_t_midclstok_76p1acc.pth'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(512,512), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=16,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'train/',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'train/',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/train.json',
        img_prefix=data_root + 'train/',
        classes=classes,
        pipeline=test_pipeline))
# evaluation = dict(metric=['bbox', 'segm'])
# evaluation = dict(metric=['segm'])
evaluation = dict(interval=100, metric=['segm'])
