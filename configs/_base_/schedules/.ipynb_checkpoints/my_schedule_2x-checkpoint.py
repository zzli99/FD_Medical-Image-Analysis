# # optimizer
# optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
# optimizer_config = dict(grad_clip=None)
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[16, 22]
#     # step=[1]
# )
# runner = dict(type='EpochBasedRunner', max_epochs=24)

# work_dir = './checkpoint'




optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(grad_clip=None)
# optimizer_config = dict(grad_clip=dict(max_norm=1.0, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,         # 减少热身迭代，避免前期爆炸
    warmup_ratio=0.001,
    step=[16, 22]
)

runner = dict(type='EpochBasedRunner', max_epochs=24)

work_dir = './checkpoint'

