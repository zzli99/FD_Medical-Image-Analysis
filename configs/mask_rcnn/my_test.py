# configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py (标准 ResNet-50 配置)

_base_ = [
    # '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/models/vim_maskrcnn.py',
    '../_base_/datasets/my_coco_instance.py',
    '../_base_/schedules/my_schedule_2x.py',
    '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['hooks.freeze_hook'],  # 修改为 freeze_hook 的真实路径
    allow_failed_imports=False
)

# resume_from = '/cpfs01/projects-HDD/cfff-95eb48b12daa_HDD/lzz_24110240047/vim_maskrcnn/source_pth/vim_t_midclstok_76p1acc.pth'



# model = dict(
#     roi_head=dict(
#         bbox_head=dict(num_classes=1),
#         mask_head=dict(num_classes=1)
#     )
# )

# model = dict(
#     backbone=dict(
#         type='ResNet',
#         depth=50,
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
#         frozen_stages=1,
#         norm_cfg=dict(type='BN', requires_grad=True),
#         norm_eval=True,
#         style='pytorch',
#         init_cfg=dict(
#             type='Pretrained',
#             checkpoint='torchvision://resnet50')
#     ),
#     neck=dict(
#         type='FPN',
#         in_channels=[256, 512, 1024, 2048],
#         out_channels=256,
#         num_outs=5
#     ),
#     roi_head=dict(
#         bbox_head=dict(num_classes=80),
#         mask_head=dict(num_classes=80)
#     )
# )

# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=1
# )