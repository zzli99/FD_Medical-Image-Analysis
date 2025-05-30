from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class FreezeBackboneHook(Hook):
    def __init__(self, freeze_backbone=True):
        self.freeze_backbone = freeze_backbone

    def before_train_epoch(self, runner):
        if self.freeze_backbone:
            print(">> Freezing ViM backbone parameters")
            for name, param in runner.model.module.backbone.named_parameters():
                param.requires_grad = False
