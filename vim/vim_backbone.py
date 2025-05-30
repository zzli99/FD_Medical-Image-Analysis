import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from vim.models_mamba import build_vim_model
from detectron2.layers import ShapeSpec

from mmdet.models import BACKBONES
from mmcv.runner import BaseModule
@BACKBONES.register_module()
class VimBackboneMM(BaseModule):
    def __init__(self, img_size=512, patch_size=8, in_chans=3, embed_dim=192, depth=24,
                 out_indices=[5, 11, 17, 23], use_checkpoint=False, pretrained=None, init_cfg=None):
        super().__init__(init_cfg)
        self.vim_backbone = VimBackbone(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            out_indices=out_indices,
            # use_checkpoint=use_checkpoint,
            # pretrained=pretrained
        )
        if pretrained:
            state_dict = torch.load(pretrained, map_location='cpu')
            self.vim_backbone.vim.load_state_dict(state_dict, strict=False)
            print("==> Pretrained weights loaded.")

    def forward(self, x):
        out = self.vim_backbone(x)
        return [out["res2"], out["res3"], out["res4"], out["res5"]]




class VimBackbone(nn.Module):
    """
    轻量化 ViM 编码器：用于分割任务，输出 4 个尺度特征。
    去除 mmcv 依赖，适配纯 PyTorch 和 Mask2Former。
    """
    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 24,
        out_indices: list = [5, 11, 17, 23],
        # use_checkpoint: bool = False,
        # pretrained: str = None,
    ):
        super().__init__()

        # 构建支持多尺度特征输出的 ViM 模型
        self.out_indices = out_indices
        self.img_size = img_size
        self.patch_size = patch_size
        self.vim = build_vim_model(
            img_size=img_size,
            patch_size=patch_size,
            stride=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            out_indices=out_indices,
            if_fpn=True,
            # use_checkpoint=use_checkpoint,
            # pretrained=pretrained,
            norm_layer=nn.LayerNorm,
            rms_norm=False,
        )
        # 改变WH, 128,64,32,16
        self.proj_res2 = nn.ConvTranspose2d(embed_dim, 192, 2, stride=2)
        self.proj_res3 = nn.Conv2d(embed_dim, 256, kernel_size=1)
        self.proj_res4 = nn.Conv2d(embed_dim, 384, kernel_size=3, stride=2, padding=1)
        self.proj_res5 = nn.Conv2d(embed_dim, 512, kernel_size=3, stride=4, padding=1)
        

    def forward(self, x):
        B = x.shape[0]
        H = W = self.img_size // self.patch_size
        # H, W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size

        # 获取中间层 token 输出序列列表，每个为 [B, N, C]
        token_features = self.vim(x, return_features=True)
        # print(f"source output: {len(token_features)} *", token_features[0].shape)

        feature_maps = []
        for feat in token_features:
            # 如果有 cls token，先排除掉
            if feat.shape[1] == H * W + 1:
                feat = feat[:, 1:, :]
            feat = feat.transpose(1, 2).reshape(B, -1, H, W)
            feature_maps.append(feat)
        # print(feature_maps[0].shape)
        # print(self.proj_res2(feature_maps[0]).shape)
        return {
            "res2": self.proj_res2(feature_maps[0]),
            "res3": self.proj_res3(feature_maps[1]),
            "res4": self.proj_res4(feature_maps[2]),
            "res5": self.proj_res5(feature_maps[3]),
        }
    
    def output_shape(self):
        """
        返回各尺度特征的 ShapeSpec，供 Detectron2 接口使用。
        """
        stride = self.patch_size
        return {
            "res2": ShapeSpec(channels=256, stride=stride),
            "res3": ShapeSpec(channels=256, stride=stride),
            "res4": ShapeSpec(channels=256, stride=stride),
            "res5": ShapeSpec(channels=256, stride=stride),
        }


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VimBackboneMM(pretrained='/cpfs01/projects-HDD/cfff-95eb48b12daa_HDD/lzz_24110240047/vim_maskrcnn/mmdetection-2.28.2/checkpoint_use3/epoch_3.pth').to(device)
    x = torch.randn(1, 3, 512, 512).to(device)
    # out = model(x)
    # for k, v in out.items():
    #     print(f"{k}: {v.shape}")
    out = model(x)
    for i, feat in enumerate(out):
        print(f"Level {i + 2}: {feat.shape}")
