from ..base_model import BaseModel
from ..backbones.cswin import CSWinTransformer
from ..modules import SeparateBNNecks,SeparateFCs
import torch
import random

class MyModel(BaseModel):
    def __init__(self, cfgs, is_training):
        super().__init__(cfgs, is_training)

    def build_network(self, model_cfg):
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.backbone = CSWinTransformer(patch_size=1, embed_dim=16, depth=[4, 4, 2, 1],
                             split_size=[1, 1, 1, 1], num_heads=[2, 4, 8, 16], mlp_ratio=4.,drop_path_rate=0.)

        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
    def forward(self, inputs):
        ipts, labs, typs, viws, seqL = inputs
        sils = ipts[0]
        if len(sils.size()) == 4:
            # sils = sils.unsqueeze(2)
            n, t, h, w = sils.size()
            # sils = sils.view(n*t,h,w)
            # 随机选择select_num个子张量的索引
            # select_num = random.randint(1,96)
            # selected_indices = random.sample(range(96),k=select_num)
            #
            # # 对选择的select_num个子张量的0和1进行对调
            # for index in selected_indices:
            #     selected_tensor = sils[:, index, :, :]
            #     selected_tensor[selected_tensor == 0.] = 1.
            #     selected_tensor[selected_tensor == 1.] = 0.
            # 创建随机噪音张量，取值范围在[0, 1)
            # noise = torch.rand(n, t, h, w).cuda()
            #
            # center_h = (h - 1) / 2
            # center_w = (w - 1) / 2
            # # 创建噪音尺度张量，距离越接近边缘，尺度越大
            # h_distances = torch.arange(h).abs().unsqueeze(-1).expand(h, w).cuda()
            # w_distances = torch.arange(w).abs().unsqueeze(-2).expand(h, w).cuda()
            # # noise_scale = torch.min(h_distances, h - h_distances - 1, w_distances, w - w_distances - 1)
            # distances = torch.sqrt((h_distances - center_h) ** 2 + (w_distances - center_w) ** 2)
            # noise_scale = 1 / (1 + distances)
            # noise_scale = noise_scale.unsqueeze(0).unsqueeze(0)
            # # 添加噪音并修改尺度
            # inputs = sils + noise * noise_scale
            # sils = sils.reshape(n, t//3, 3, h, w)
            sils = sils.reshape(n, t, 1, h, w)

        del ipts
        n, t, c, h, w = sils.size()
        outs = self.backbone(sils)
        outs = outs.permute(0, 2, 1).contiguous()
        # outs = torch.mean(outs, dim=-1, keepdim=True)
        # embed_1 = outs
        embed_1 = self.FCs(outs)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
        embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': sils.view(n*t, 1, h, w)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval