from ..base_model import BaseModel
from ..backbones.cswin import CSWinTransformer
from ..modules import SeparateBNNecks
import torch
# 将cswin用作单张识别，准确率66-68,但速度慢，part设置为t

class MyModel2(BaseModel):
    def __init__(self, cfgs, is_training):
        super().__init__(cfgs, is_training)

    def build_network(self, model_cfg):
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.backbone = CSWinTransformer(patch_size=1, embed_dim=16, depth=[1, 1, 2, 1],
                             split_size=[1, 1, 2, 3], num_heads=[2, 4, 8, 16], mlp_ratio=4.)

    def forward(self, inputs):
        ipts, labs, typs, viws, seqL = inputs
        sils = ipts[0]
        if len(sils.size()) == 4:
            n, t, h, w = sils.size()
            sils = sils.unsqueeze(2)

            # sils = sils.reshape(n, t//3, 3, h, w)

        del ipts
        n, t, c, h, w = sils.size()
        outs = self.backbone(sils)
        outs =outs.view(n,t,128)
        outs = outs.permute(0, 2, 1).contiguous()
        embed_1 = outs
        # embed_1 = self.FCs(outs)  # [n, c, p]
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