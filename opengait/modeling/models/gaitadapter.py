import torch
import torch.nn as nn
import copy
from ..base_model_lifelong import BaseModellifelong
from ..modules import SetBlockWrapper, GeMPoolingPyramid, PackSequenceWrapper, SeparateFCs
from ..expandable import ExpandHeads
from ..lifelongmodule import Gaitpart_Graph


class GaitAdapter(BaseModellifelong):

    def build_network(self, model_cfg):
        self.ResNetbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.ResNetbone = SetBlockWrapper(self.ResNetbone)
        self.Taskbone = TaskBone(model_cfg, self.ResNetbone)
        metagraph_cfg = model_cfg['graph_cfg']
        self.Graphbone = Gaitpart_Graph(hidden_dim=metagraph_cfg['hidden_dim'],
                                                 input_dim=metagraph_cfg['input_dim'],
                                                 sigma=2.0,
                                                 proto_graph_vertex_num=-1,# 废弃
                                                 meta_graph_vertex_num=metagraph_cfg['meta_graph_vertex_num'])



    def forward(self, inputs, old_task_model=None, old_graph_model=None):
        #数据处理
        ipts, labs, _, _, seqL = inputs
        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        del ipts
        # print(sils.shape)
        if old_task_model is None:
            feature_map,logits = self.Taskbone(sils, seqL, self.current_step)  # [n, c, p] [n, cls, p]
            protos,_ = self.Graphbone(feature_map.detach())
            feature_fuse = feature_map + protos
            old_feature_fuse = None
            new_logit = None
            old_vertex = None
            old_logit = None
        else:
            old_current_step = list(range(self.current_step))
            new_current_step = list(range(self.current_step+1))
            feature_map,logits_list = self.Taskbone(sils, seqL, new_current_step)
            protos,_ = self.Graphbone(feature_map.detach())
            feature_fuse = feature_map + protos
            logits = logits_list[-1]

            with torch.no_grad():
                old_feature_map,old_logits_list = old_task_model(sils, seqL, old_current_step)
                old_vertex = old_graph_model.meta_graph_vertex
                old_protos,_ = old_graph_model(old_feature_map)
                old_feature_fuse = old_feature_map + old_protos
            # del old_feature_map
            torch.cuda.empty_cache()
            new_logit = torch.cat(logits_list[:-1], dim=1) #[n, all_class_num_this_step, p]
            # new_logit = torch.cat(logits_list[:-1], dim=1)
            old_logit = torch.cat(old_logits_list, dim=1)#[n, all_class_num_last_step, p]

            n, cn, p = new_logit.shape
            new_logit = new_logit.transpose(-2,-1).contiguous().view(n*p, -1)
            old_logit = old_logit.transpose(-2, -1).contiguous().view(n * p, -1)
        embed = feature_fuse
        n, _, s, h, w = sils.size()
        if self.current_step>0:
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': feature_fuse, 'labels': labs},
                    'softmax': {'logits': logits, 'labels': labs},
                    'edsn': {'embeddings': feature_fuse, 'old_embeddings': old_feature_fuse, 'labels': labs},
                    'kd_loss': {'new_logit': new_logit, 'old_logit': old_logit},
                    'stability_loss': {'old_vertex': old_vertex, 'new_vertex': self.Graphbone.meta_graph_vertex}
                },
                'visual_summary': {
                    'image/sils': sils.view(n*s, 1, h, w)
                },
                'inference_feat': {
                    'embeddings': embed
                }
            }
        else:
            retval = {
                'training_feat': {
                    'triplet': {'embeddings': feature_fuse, 'labels': labs},
                    'softmax': {'logits': logits, 'labels': labs},
                },
                'visual_summary': {
                    'image/sils': sils.view(n * s, 1, h, w)
                },
                'inference_feat': {
                    'embeddings': embed
                }
            }

        return retval


class TaskBone(nn.Module):

    def __init__(self, model_cfg, ResNetbone):
        super(TaskBone, self).__init__()
        self.Backbone = ResNetbone
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.TP = PackSequenceWrapper(torch.max)
        self.gem = GeMPoolingPyramid(bin_num=model_cfg['bin_num'])
        self.eheads = ExpandHeads(**model_cfg['ExpandHeads'])

    def forward(self, sils, seqL, step_cls_list):

        outs = self.Backbone(sils)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        feat = self.gem(outs)  # [n, c, p]

        embed_1 = self.FCs(feat)  # [n, c, p]

        embed_2, logits = self.eheads(embed_1, step_cls_list)

        return embed_1, logits

