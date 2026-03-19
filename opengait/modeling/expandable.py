import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import clones, is_list_or_tuple
from torchvision.ops import RoIAlign



class ExpandHeads(nn.Module):

    def __init__(self, parts_num, in_channels, class_num_list, norm=True, parallel_BN1d=True):
        super(ExpandHeads, self).__init__()
        self.p = parts_num
        self.class_num_list = class_num_list
        self.norm = norm
        self.fc_bin = nn.ParameterDict()
        for step, class_num in enumerate(self.class_num_list):
            self.fc_bin[f'step:{step}'] = nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(parts_num, in_channels, class_num)))
        if parallel_BN1d:
            self.bn1d = nn.BatchNorm1d(in_channels * parts_num)
        else:
            self.bn1d = clones(nn.BatchNorm1d(in_channels), parts_num)
        self.parallel_BN1d = parallel_BN1d

    def forward(self, x, current_step = 0):
        """
            x: [n, c, p]
        """
        if self.parallel_BN1d:
            n, c, p = x.size()
            x = x.view(n, -1)  # [n, c*p]
            x = self.bn1d(x)
            x = x.view(n, c, p)
        else:
            x = torch.cat([bn(_x) for _x, bn in zip(
                x.split(1, 2), self.bn1d)], 2)  # [p, n, c]
        feature = x.permute(2, 0, 1).contiguous()
        if isinstance(current_step, list):
            logits_list = []
            if self.norm:
                feature = F.normalize(feature, dim=-1)  # [p, n, c]
                for c_s in current_step:
                    logits = feature.matmul(F.normalize(self.fc_bin[f'step:{c_s}'], dim=1))  # [p, n, c]
                    logits = logits.permute(1, 2, 0).contiguous()
                    logits_list.append(logits)
            else:
                for c_s in current_step:
                    logits = feature.matmul(self.fc_bin[f'step:{c_s}'])
                    logits = logits.permute(1, 2, 0).contiguous()
                    logits_list.append(logits)
            return feature.permute(1, 2, 0).contiguous(), logits_list
        else:
            if self.norm:
                feature = F.normalize(feature, dim=-1)  # [p, n, c]
                logits = feature.matmul(F.normalize(
                    self.fc_bin[f'step:{current_step}'], dim=1))  # [p, n, c]
            else:
                logits = feature.matmul(self.fc_bin[f'step:{current_step}'])
            return feature.permute(1, 2, 0).contiguous(), logits.permute(1, 2, 0).contiguous()


