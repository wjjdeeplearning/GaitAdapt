import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class STALoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0):
        super(STALoss, self).__init__(loss_term_weight)

    @gather_and_scale_wrapper
    def forward(self, old_vertex, new_vertex):
        if old_vertex is None:
            loss = torch.tensor([0.0])
        else:

            # old_vertex = old_vertex.to("cuda:0")
            # new_vertex = new_vertex.to("cuda:0")
            old_vertex = F.normalize(old_vertex)
            new_vertex = F.normalize(new_vertex)
            # print(old_vertex)
            loss = torch.mean(torch.sum((old_vertex - new_vertex).pow(2), dim=1, keepdim=False))
            # loss = torch.sum(torch.sum((old_vertex - new_vertex).pow(2), dim=1, keepdim=False))
            # loss = 10000*loss
            # print(loss)
        self.info.update({
            'loss': loss.detach().clone()})

        return loss, self.info

