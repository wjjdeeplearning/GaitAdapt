import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class SPDLoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0):
        super(SPDLoss, self).__init__(loss_term_weight)

    @gather_and_scale_wrapper
    def forward(self, embeddings, old_embeddings):
        if old_embeddings is None:
            loss = torch.tensor([0.0])
        else:
            """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

                    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
                    'Hyperparameter': temperature"""
            N,C,P = embeddings.shape
            embeddings = embeddings.permute(
                2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]
            old_embeddings = old_embeddings.permute(
                2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]

            sim = embeddings.matmul(embeddings.transpose(1, 2)).view(P,-1)# [p, n*n]
            old_sim = old_embeddings.matmul(old_embeddings.transpose(1, 2)).view(P,-1) # [p, n*n]
            sim = F.normalize(sim, dim=-1)
            old_sim = F.normalize(old_sim, dim=-1)
            s_diff = old_sim - sim
            loss =((s_diff * s_diff).sum(dim=1, keepdim=False))/(N*N)



        self.info.update({
            'spd_loss': loss.detach().clone()})

        return loss, self.info

