import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class CRLLoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0):
        super(CRLLoss, self).__init__(loss_term_weight)

    @gather_and_scale_wrapper
    def forward(self, new_logit, old_logit):
        if old_logit is None:
            loss = torch.tensor([0.0])
        else:
            """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

                    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
                    'Hyperparameter': temperature"""

            device = new_logit.device
            # print("new_logits:", new_logit[0])
            # print("old_logits:", old_logit[0])
            N,C = old_logit.shape
            p = torch.zeros(N, C//2)
            q = torch.zeros(N, C // 2)
            for i in range(N):
                row = old_logit[i]
                row_new = new_logit[i]
                sort_index = torch.argsort(row, descending=True)
                top_index = sort_index[:C//2]
                top_value = row[top_index]
                new_top_value = row_new[top_index]
                p[i] = F.softmax(top_value, dim=0)
                q[i] = F.softmax(new_top_value, dim=0)


            result = torch.sum(p*torch.log(p)-p*torch.log(q),dim=1)
            result = result - 0.01*torch.sum(p*torch.log(p),dim=1)
            loss = F.relu(result)
            loss = loss.mean()

        self.info.update({
            'crl_loss': loss.detach().clone()})

        return loss, self.info

