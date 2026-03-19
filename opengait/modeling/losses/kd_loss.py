import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class KDLoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0):
        super(KDLoss, self).__init__(loss_term_weight)

    @gather_and_scale_wrapper
    def forward(self, new_logit, old_logit, T = 2.):
        if old_logit is None:
            loss = torch.tensor([0.0])
        else:
            """Compute knowledge-distillation (KD) loss given [scores] and [target_scores].

                    Both [scores] and [target_scores] should be tensors, although [target_scores] should be repackaged.
                    'Hyperparameter': temperature"""

            device = new_logit.device
            # print("new_logits:", new_logit[0])
            # print("old_logits:", old_logit[0])
            log_scores_norm = F.log_softmax(new_logit / T, dim=1)
            targets_norm = F.softmax(old_logit / T, dim=1)
            # print("log_scores_norm:", log_scores_norm[0])
            # print("targets_norm:", targets_norm[0])

            # if [scores] and [target_scores] do not have equal size, append 0's to [targets_norm]
            n = new_logit.size(1)
            if n > old_logit.size(1):
                n_batch = new_logit.size(0)
                zeros_to_add = torch.zeros(n_batch, n - old_logit.size(1))
                zeros_to_add = zeros_to_add.to(device)
                targets_norm = torch.cat([targets_norm.detach(), zeros_to_add], dim=1)

            # Calculate distillation loss (see e.g., Li and Hoiem, 2017)
            KD_loss_unnorm = -(targets_norm * log_scores_norm)
            KD_loss_unnorm = KD_loss_unnorm.sum(dim=1)  # --> sum over classes
            KD_loss_unnorm = KD_loss_unnorm.mean()  # --> average over batch
            # print(KD_loss_unnorm)
            # normalize
            loss = KD_loss_unnorm * T ** 2
        self.info.update({
            'loss': loss.detach().clone()})

        return loss, self.info

