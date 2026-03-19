import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class EDSNLoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0):
        super(EDSNLoss, self).__init__(loss_term_weight)

    @gather_and_scale_wrapper
    def forward(self, embeddings, old_embeddings, labels):
        # embeddings: [n, c, p], label: [n]
        if old_embeddings is None:
            edsn_loss = torch.tensor([0.0])
        else:
            embeddings = embeddings.permute(
                2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]

            ref_embed, ref_label = embeddings, labels
            dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n1, n2]

            an_dist = self.Convert2Triplets(labels, ref_label, dist) #[p, n, n_an]
            dist_norm = F.softmax(an_dist,dim=-1)#[p, n, n_an]

            # old_embeddings: [n, c, p], label: [n]
            old_embeddings = old_embeddings.permute(
                2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]

            old_ref_embed, ref_label = old_embeddings, labels
            old_dist = self.ComputeDistance(old_embeddings, old_ref_embed)  # [p, n1, n2]

            old_an_dist = self.Convert2Triplets(labels, ref_label, old_dist)  # [p, n, n_an]
            old_dist_norm = F.softmax(old_an_dist, dim=-1)  # [p, n, n_an]

            p, n, n_an = old_dist_norm.shape
            edsn_loss = (old_dist_norm.clamp(min=1e-4) * (old_dist_norm.clamp(min=1e-4)
                      /dist_norm.clamp(min=1e-4)).log()).sum(dim=(1, 2))/n

        self.info.update({
            'edsn_loss': edsn_loss.detach().clone()})

        return edsn_loss, self.info

    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeDistance(self, x, y):
        """
            x: [p, n_x, c]
            y: [p, n_y, c]
        """
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist

    def Convert2Triplets(self, row_labels, clo_label, dist):
        """
            row_labels: tensor with size [n_r]
            clo_label : tensor with size [n_c]
        """
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).bool()  # [n_r, n_c]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        p, n, _ = dist.size()
        an_dist = dist[:, diffenc].view(p, n, -1)
        return an_dist
