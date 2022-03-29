from fastai.vision.all import *

class FocalOHNM(Module):
    y_int = True
    def __init__(self):
        store_attr()
        self.hard_ratio = 6e-3
        self.ce_loss = CrossEntropyLossFlat()
        
    def focal_loss(self, preds, target, OHEM_percent=None):
        gamma = 2
        assert target.size() == preds.size()

        max_val = (-preds).clamp(min=0)
        loss = preds - preds * target + max_val + ((-max_val).exp() + (-preds - max_val).exp()).log()
        invprobs = F.logsigmoid(-preds * (target * 2 - 1))
        loss = (invprobs * gamma).exp() * loss

        if OHEM_percent is None:
            return loss.mean()
        else:
            OHEM, _ = loss.topk(k=int(15587 * OHEM_percent), dim=1, largest=True, sorted=True)
            return OHEM.mean()
        
    def bce_loss(self, preds, target, OHEM_percent=None):
        if OHEM_percent is None:
            loss = F.binary_cross_entropy_with_logits(preds, target, reduce=True)
            return loss
        else:
            loss = F.binary_cross_entropy_with_logits(preds, target, reduction='none')#, reduce=False)
            value, index= loss.topk(int(15587 * OHEM_percent), dim=1, largest=True, sorted=True)
            return value.mean()
        
    def forward(self, output, target):
        if not isinstance(output, tuple):
            return self.ce_loss(output, target)
        target_onehot = torch.zeros([len(target), 15587]).to(output[1].device)
        target_onehot.scatter_(1, target.view(-1, 1).long(), 1)
        
        # binary ce loss
        loss_0 = self.bce_loss(output[2], target_onehot, self.hard_ratio)
        
        ## focal loss
        loss_1 = self.focal_loss(output[2], target_onehot, self.hard_ratio)
        
        ### arcface loss
        loss_2 = self.ce_loss(output[0], target)
        
        return loss_0 + loss_1 + loss_2
        

class CombLoss(Module):
    y_int = True
    def __init__(self):
        store_attr()
        self.ce_loss = CrossEntropyLossFlat()
        self.triplet_loss = TripletLoss(margin=0.3)
        
    def forward(self, output, target):
        if not isinstance(output, tuple):
            return self.ce_loss(output, target)
        return self.ce_loss(output[0], target) + self.triplet_loss(output[1], target) 

def hard_example_mining(dist_mat, labels, return_inds=False):

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).T)
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).T)

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an

class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None):
        self.margin = margin

        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels):
        '''
            Calculate Tripletloss
            Arg:
            - Global feat needs to be normalized
            - labels, N * 1
        '''

        global_feat = F.normalize(global_feat)
        dist_mat = torch.cdist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        y = dist_an.new().resize_as_(dist_an).fill_(1)

        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)

        return loss