import torch
import torch.nn as nn
import torch.nn.functional as F

## Asymmetric Loss + F1 Loss
class MultiClassASLoss(nn.Module):
    '''
    MultiClass ASL(single label) + F1 Loss.
    
    References:
    - ASL paper: https://arxiv.org/abs/2009.14119
    - optimized ASL: https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
    '''
    def __init__(self, gamma_pos=1, gamma_neg=4, eps: float = 0.1, reduction='mean', average='macro'):
        super(MultiClassASLoss, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction
        self.average = average

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1) # make binary label

        targets = self.targets_classes
        anti_targets = 1. - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1. - xs_pos
        
        # TP / FP / FN
        tp = (xs_pos * targets).sum(dim=0)
        fp = (xs_pos * anti_targets).sum(dim=0)
        fn = (xs_neg * targets).sum(dim=0) 
        
        if self.average == 'micro':
            tp = tp.sum()
            fp = fp.sum()
            fn = fn.sum()
        
        # F1 score
        f1 = (tp / (tp + 0.5*(fp + fn) + self.eps)).clamp(min=self.eps, max=1.-self.eps).mean()
        
        # ASL weights
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        if self.gamma_pos > 0 or self.gamma_neg > 0:
            asymmetric_w = torch.pow(1. - xs_pos - xs_neg,
                                    self.gamma_pos * targets + self.gamma_neg * anti_targets)
            log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            num_classes = inputs.size()[-1]
            self.targets_classes = self.targets_classes.mul(1. - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss + (1. - f1)


## Focal Loss + F1 Loss
class MultiClassFocalLoss(nn.Module):
    '''
    MultiClass F1 Loss + FocalLoss.
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    - https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
    '''
    def __init__(self, eps=1e-8, average='macro', reduction='mean', gamma=2):
        super().__init__()
        self.eps = eps
        self.average = average
        self.reduction = reduction
        self.gamma = gamma
        
    def forward(self, pred, target):
        # focal loss
        loss = F.cross_entropy(pred, target, reduction=self.reduction)
        pt = torch.exp(-loss)
        if self.gamma>0:
            loss = (1-pt)**self.gamma * loss
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        
        # f1 loss
        target = F.one_hot(target, pred.size(-1)).float()
        pred = F.softmax(pred, dim=1)
        
        tp = (target * pred).sum(dim=0).float()
        fp = ((1 - target) * pred).sum(dim=0).float()
        fn = (target * (1 - pred)).sum(dim=0).float()

        if self.average == 'micro':
            tp, fp, fn = tp.sum(), fp.sum(), fn.sum()
        
        f1 = (tp / (tp + 0.5*(fp + fn) + self.eps)).clamp(min=self.eps, max=1-self.eps).mean()

        return 1. - f1 + loss


## MultiLabel Asymmetric Loss
class MultiLabelASLoss(nn.Module):
    '''
    MultiLabel ASL Loss + F1 Loss
    
    References:
    - ASL paper: https://arxiv.org/abs/2009.14119
    - optimized ASL: https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False, average='macro'):
        super(MultiLabelASLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps
        self.average = average

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid
        
        # TP / FP / FN
        tp = (xs_pos * y).sum(dim=0)
        fp = (xs_pos * (1-y)).sum(dim=0)
        fn = (xs_neg * y).sum(dim=0) 
        
        if self.average == 'micro':
            tp = tp.sum()
            fp = fp.sum()
            fn = fn.sum()
        
        # F1 score
        f1 = (tp / (tp + 0.5*(fp + fn) + self.eps)).clamp(min=self.eps, max=1.-self.eps).mean()

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean(dim=0).sum() + 1. - f1


## MultiLabel Binary Cross Entropy Loss
class MultiLabelBCELoss(nn.Module):
    """
    MultiLabel weighted F1_Loss + BCEWithLogitsLosss.
    """
    def __init__(self, eps=1e-8, average='macro', reduction='mean', weight=None, gamma=2):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)
        self.eps = eps
        self.average = average
        self.reduction = reduction
        self.weight = weight
        self.gamma = gamma
        if average not in ['macro', 'micro']:
            raise ValueError('average should be macro or micro.')
        
    def forward(self, pred, target): # same dimension
        bce_loss = self.bce(pred.float(), target.float())
        
        # f1 loss
        pred = F.softmax(pred, dim=1)
        
        tp = (target * pred).sum(dim=0).float()
        fp = ((1 - target) * pred).sum(dim=0).float()
        fn = (target * (1 - pred)).sum(dim=0).float()

        if self.average == 'micro':
            tp, fp, fn = tp.sum(), fp.sum(), fn.sum()
        
        f1 = tp / (tp + 0.5*(fp + fn) + self.eps).clamp(min=self.eps, max=1-self.eps)

        return 1 - f1.mean() + bce_loss