import torch
from torch.nn import functional as F


class CrossEntropy(torch.nn.Module):

    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, x, target, coarse_anno, reduction=True):
        N_rep = x.shape[0]
        N = target.shape[0]
        if not N==N_rep:
            target = target.repeat(N_rep//N,1)
        x = F.softmax(x, dim=-1)
        x = torch.clamp(x, min=1e-6, max=1.0)
        x = torch.log(x)
        loss = torch.sum(-target * x, dim=-1)
        loss = loss * coarse_anno
        if reduction:
            return loss.mean()
        else:
            return loss


class CERCELoss(torch.nn.Module):

    def __init__(self, img_cls=3, tk_cls=4, RCE=True):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.base_criterion = CrossEntropy()
        self.token_weight = 0.5

        self.img_cls = img_cls
        self.tk_cls = tk_cls

        self.RCE = RCE

    def forward(self, outputs, labels, token_mask, coarse_anno, epoch):
        pred_cls, token_pred, idd, idx, _ = outputs
        B, N, C = token_pred.shape

        binary = (token_mask > 0).long()
        binary = binary.view(B, -1)[:, idx]

        lam = 1 - (torch.sum(binary[:, idd], dim=1) / N)
        lam = lam.unsqueeze(1)

        target_cls = F.one_hot(labels, self.img_cls).float().to(self.device)
        target_cls = lam * target_cls + (1 - lam) * target_cls.flip(0)
        cls_loss = self.base_criterion(pred_cls, target_cls, coarse_anno, reduction=True)

        token_mask_f = token_mask.reshape(-1)
        token_mask_h = F.one_hot(token_mask_f, self.tk_cls).float().to(self.device)
        token_mask_h = torch.clamp(token_mask_h, min=1e-6, max=1.0)

        token_pred = token_pred.reshape(-1, C)

        weight_ce = coarse_anno.unsqueeze(1).expand(B, N).reshape(-1)
        xentropy = self.base_criterion(token_pred, token_mask_h, weight_ce, reduction=True)

        if self.RCE:
            pred = F.softmax(token_pred, dim=1)
            pred = torch.clamp(pred, min=1e-7, max=1.0)
            rce = (-1 * torch.sum(pred * torch.log(token_mask_h ), dim=1))

            weight_rce = 1 / (10 * (1 + torch.exp(torch.tensor(-epoch))))

            token_loss = torch.mean(weight_ce * xentropy + weight_rce * rce)
        else:
            token_loss = torch.mean(weight_ce * xentropy)

        loss = cls_loss + self.token_weight * token_loss

        return loss


class focal_loss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)
        preds_softmax = torch.exp(preds_logsoft)

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
