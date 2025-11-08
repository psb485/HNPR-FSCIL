import torch
from torch import nn

class LossFactory(nn.Module):
    def __init__(self, loss_type='cosface', eps=1e-7, s=None, m=None):
        ''' Angular Margin Loss (CosFace, https://arxiv.org/abs/1801.05599) '''
        super(LossFactory, self).__init__()
        self.loss_type = loss_type
        
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        elif loss_type == 'cross_entropy':
            self.ce = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError

    def forward(self, wf, labels):
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
            excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
            denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
            L = numerator - torch.log(denominator)            
            return -torch.mean(L)
        elif self.loss_type == 'cross_entropy':
            return self.ce(wf, labels)
        else:
            raise NotImplementedError