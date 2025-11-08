import os
import torch
import random
import numpy as np

class AverageMeter(object):
    """ Computes and stores the average and current value """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def set_seed(seed=0, msg=False):
    if msg: 
        print(f'Seed: {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_ckpt(epoch, network, model, optimizer, ncm_acc, filename):
    state = {
        'epoch': epoch,
        'network': network,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    if ncm_acc is not None:
        state['ncm_acc'] = ncm_acc

    torch.save(state, filename)

def load_ckpt(model, optimizer, ckpt_file):
    state = torch.load(ckpt_file)
    
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])

    return model, optimizer
