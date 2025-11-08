import os
import re
import sys
import errno
import torch
import shutil
import warnings
import tempfile
import hashlib
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from urllib.parse import urlparse
from urllib.request import urlopen

ENV_TORCH_HOME = 'TORCH_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'
HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def _download_url_to_file(url, dst, hash_prefix, progress):
    file_size = None
    u = urlopen(url)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overriden by a broken download.
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)

def _get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(ENV_TORCH_HOME,
                  os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch')))
    return torch_home

def load_state_dict_from_url(url, model_dir=None, map_location=None, progress=True):
    r"""Loads the Torch serialized object at the given URL.

    If the object is already present in `model_dir`, it's deserialized and
    returned. The filename part of the URL should follow the naming convention
    ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
    digits of the SHA256 hash of the contents of the file. The hash is used to
    ensure unique names and to verify the contents of the file.

    The default value of `model_dir` is ``$TORCH_HOME/checkpoints`` where
    environment variable ``$TORCH_HOME`` defaults to ``$XDG_CACHE_HOME/torch``.
    ``$XDG_CACHE_HOME`` follows the X Design Group specification of the Linux
    filesytem layout, with a default value ``~/.cache`` if not set.

    Args:
        url (string): URL of the object to download
        model_dir (string, optional): directory in which to save the object
        map_location (optional): a function or a dict specifying how to remap storage locations (see torch.load)
        progress (bool, optional): whether or not to display a progress bar to stderr

    Example:
        >>> state_dict = torch.hub.load_state_dict_from_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')

    """
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn('TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    if model_dir is None:
        torch_home = _get_torch_home()
        model_dir = os.path.join(torch_home, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = HASH_REGEX.search(filename).group(1)
        _download_url_to_file(url, cached_file, hash_prefix, progress=progress)
    return torch.load(cached_file, map_location=map_location)

class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        hidden_dim = out_dim

        self.num_layers = num_layers

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)
        )

    def forward(self, x):
        if self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        elif self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        return x

class Model(nn.Module):
    def __init__(self, args, pretrained=False, progress=True):
        super(Model, self).__init__()
        self.args = args
        self.backbone = self.get_backbone(args.dataset, args.network)
        out_dim = self.backbone.fc.weight.shape[1]
        self.backbone.fc = nn.Identity()
        
        if pretrained:
            print('Model: Loading pre-trained model')
            model_dict = self.backbone.state_dict()
            state_dict = load_state_dict_from_url(model_urls[args.network], progress=progress)
            state_dict = {k: v for k, v in state_dict.items() if k not in ['fc.weight', 'fc.bias']}
            model_dict.update(state_dict)
            self.backbone.load_state_dict(model_dict)
        
        if self.args.use_static_topk:
            sim_file = f'data/similarity/{self.args.dataset}_{self.args.static_feat_model}.csv'
            self.static_idx = pd.read_csv(sim_file, index_col=0)

        proj_feat_dim = args.proj_feat_dim
        self.projector = ProjectionMLP(out_dim, proj_feat_dim, args.num_proj_layers)
        self.encoder = nn.Sequential(self.backbone, self.projector)

        self.angular_fc = nn.Linear(proj_feat_dim, args.base_class, bias=False)
        nn.init.xavier_uniform_(self.angular_fc.weight)

    @staticmethod
    def get_backbone(dataset, backbone_name):
        if dataset == 'cifar100' or dataset == 'mini_imagenet':
            from .resnet_CIFAR import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
        elif dataset == 'cub200':
            from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
        else:
            raise NotImplementedError

        backbone_dict = {
            'resnet18': ResNet18(),
            'resnet34': ResNet34(),
            'resnet50': ResNet50(),
            'resnet101': ResNet101(),
            'resnet152': ResNet152(),
        }

        return backbone_dict[backbone_name]

    def forward(self, image):
        return self.backbone.encode(image)

    def get_angular_output(self, image, label):
        pivot = self.angular_fc.weight
        feat = self.encoder(image)
        cos_mat = F.linear(F.normalize(feat, p=2, dim=1), F.normalize(pivot, p=2, dim=1))
        cos_mat = self.add_penalty_margin(cos_mat, label)

        return cos_mat, feat

    def add_penalty_margin(self, cos_mat, label):
        k = self.args.penalty_k
        use_static_topk = self.args.use_static_topk
        use_easy_neg = self.args.use_easy_neg
        use_random_topk = self.args.use_random_topk

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        penalty = self.args.penalty_m
        batch_size = cos_mat.shape[0]

        # mask label indicies
        cos_mat_copy = cos_mat.clone()
        cos_mat_copy[torch.arange(batch_size), label] = float('-inf')

        # find top-k values and indices
        if use_static_topk:
            topk_idxs = self.static_topk(k, label)
        else:
            if use_easy_neg:
                # use top-k w/ negative matrix (find min index)
                neg_cos_mat_copy = (-cos_mat_copy)
                neg_cos_mat_copy[torch.arange(batch_size), label] = float('-inf')   # re-mask label indices
                _, topk_idxs = torch.topk(neg_cos_mat_copy, k, dim=1)
            elif use_random_topk: 
                topk_idxs = self.random_topk(k, label)
            else:   # hard-neg
                _, topk_idxs = torch.topk(cos_mat_copy, k, dim=1)

        # add penalty margin
        batch_idxs = torch.arange(batch_size).view(-1, 1).expand(-1, k)
        penalty_m = torch.zeros_like(cos_mat).to(device)
        penalty_m[batch_idxs, topk_idxs] = penalty

        cos_mat += penalty_m

        return cos_mat

    def static_topk(self, k, label):
        idxs = self.static_idx.iloc[label.cpu().numpy(), 1:k+1].to_numpy()
        idxs = torch.tensor(idxs)
 
        return idxs

    def random_topk(self, k, label):
        batch_size = label.size(0)
        num_classes = self.args.base_class

        # all possible indices (including target)
        all_indices = torch.arange(num_classes).repeat(batch_size, 1)

        # remove target indices
        all_indices[torch.arange(batch_size), label] = num_classes
        valid_indices = all_indices[all_indices != num_classes]

        # reshape valid_indices
        valid_indices = valid_indices.view(batch_size, num_classes-1)

        # shuffle indices (for random) and select k indices
        indices = torch.rand(batch_size, num_classes-1).argsort(dim=1)
        random_indices = torch.gather(valid_indices, 1, indices)[:, :k]

        return random_indices