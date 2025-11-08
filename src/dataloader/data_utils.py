import torch
import numpy as np

from .cifar import CIFAR100
from .cub200 import CUB200
from .mini import MiniImageNet

def set_up_dataset_args(args):
    if args.dataset == 'cifar100':
        args.base_class = 60
        args.way = 5
        args.shot = 5
        args.sessions = 9
    elif args.dataset == 'cub200':
        args.base_class = 100
        args.way = 10
        args.shot = 5
        args.sessions = 11
    elif args.dataset == 'mini_imagenet':
        args.base_class = 60
        args.way = 5
        args.shot = 5
        args.sessions = 9
    else:
        raise NotImplementedError

    return args

def get_dataloader(args, session):
    if session == 0:
        trainloader, testloader = get_base_dataloader(args)
    else:
        trainloader, testloader = get_inc_dataloader(args, session)

    return trainloader, testloader

def get_base_dataloader(args):
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = CIFAR100(root=args.data_root, train=True, download=True, index=class_index, base_sess=True, min_scale=args.min_scale)
        testset = CIFAR100(root=args.data_root, train=False, download=False, index=class_index, base_sess=True, min_scale=args.min_scale)
    elif args.dataset == 'cub200':
        trainset = CUB200(root=args.data_root, train=True, index=class_index, base_sess=True, min_scale=args.min_scale)
        testset = CUB200(root=args.data_root, train=False, index=class_index, min_scale=args.min_scale)
    elif args.dataset == 'mini_imagenet':
        trainset = MiniImageNet(root=args.data_root, train=True, index=class_index, base_sess=True, min_scale=args.min_scale)
        testset = MiniImageNet(root=args.data_root, train=False, index=class_index, min_scale=args.min_scale)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainloader, testloader

def get_inc_dataloader(args, session):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = CIFAR100(root=args.data_root, train=True, download=False, index=class_index, base_sess=False, min_scale=args.min_scale)
    elif args.dataset == 'cub200':
        trainset = CUB200(root=args.data_root, train=True, index_path=txt_path, min_scale=args.min_scale)
    elif args.dataset == 'mini_imagenet':
        trainset = MiniImageNet(root=args.data_root, train=True, index_path=txt_path, min_scale=args.min_scale)
    else:
        raise NotImplementedError

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)
    if args.dataset == 'cifar100':
        testset = CIFAR100(root=args.data_root, train=False, download=False, index=class_new, base_sess=False, min_scale=args.min_scale)
    elif args.dataset == 'cub200':
        testset = CUB200(root=args.data_root, train=False, index=class_new, min_scale=args.min_scale)
    elif args.dataset == 'mini_imagenet':
        testset = MiniImageNet(root=args.data_root, train=False, index=class_new, min_scale=args.min_scale)
    else:
        raise NotImplementedError

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainloader, testloader

def get_session_classes(args, session):
    class_list=np.arange(args.base_class + session * args.way)

    return class_list