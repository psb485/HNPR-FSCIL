import os
import time
import utils
import wandb
import torch
import warnings
import numpy as np
import transformers
import torch.cuda.amp as amp
import dataloader.data_utils as data_utils
warnings.filterwarnings('ignore')

from torch.optim import SGD
from criterion import LossFactory
from argparse import ArgumentParser
from validation import NCMValidation
from model.model_factory import Model

def get_args():
    parser = ArgumentParser('Training arguments')

    # model
    parser.add_argument('--network', type=str, default='resnet18', help='Backbone network for model')
    parser.add_argument('--proj_feat_dim', type=int, default=2048, help='Feature dimension of projection layer')
    parser.add_argument('--num_proj_layers', type=int, default=2, help='Number of projection layers')
 
    # dataset
    parser.add_argument('--dataset', type=str, required=True, choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('--data_root', type=str, default='./data', help='Root directory of datasets')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--min_scale', type=float, default=None, help='Minimum scale factor of RandomResizedCrop')

    # train
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--loss_type', type=str, default='cosface', choices=['cosface', 'cross_entropy'])
    parser.add_argument('--loss_s', type=float, default=30.0, help='Scale factor of angular margin loss')
    parser.add_argument('--loss_m', type=float, default=0.4, help='Margin factor of angular margin loss')
    parser.add_argument('--penalty_k', type=int, default=2, help='Top-k classes to impose penalty margin')
    parser.add_argument('--penalty_m', type=float, default=0.05, help='Penalty margin for similar classes')
    parser.add_argument('--lr', type=float, default=1e-02, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    parser.add_argument('--w_decay', type=float, default=5e-04, help='Weight decay')

    # top-k options
    parser.add_argument('--use_static_topk', action='store_true', help='Use static top-k')
    parser.add_argument('--static_feat_model', type=str, choices=['IN1K', 'CLIP'])
    parser.add_argument('--use_easy_neg', action='store_true', help='Use easy-negative pairs (for analysis)')
    parser.add_argument('--use_random_topk', action='store_true', help='Use random pairs (for analysis)')
    
    # test
    parser.add_argument('--test_pretrained', action='store_true', default=False, help='Evaluate pre-trained model')
    parser.add_argument('--test_ckpt_file', type=str, default=None, help='Pre-trained checkpoint file for evaluation')

    # log
    parser.add_argument('--log_dir', type=str, default='./log', help='Path to experiment directory')

    # wandb
    parser.add_argument('--wandb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--run_name', type=str, default=None, help='Name of run for wandb')
    
    # misc
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    # ------ Preliminaries ------ #
    exp_dir = os.path.join(args.log_dir, args.dataset, args.run_name)
    os.makedirs(exp_dir, exist_ok=True)
    utils.set_seed(args.seed, msg=True)                 # set seed

    args = data_utils.set_up_dataset_args(args)         # set dataset args
    if args.wandb:                                      # init wandb
        wandb.init(project='FSCIL', name=f'{args.run_name}', config=args)

    # ------ Model, Optimizer, Loss ------ #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrained = True if args.dataset == 'cub200' else False

    model = Model(args, pretrained).to(device)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.w_decay)
    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer = optimizer, 
        num_warmup_steps = int(0.03 * args.n_epochs),
        num_training_steps = args.n_epochs
    )
    criterion = LossFactory(loss_type=args.loss_type, s=args.loss_s, m=args.loss_m).to(device)

    # ------ Train, Test ------ #
    utils.set_seed(args.seed, msg=False)                # set seed again (reproducibility issue)
    session_acc = [0.0] * args.sessions
    val_ncm = NCMValidation(model)
    for session in range(args.sessions):
        train_loader, val_loader = data_utils.get_dataloader(args, session)
        if session == 0:
            num_train_cls = args.base_class
            if args.test_pretrained:
                model, _ = utils.load_ckpt(model, optimizer, args.test_ckpt_file)
                val_ncm_acc = val_ncm.eval(train_loader, val_loader, num_train_cls, base_sess=True)
            else:
                best_acc = 0.0
                print('\n' + '='*31 + ' TRAIN ' + '='*31)
                for epoch in range(args.n_epochs):
                    tic = time.time()
                    train_loss = train(train_loader, model, criterion, optimizer)

                    # validation
                    val_ncm_acc = val_ncm.eval(train_loader, val_loader, num_train_cls, base_sess=True)
                    toc = time.time()
                    print(f'[Epoch {epoch}]'.ljust(13) + f'val acc: {val_ncm_acc:.2f} | ' + \
                        f'train loss: {train_loss:.4f} | '.ljust(20) + \
                        f'runtime: {toc-tic:.2f}s')

                    # log via wandb
                    if args.wandb:
                        wandb.log({
                            'train_loss': train_loss,
                            'val_acc': val_ncm_acc,
                            'lr': optimizer.param_groups[0]['lr'],
                            'epoch_runtime': toc - tic,
                        }, step=epoch)

                    # save best model
                    best_ckpt_file = os.path.join(exp_dir, 'best.pth')
                    if val_ncm_acc > best_acc:
                        best_acc = val_ncm_acc
                        utils.save_ckpt(epoch, args.network, model, optimizer, val_ncm_acc, best_ckpt_file)
                    
                    scheduler.step()
                
                # save model of last epoch
                last_ckpt_file = os.path.join(exp_dir, 'last.pth')
                utils.save_ckpt(epoch, args.network, model, optimizer, val_ncm_acc, last_ckpt_file)
        else:
            if args.test_pretrained:
                num_train_cls = args.way
                val_ncm_acc = val_ncm.eval(train_loader, val_loader, num_train_cls, base_sess=False)
        
        # save each session's acc     
        if args.test_pretrained:   
            session_acc[session] = val_ncm_acc
            if session == 0:
                print('\n' + '='*27 + ' TEST ' + '='*28)
            print(f'[Session {session}]'.ljust(13) + f'val acc: {val_ncm_acc:.2f} | ' + \
                f'avg acc: {np.average(session_acc[:session+1]):.2f} | '.ljust(17) + \
                f'forget: {session_acc[0]-val_ncm_acc:.2f}')
    
    # ------ Log Results ------ #
    if args.test_pretrained:
        avg_acc = np.average(session_acc)
        pd = session_acc[0] - session_acc[-1]

        global_log_dir = os.path.join(args.log_dir, args.dataset)
        
        result_file = os.path.join(exp_dir, 'result.txt')
        global_log_file = os.path.join(global_log_dir, 'exp_results.txt')

        # local result logging
        with open(result_file, 'a') as f:
            # header
            for i in range(len(session_acc)):
                f.write(f'T{i}'.ljust(7) + '\t')
            f.write('AA'.ljust(7) + '\t' + 'PD'.ljust(7) + '\n')

            # accs
            for acc in session_acc:
                f.write(f'{acc:.2f}'.ljust(7) + '\t')
            f.write(f'{avg_acc:.2f}'.ljust(7) + '\t' + f'{pd:.2f}'.ljust(7) + '\n')
        
        # global result logging
        with open(global_log_file, 'a') as f:
            for acc in session_acc:
                f.write(f'{acc:.2f}'.ljust(7) + '\t')
            f.write(f'{avg_acc:.2f}'.ljust(7) + '\t' + f'{pd:.2f}'.ljust(7) + '\t' + f'{args.run_name}'+ '\n')
           
def train(train_loader, model, criterion, optimizer):
    batch_time = utils.AverageMeter('Time', ':6.3f')
    losses = utils.AverageMeter('Loss', ':.4e')
    scaler = amp.GradScaler()

    tic = time.time()
    model.train()
    for images, target in train_loader:
        loss = 0
        target = target.cuda()
        with amp.autocast():
            for i in range(len(images)):
                images[i] = images[i].cuda()
                # forward
                cos_mat, _ = model.get_angular_output(images[i], target)
                loss += criterion(cos_mat, target)

            loss /= len(images)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

        # log each step
        toc = time.time()
        losses.update(loss.item(), images[0].size(0))
        batch_time.update(toc - tic)

    return losses.avg

if __name__ == '__main__':
    main()