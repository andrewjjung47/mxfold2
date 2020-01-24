import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import BPseqDataset
from .fold.nussinov import NussinovFold
from .fold.rnafold import RNAFold
from .fold.zuker import ZukerFold
from .fold.mix import MixedFold


class StructuredLoss(nn.Module):
    def __init__(self, model, loss_pos_paired=0, loss_neg_paired=0, loss_pos_unpaired=0, loss_neg_unpaired=0, 
                l1_weight=0., l2_weight=0., verbose=False):
        super(StructuredLoss, self).__init__()
        self.model = model
        self.loss_pos_paired = loss_pos_paired
        self.loss_neg_paired = loss_neg_paired
        self.loss_pos_unpaired = loss_pos_unpaired
        self.loss_neg_unpaired = loss_neg_unpaired
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.verbose = verbose


    def forward(self, seq, structure, pairs, fname=None):
        pred, pred_s, _, param = self.model(seq, return_param=True, reference=structure,
                                loss_pos_paired=self.loss_pos_paired, loss_neg_paired=self.loss_neg_paired, 
                                loss_pos_unpaired=self.loss_pos_unpaired, loss_neg_unpaired=self.loss_neg_unpaired)
        ref, ref_s, _ = self.model(seq, param=param, constraint=structure, max_internal_length=None)
        loss = pred - ref
        if self.verbose:
            print("Loss = {} = ({} - {})".format(loss.item(), pred.item(), ref.item()))
            print(seq)
            print(pred_s)
            print(ref_s)
        if loss.item()> 1e10 or torch.isnan(loss):
            print()
            print(fname)
            print(loss.item(), pred.item(), ref.item())
            print(seq)
            print(structure)

        if self.l1_weight > 0.0:
            for p in self.model.parameters():
                loss += self.l1_weight * torch.sum(torch.abs(p))

        # if self.l2_weight > 0.0:
        #     l2_reg = 0.0
        #     for p in self.model.parameters():
        #         l2_reg += torch.sum((self.l2_weight * p) ** 2)
        #     loss += torch.sqrt(l2_reg)

        return loss

class PiecewiseLoss(nn.Module):
    def __init__(self, model, l1_weight=0., l2_weight=0., 
                weak_label_weight=1., label_smoothing=0.1, gamma=5., verbose=False):
        super(PiecewiseLoss, self).__init__()
        self.model = model
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.weak_label_weight = weak_label_weight
        self.label_smoothing = label_smoothing
        self.gamma = gamma
        self.verbose = verbose
        self.loss_fn = nn.BCELoss(reduction='sum')


    def forward(self, seq, structure, pairs, fname=None): # BCELoss with 'sum' reduction
        pred_sc, pred_s, pred_bp, param = self.model(seq, return_param=True)
        loss = torch.zeros((len(param),), device=param[0]['score_paired'].device)
        for k in range(len(seq)):
            score_paired = param[k]['score_paired'] / (self.model.gamma*2)
            score_unpaired = param[k]['score_unpaired']
            # print(torch.max(score_unpaired[1:]), torch.max(score_paired[1:, 1:]))
            # print(score_unpaired[score_unpaired>0.5].shape)
            # print(score_paired[1:, 1:])
            # print(pred_bp)
            if len(structure[k]) > 0:
                ref_sc, ref_s, ref_bp = self.model([seq[k]], param=[param[k]], constraint=[structure[k]], max_internal_length=None)
                loss[k] += self.loss_known_structure(seq[k], structure[k], score_paired, score_unpaired, pred_bp[k], ref_bp[0])
            else:
                loss[k] += self.loss_unknown_structure(seq[k], pairs[k], score_paired, score_unpaired, pred_bp[k]) * self.weak_label_weight

            if self.l1_weight > 0.0:
                for p in self.model.parameters():
                    loss[k] += self.l1_weight * torch.sum(torch.abs(p))
        
        return loss


    def loss_known_structure(self, seq, structure, score_paired, score_unpaired, pred_bp, ref_bp):
        pred_paired = torch.zeros_like(score_paired, dtype=torch.bool)
        pred_unpaired = torch.zeros_like(score_unpaired, dtype=torch.bool)
        for i, j in enumerate(pred_bp):
            if i < j:
                pred_paired[i, j] = True
            else:
                pred_unpaired[i] = True
        pred_paired = pred_paired[1:, 1:]
        pred_unpaired = pred_unpaired[1:]

        ref_paired = torch.zeros_like(score_paired, dtype=torch.bool)
        ref_unpaired = torch.zeros_like(score_unpaired, dtype=torch.bool)
        for i, j in enumerate(ref_bp):
            if i < j:
                ref_paired[i, j] = True
            else:
                ref_unpaired[i] = True
        ref_paired = ref_paired[1:, 1:]
        ref_unpaired = ref_unpaired[1:]

        score_paired = score_paired[1:, 1:]
        loss_paired = torch.zeros((1,), device=score_paired.device)
        # fp = score_paired[(pred_paired==True) & (ref_paired==False)]
        # if len(fp) > 0:
        #     #p = (1 - self.label_smoothing) * 0 + self.label_smoothing * 0.5
        #     p = self.label_smoothing * 0.5
        #     loss_paired += self.loss_fn(fp, torch.full_like(fp, p))

        fn = score_paired[(pred_paired==False) & (ref_paired==True)]
        if len(fn) > 0:
            #p = (1 - self.label_smoothing) * 1 + self.label_smoothing * 0.5
            p = 1 - self.label_smoothing * 0.5
            loss_paired += self.gamma * self.loss_fn(fn, torch.full_like(fn, p))

        score_unpaired = score_unpaired[1:]
        loss_unpaired = torch.zeros((1,), device=score_unpaired.device)
        # fp = score_unpaired[(pred_unpaired==True) & (ref_unpaired==False)]
        # if len(fp) > 0:
        #     p = self.label_smoothing * 0.5
        #     loss_unpaired += self.loss_fn(fp, torch.full_like(fp, p))

        fn = score_unpaired[(pred_unpaired==False) & (ref_unpaired==True)]
        if len(fn) > 0:
            #p = (1 - self.label_smoothing) * 1 + self.label_smoothing * 0.5
            p = 1 - self.label_smoothing * 0.5
            loss_unpaired += self.loss_fn(fn, torch.full_like(fn, p))

        return (loss_paired[0] + loss_unpaired[0]) / len(seq)
        #return loss_paired[0]


    def loss_unknown_structure(self, seq, pairs, score_paired, score_unpaired, pred_bp):
        pred_unpaired = torch.zeros_like(score_unpaired)
        for i, j in enumerate(pred_bp):
            if j == 0:
                pred_unpaired[i] = True
        pred_unpaired = pred_unpaired[1:]

        pairs = pairs.to(score_paired.device)
        score_unpaired = score_unpaired[1:]
        #print(pred_bp)
        #print(score_unpaired)
        pairs_not_nan = torch.logical_not(torch.isnan(pairs))
        pairs_not_nan = pairs_not_nan[:, 0] * pairs_not_nan[:, 1]
        pairs = pairs[pairs_not_nan, 0] - pairs[pairs_not_nan, 1]
        score_unpaired = score_unpaired[pairs_not_nan]
        pred_unpaired = pred_unpaired[pairs_not_nan]        

        loss_unpaired = torch.zeros((1,), device=score_unpaired.device)
        fp = score_unpaired[(pred_unpaired==True) & (pairs>0)]
        if len(fp) > 0:
            loss_unpaired += torch.sum(fp * pairs[(pred_unpaired==True) & (pairs>0)])
            # print(len(fp), torch.sum(fp * pairs[(pred_unpaired==True) & (pairs>0)]))

        fn = score_unpaired[(pred_unpaired==False) & (pairs<=0)]
        if len(fn) > 0:
            loss_unpaired += torch.sum((1-fn) * -pairs[(pred_unpaired==False) & (pairs<=0)])
            # print(len(fn), torch.sum((1-fn) * -pairs[(pred_unpaired==False) & (pairs<=0)]))

        #print(loss_unpaired[0], len(fp), len(fn))
        return loss_unpaired[0] / len(seq)

class F1Loss(nn.Module):
    def __init__(self, model, l1_weight=0., l2_weight=0., 
                weak_label_weight=1., verbose=False):
        super(F1Loss, self).__init__()
        self.model = model
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.weak_label_weight = weak_label_weight
        self.verbose = verbose


    def forward(self, seq, structure, pairs, fname=None): # BCELoss with 'sum' reduction
        pred_sc, pred_s, pred_bp, param = self.model(seq, return_param=True)
        loss = torch.zeros((len(param),), device=param[0]['score_paired'].device)
        for k in range(len(seq)):
            score_paired = param[k]['score_paired'] / (self.model.gamma*2)
            score_unpaired = param[k]['score_unpaired']
            # print(torch.max(score_unpaired[1:]), torch.max(score_paired[1:, 1:]))
            # print(score_unpaired[score_unpaired>0.5].shape)
            # print(score_paired[1:, 1:])
            # print(pred_bp)
            if len(structure[k]) > 0:
                ref_sc, ref_s, ref_bp = self.model([seq[k]], param=[param[k]], constraint=[structure[k]], max_internal_length=None)
                loss[k] += self.loss_known_structure(seq[k], score_paired, score_unpaired, pred_bp[k], ref_bp[0])
            else:
                loss[k] += self.loss_unknown_structure(seq[k], pairs[k], score_paired, score_unpaired, pred_bp[k]) * self.weak_label_weight

            if self.l1_weight > 0.0:
                for p in self.model.parameters():
                    loss[k] += self.l1_weight * torch.sum(torch.abs(p))
        
        return loss


    def loss_known_structure(self, seq, score_paired, score_unpaired, pred_bp, ref_bp):
        pred_paired = torch.zeros_like(score_paired, dtype=torch.bool)
        for i, j in enumerate(pred_bp):
            if i < j:
                pred_paired[i, j] = True
        pred_paired = pred_paired[1:, 1:]

        ref_paired = torch.zeros_like(score_paired, dtype=torch.bool)
        for i, j in enumerate(ref_bp):
            if i < j:
                ref_paired[i, j] = True
        ref_paired = ref_paired[1:, 1:]

        score_paired = score_paired[1:, 1:]

        tp = torch.sum(score_paired[(pred_paired==True) & (ref_paired==True)])
        fp = torch.sum(score_paired[(pred_paired==True) & (ref_paired==False)])
        fn = torch.sum(1-score_paired[(pred_paired==False) & (ref_paired==True)])
        
        f = 2*tp / (2*tp + fn + fp) #if tp>0 else tp
        #print(f, tp, fp, fn)
        #print((pred_paired==False) & (ref_paired==True))
        return 1-f


    def loss_unknown_structure(self, seq, pairs, score_paired, score_unpaired, pred_bp):
        pred_unpaired = torch.zeros_like(score_unpaired)
        for i, j in enumerate(pred_bp):
            if j == 0:
                pred_unpaired[i] = True
        pred_unpaired = pred_unpaired[1:]

        pairs = pairs.to(score_paired.device)
        score_unpaired = score_unpaired[1:]
        #print(pred_bp)
        #print(score_unpaired)
        pairs_not_nan = torch.logical_not(torch.isnan(pairs))
        pairs_not_nan = pairs_not_nan[:, 0] * pairs_not_nan[:, 1]
        pairs = pairs[pairs_not_nan, 0] - pairs[pairs_not_nan, 1]
        ref_unpaired = torch.sigmoid(-pairs)
        score_unpaired = score_unpaired[pairs_not_nan]
        pred_unpaired = pred_unpaired[pairs_not_nan]        

        tp_ind = (pred_unpaired==True) & (ref_unpaired>=0.5)
        fp_ind = (pred_unpaired==True) & (ref_unpaired<0.5)
        fn_ind = (pred_unpaired==False) & (ref_unpaired>=0.5)
        tp = torch.sum(score_unpaired[tp_ind] * ref_unpaired[tp_ind])
        fp = torch.sum(score_unpaired[fp_ind] * (1-ref_unpaired[fp_ind]))
        fn = torch.sum((1-score_unpaired[fn_ind]) * ref_unpaired[fn_ind])

        f = 2*tp / (2*tp + fn + fp) if tp>0 else tp
        #print(f.item(), tp.item(), fp.item(), fn.item(), torch.sum(score_unpaired>0.5).item() / score_unpaired.shape[0])
        return 1-f


class Train:
    step = 0

    def __init__(self):
        self.train_loader = None
        self.test_loader = None


    def train(self, epoch):
        self.model.train()
        n_dataset = len(self.train_loader.dataset)
        loss_total, num = 0, 0
        running_loss, n_running_loss = 0, 0
        with tqdm(total=n_dataset, disable=self.disable_progress_bar) as pbar:
            for fnames, seqs, structures, pairs in self.train_loader:
                if self.verbose:
                    print()
                    print("Step: {}, {}".format(self.step, fnames))
                    self.step += 1
                n_batch = len(seqs)
                self.optimizer.zero_grad()
                loss = torch.sum(self.loss_fn(seqs, structures, pairs, fname=fnames))
                loss_total += loss.item()
                num += n_batch
                if loss.item() > 0.:
                    loss.backward()
                    if self.verbose:
                        for n, p in self.model.named_parameters():
                            print(n, torch.min(p).item(), torch.max(p).item(), torch.min(p.grad).item(), torch.max(p.grad).item())
                    self.optimizer.step()

                pbar.set_postfix(train_loss='{:.3e}'.format(loss_total / num))
                pbar.update(n_batch)

                running_loss += loss.item()
                n_running_loss += n_batch
                if n_running_loss >= 100 or num >= n_dataset:
                    running_loss /= n_running_loss
                    if self.writer is not None:
                        self.writer.add_scalar("train/loss", running_loss, (epoch-1) * n_dataset + num)
                    running_loss, n_running_loss = 0, 0
        if self.verbose:
            print()
        print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss_total / num))


    def test(self, epoch):
        self.model.eval()
        n_dataset = len(self.test_loader.dataset)
        loss_total, num = 0, 0
        with torch.no_grad(), tqdm(total=n_dataset, disable=self.disable_progress_bar) as pbar:
            for fnames, seqs, structures, pairs in self.test_loader:
                n_batch = len(seqs)
                loss = self.loss_fn(seqs, structures, pairs, fname=fnames)
                loss_total += loss.item()
                num += n_batch
                pbar.set_postfix(test_loss='{:.3e}'.format(loss_total / num))
                pbar.update(n_batch)

        if self.writer is not None:
            self.writer.add_scalar("test/loss", epoch * n_dataset, loss_total / num)
        print('Test Epoch: {}\tLoss: {:.6f}'.format(epoch, loss_total / num))


    def save_checkpoint(self, outdir, epoch):
        filename = os.path.join(outdir, 'epoch-{}'.format(epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filename)


    def resume_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return epoch


    def build_model(self, args):
        if args.model == 'Turner':
            return RNAFold(), {}

        config = {
            'embed_size' : args.embed_size,
            'num_filters': args.num_filters if args.num_filters is not None else (96,),
            'filter_size': args.filter_size if args.filter_size is not None else (5,),
            'pool_size': args.pool_size if args.pool_size is not None else (1,),
            'dilation': args.dilation, 
            'num_lstm_layers': args.num_lstm_layers, 
            'num_lstm_units': args.num_lstm_units,
            'num_hidden_units': args.num_hidden_units if args.num_hidden_units is not None else (32,),
            'num_paired_filters': args.num_paired_filters,
            'paired_filter_size': args.paired_filter_size,
            'dropout_rate': args.dropout_rate,
            'fc_dropout_rate': args.fc_dropout_rate,
            'num_att': args.num_att,
            'pair_join': args.pair_join,
            'no_split_lr': args.no_split_lr,
        }

        if args.model == 'Zuker':
            model = ZukerFold(model_type='M', **config)

        elif args.model == 'ZukerL':
            model = ZukerFold(model_type="L", **config)

        elif args.model == 'ZukerS':
            model = ZukerFold(model_type="S", **config)

        elif args.model == 'Nussinov':
            model = NussinovFold(model_type='N', **config)

        elif args.model == 'NussinovS':
            config.update({ 'gamma': args.gamma, 'sinkhorn': args.sinkhorn,
                            'sinkhorn_tau': args.sinkhorn_tau})
            model = NussinovFold(model_type='S', gumbel_sinkhorn=args.gumbel_sinkhorn, **config)

        elif args.model == 'Mix':
            from . import param_turner2004
            model = MixedFold(init_param=param_turner2004, **config)

        else:
            raise('not implemented')

        return model, config


    def build_optimizer(self, optimizer, model, lr, l2_weight):
        if optimizer == 'Adam':
            return optim.Adam(model.parameters(), lr=lr, amsgrad=False, weight_decay=l2_weight)
        elif optimizer =='AdamW':
            return optim.AdamW(model.parameters(), lr=lr, amsgrad=False, weight_decay=l2_weight)
        elif optimizer == 'RMSprop':
            return optim.RMSprop(model.parameters(), lr=lr, weight_decay=l2_weight)
        elif optimizer == 'SGD':
            return optim.SGD(model.parameters(), nesterov=True, lr=lr, momentum=0.9, weight_decay=l2_weight)
            #return optim.SGD(model.parameters(), lr=lr, weight_decay=l2_weight)
        elif optimizer == 'ASGD':
            return optim.ASGD(model.parameters(), lr=lr, weight_decay=l2_weight)
        else:
            raise('not implemented')


    def build_loss_function(self, loss_func, model, args):
        if loss_func == 'hinge':
            return StructuredLoss(model, verbose=self.verbose,
                            loss_pos_paired=args.loss_pos_paired, loss_neg_paired=args.loss_neg_paired, 
                            loss_pos_unpaired=args.loss_pos_unpaired, loss_neg_unpaired=args.loss_neg_unpaired, 
                            l1_weight=args.l1_weight, l2_weight=args.l2_weight)
        elif loss_func == 'piecewise':
            return PiecewiseLoss(model, verbose=self.verbose, label_smoothing=args.label_smoothing, gamma=args.gamma,
                            l1_weight=args.l1_weight, l2_weight=args.l2_weight, weak_label_weight=args.weak_label_weight)
        elif loss_func == 'f1':
            return F1Loss(model, verbose=self.verbose, 
                            l1_weight=args.l1_weight, l2_weight=args.l2_weight, weak_label_weight=args.weak_label_weight)
        else:
            raise('not implemented')


    def save_config(self, file, config):
        with open(file, 'w') as f:
            for k, v in config.items():
                k = '--' + k.replace('_', '-')
                if type(v) is bool: # pylint: disable=unidiomatic-typecheck
                    if v:
                        f.write('{}\n'.format(k))
                elif isinstance(v, list) or isinstance(v, tuple):
                    for vv in v:
                        f.write('{}\n{}\n'.format(k, vv))
                else:
                    f.write('{}\n{}\n'.format(k, v))


    def run(self, args):
        self.disable_progress_bar = args.disable_progress_bar
        self.verbose = args.verbose
        self.writer = None
        if args.log_dir is not None:
            self.writer = SummaryWriter(log_dir=args.log_dir)

        train_dataset = BPseqDataset(args.input, unpaired='x')
        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        if args.test_input is not None:
            test_dataset = BPseqDataset(args.test_input, unpaired='x')
            self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        if args.seed >= 0:
            torch.manual_seed(args.seed)
            random.seed(args.seed)

        self.model, config = self.build_model(args)
        config.update({ 'model': args.model, 'param': args.param })
        
        if args.init_param is not '':
            p = torch.load(args.init_param)
            if isinstance(p, dict) and 'model_state_dict' in p:
                p = p['model_state_dict']
            self.model.load_state_dict(p)

        if args.gpu >= 0:
            self.model.to(torch.device("cuda", args.gpu))
        self.optimizer = self.build_optimizer(args.optimizer, self.model, args.lr, args.l2_weight)
        self.loss_fn = self.build_loss_function(args.loss_func, self.model, args)

        checkpoint_epoch = 0
        if args.resume is not None:
            checkpoint_epoch = self.resume_checkpoint(args.resume)

        for epoch in range(checkpoint_epoch+1, args.epochs+1):
            self.train(epoch)
            if self.test_loader is not None:
                self.test(epoch)
            if args.log_dir is not None:
                self.save_checkpoint(args.log_dir, epoch)

        if args.param is not None:
            torch.save(self.model.state_dict(), args.param)
        if args.save_config is not None:
            self.save_config(args.save_config, config)

        return self.model


    @classmethod
    def add_args(cls, parser):
        subparser = parser.add_parser('train', help='training')
        # input
        subparser.add_argument('input', type=str,
                            help='Training data of BPSEQ-formatted file')
        subparser.add_argument('--test-input', type=str,
                            help='Test data of BPSEQ-formatted file')
        subparser.add_argument('--gpu', type=int, default=-1, 
                            help='use GPU with the specified ID (default: -1 = CPU)')
        subparser.add_argument('--seed', type=int, default=0, metavar='S',
                            help='random seed (default: 0)')
        subparser.add_argument('--param', type=str, default='param.pth',
                            help='output file name of trained parameters')
        subparser.add_argument('--init-param', type=str, default='',
                            help='the file name of the initial parameters')

        gparser = subparser.add_argument_group("Training environment")
        subparser.add_argument('--epochs', type=int, default=10, metavar='N',
                            help='number of epochs to train (default: 10)')
        subparser.add_argument('--log-dir', type=str, default=None,
                            help='Directory for storing logs')
        subparser.add_argument('--resume', type=str, default=None,
                            help='Checkpoint file for resume')
        subparser.add_argument('--save-config', type=str, default=None,
                            help='save model configurations')
        subparser.add_argument('--disable-progress-bar', action='store_true',
                            help='disable the progress bar in training')
        subparser.add_argument('--verbose', action='store_true',
                            help='enable verbose outputs for debugging')

        gparser = subparser.add_argument_group("Optimizer setting")
        gparser.add_argument('--optimizer', choices=('Adam', 'AdamW', 'RMSprop', 'SGD', 'ASGD'), default='AdamW')
        gparser.add_argument('--l1-weight', type=float, default=0.,
                            help='the weight for L1 regularization (default: 0)')
        gparser.add_argument('--l2-weight', type=float, default=0.,
                            help='the weight for L2 regularization (default: 0)')
        gparser.add_argument('--weak-label-weight', type=float, default=1.,
                            help='the weight for weak label data (default: 1)')
        gparser.add_argument('--lr', type=float, default=0.001,
                            help='the learning rate for optimizer (default: 0.001)')
        gparser.add_argument('--loss-func', choices=('hinge', 'piecewise', 'f1'), default='hinge',
                            help="loss fuction ('hinge', 'piecewise', 'f1') ")
        gparser.add_argument('--label-smoothing', type=float, default=0.0,
                            help='the label smoothing for piecewise loss (default: 0.0)')
        gparser.add_argument('--loss-pos-paired', type=float, default=0.5,
                            help='the penalty for positive base-pairs for loss augmentation (default: 0.5)')
        gparser.add_argument('--loss-neg-paired', type=float, default=0.005,
                            help='the penalty for negative base-pairs for loss augmentation (default: 0.005)')
        gparser.add_argument('--loss-pos-unpaired', type=float, default=0,
                            help='the penalty for positive unpaired bases for loss augmentation (default: 0)')
        gparser.add_argument('--loss-neg-unpaired', type=float, default=0,
                            help='the penalty for negative unpaired bases for loss augmentation (default: 0)')

        gparser = subparser.add_argument_group("Network setting")
        gparser.add_argument('--model', choices=('Turner', 'Zuker', 'ZukerS', 'ZukerL', 'Mix', 'Nussinov', 'NussinovS'), default='Turner', 
                            help="Folding model ('Turner', 'Zuker', 'ZukerS', 'ZukerL', 'Mix', 'Nussinov', 'NussinovS')")
        gparser.add_argument('--embed-size', type=int, default=0,
                        help='the dimention of embedding (default: 0 == onehot)')
        gparser.add_argument('--num-filters', type=int, action='append',
                        help='the number of CNN filters (default: 96)')
        gparser.add_argument('--filter-size', type=int, action='append',
                        help='the length of each filter of CNN (default: 5)')
        gparser.add_argument('--pool-size', type=int, action='append',
                        help='the width of the max-pooling layer of CNN (default: 1)')
        gparser.add_argument('--dilation', type=int, default=0, 
                        help='Use the dilated convolution (default: 0)')
        gparser.add_argument('--num-lstm-layers', type=int, default=0,
                        help='the number of the LSTM hidden layers (default: 0)')
        gparser.add_argument('--num-lstm-units', type=int, default=0,
                        help='the number of the LSTM hidden units (default: 0)')
        gparser.add_argument('--num-paired-filters', type=int, action='append', default=[],
                        help='the number of CNN filters (default: 96)')
        gparser.add_argument('--paired-filter-size', type=int, action='append', default=[],
                        help='the length of each filter of CNN (default: 5)')
        gparser.add_argument('--num-hidden-units', type=int, action='append',
                        help='the number of the hidden units of full connected layers (default: 32)')
        gparser.add_argument('--dropout-rate', type=float, default=0.0,
                        help='dropout rate of the CNN and LSTM units (default: 0.0)')
        gparser.add_argument('--fc-dropout-rate', type=float, default=0.0,
                        help='dropout rate of the hidden units (default: 0.0)')
        gparser.add_argument('--num-att', type=int, default=0,
                        help='the number of the heads of attention (default: 0)')
        gparser.add_argument('--pair-join', choices=('cat', 'add', 'mul', 'bilinear'), default='cat', 
                            help="how pairs of vectors are joined ('cat', 'add', 'mul', 'bilinear') (default: 'cat')")
        gparser.add_argument('--no-split-lr', default=False, action='store_true')
        gparser.add_argument('--gamma', type=float, default=5,
                        help='the weight of basepair scores in NussinovS model (default: 5)')
        gparser.add_argument('--sinkhorn', type=int, default=64,
                        help='the maximum numger of iteration for Shinkforn normalization in NussinovS model (default: 64)')
        gparser.add_argument('--gumbel-sinkhorn', action='store_true',
                        help='perform Gumbel sampling for secondary structures')
        gparser.add_argument('--sinkhorn-tau', type=float, default=1,
                        help='set the temparature of Sinkhorn')

        subparser.set_defaults(func = lambda args: Train().run(args))
