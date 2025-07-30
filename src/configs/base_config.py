import argparse
import os
import numpy as np
import random
import torch
from multiprocessing import Manager

from src.utils.logging import init_comet

def set_seed(seed=42, deterministic=False):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.cuda.manual_seed_all(seed)


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False
        m = Manager()
        self.cache_t = m.dict()
        self.cache_v = m.dict()

    def initialize(self):
        ###
        #   Experiment Setup
        ###
        self.parser.add_argument('--name', type=str, default='experiment_name',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./outputs/model', help='models are saved here')
        self.parser.add_argument('--model_ckp', type=str, default='output/models/gcn_14_0.pt',
                                 help='path to load existing model (fine-tuning needs --continue_train)')

        ###
        #   Data pool and batch setup
        ###
        self.parser.add_argument('--dataroot', help='path to the dataset',
                                 default='./data')
        self.parser.add_argument('--dataset', help='name of the dataset (OPFDataset, PPMI)',
                                 default='OPFDataset')
        self.parser.add_argument('--eval_splits', type=str, help='split protocols', nargs='+',
                                 choices=['test', 'generalization'], default=['test'])

        self.parser.add_argument('--case_name', help='Power grid case name',
                                 default='pglib_opf_case14_ieee')
        self.parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        self.parser.add_argument('--max_batch_size', type=int, default=100, help='max batch size')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')

        self.parser.add_argument('--train_limit', type=float, default=1, help='train dataset limit (depending of each dataset)')
        self.parser.add_argument('--test_limit', type=float, default=1, help='test dataset limit (depending of each dataset)')

        ###
        #   Model setup
        ###
        self.parser.add_argument('--layers', type=str, default='GraphConv:64:64', help='Layer architectures')

        ###
        #   GPU, cache
        ##
        self.parser.add_argument('--gpu_ids', type=str, default='0',
                                 help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU. !support only single GPU!')
        self.parser.add_argument('--n_threads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--no_cache', action='store_true', default=False, help='Do not use data cache.')

        ###
        #   Experiment tracker
        ##
        self.parser.add_argument('--experiment_tracker', type=str, default='none',
                                 help='which experiment tracker to use (comet, WandB)')
        self.parser.add_argument('--wandb_user', type=str, default='',
                                 help='organization account for WandB')

        self.parser.add_argument('--experiment_project', type=str, default='PINN-GNN-OPF', help='project name for Experiment tracker')

        self.parser.add_argument('--seed', type=int, default=42, help='fixed seed')
        self.parser.add_argument('--debug', action='store_true', default=False, help='debug flag')

    def parse(self):

        self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.is_train = self.is_train  # train or test
        set_seed(self.opt.seed)

        # gradient accumulation
        self.opt.grad_acc_iterations = max(1, self.opt.batch_size // self.opt.max_batch_size)
        self.opt.batch_size = min(self.opt.batch_size, self.opt.max_batch_size)

        str_ids = self.opt.gpu_ids.split(',')[:torch.cuda.device_count()]

        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        if len(self.opt.gpu_ids)>0:
            self.opt.device = torch.device('cuda:{}'.format(self.opt.gpu_ids[0]))
        else:
            self.opt.device = torch.device('cpu')

        self.opt.cache_t = self.cache_t
        self.opt.cache_v = self.cache_v

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        if self.opt.experiment_tracker == 'comet':
            experiment = init_comet(args, args.get('experiment_project') )
        else:
            experiment = BaseExperimentTracker()

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')
        
        return args, experiment


class BaseExperimentTracker(object):
    def __init__(self):
        pass

    def log_metrics(self, metrics, step=None, epoch=None):
        pass

    def log_metric(self,metric_name, value, step=None, epoch=None):
        pass

    def log_parameters(self, params):
        pass

    def log_parameter(self, param_name,value):
        pass

    def log_model(self, model, name, step=None, epoch=None):
        pass
