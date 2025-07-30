from src.configs.base_config import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--train_case_name', help='Power grid case name',
                                 default='pglib_opf_case14_ieee')
        self.parser.add_argument('--display_freq', type=int, default=1,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                                 help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--update_html_freq', type=int, default=1000,
                                 help='frequency of saving training results to html')
        self.parser.add_argument('--print_freq', type=int, default=1000,
                                 help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=62000,
                                 help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=10,
                                 help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', type=int, default=1,
                                 help='continue training: load the latest model')
        self.parser.add_argument('--epoch_start', type=int, default=0,
                                 help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        self.parser.add_argument('--epoch_end', type=int, default=100,
                                 help='the ending epoch count')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='best',
                                 help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate (epochs)')
        self.parser.add_argument('--niter_decay', type=int, default=50,
                                 help='# of iter to linearly decay learning rate to zero (epochs)')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.001,
                                 help='initial learning rate for adamW, for Batch Size = 1, def 0.0002, set to 0.001 for fin with transformers')

        self.parser.add_argument('--no_lsgan', action='store_true',
                                 help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='L1 penalty on fake B')
        self.parser.add_argument('--lambda_M', type=float, default=100.0,
                                 help='L1 penalty on fake B ROI (bounding boxes)')
        self.parser.add_argument('--lambda_MI', type=float, default=0, help='Mutual information loss')

        self.parser.add_argument('--pinn_loss_type', type=int, default=1, help='Which variant of PINN loss to use: 1 boundary loss + powerflow loss, 2 powerflow loss, 3 boundary loss')
        self.parser.add_argument('--use_pinn_loss', type=int, default=0,
                                 help='When to use PINN loss: 0 no PINN, 1 PINN in evaluation, 2 PINN in training, 3 PINN in training and evaluation')

        self.parser.add_argument('--lr_policy', type=str, default='plateau',
                                 help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_iters', type=int, default=50,
                                 help='multiply by a gamma every lr_decay_iters iterations')
        self.parser.add_argument('--lr_plateau_patience', type=int, default=5,
                                 help='Number of epochs with no improvement after which learning rate will be reduced (when LR policy=plateau)')
        self.parser.add_argument('--lr_step_size', type=int, default=5,
                                 help='Number of epochs for each step of lr decay (when LR policy=step)')
        self.parser.add_argument('--min_lr_plateau', type=float, default=0.00001,
                                 help='Factor by which the learning rate will be reduced. new_lr = lr * factor')
        self.parser.add_argument('--lr_factor', type=float, default=0.5,
                                 help='Factor by which the learning rate will be reduced. new_lr = lr * factor')


        self.parser.add_argument('--weight_decay_G', type=float, default=0.0, help='weight decay appled to Genetator')
        self.parser.add_argument('--weight_decay_D', type=float, default=0.0,
                                 help='weight decay appled to Discriminator')

        self.parser.add_argument('--train_batch_size', type=int, default=256, help='input batch size')

        self.is_train = True


