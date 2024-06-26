import os
import numpy as np
import matplotlib.pyplot as plt

from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
from prettytable import PrettyTable
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import utils
import csv
import pdb


class CDEvaluator():

    def __init__(self, args, dataloader):

        self.dataloader = dataloader

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)

        self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
                                   else "cpu")
        print(self.device)

        # define some other vars to record the training states
        self.running_metric = ConfuseMatrixMeter(n_class=self.n_class)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)


        #  training log
        self.epoch_acc = 0
        self.best_val_acc = 0.0
        self.best_epoch_id = 0

        self.steps_per_epoch = len(dataloader)

        self.G_pred = None
        self.batch = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir
      

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)
    
        self.pred_prog_stack = []
        self.target_prog_stack = []

    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):
        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])
            self.net_G.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)


    def _update_metric(self):
        """
        update metric
        """
        name = self.batch['name']
        target = self.batch['L'].to(self.device)
        self.pred_prog_stack.append(self.G_pred.detach().cpu().numpy())
        self.target_prog_stack.extend(target.cpu().numpy())

        return ''



    def _collect_running_batch_states(self):
        self._update_metric()


    def _collect_epoch_states_prog(self):
        pred_prog_stack = torch.tensor(np.concatenate(self.pred_prog_stack, axis=0))
        target_stack_tensor = torch.tensor(np.array(self.target_prog_stack))
    
        target_ = target_stack_tensor
        pred_ = torch.argmax(pred_prog_stack, dim=1)

        # overall
        class_recall = recall_score(target_.cpu().numpy(), pred_.cpu().numpy(), average=None)
        class_pre = precision_score(target_.cpu().numpy(), pred_.cpu().numpy(), average=None)  
        class_f1 = f1_score(target_.cpu().numpy(), pred_.cpu().numpy(), average=None)

        table_1 = PrettyTable()
        table_1.field_names = ['Metrics\\Classes', 'Worsened', 'Stable', 'Improved']
        table_1.add_row(['Precision', round(class_pre[0], 3), round(class_pre[2], 3), round(class_pre[1], 3)])
        table_1.add_row(['Recall', round(class_recall[0], 3), round(class_recall[2], 3), round(class_recall[1], 3)])
        table_1.add_row(['F1_score', round(class_f1[0], 3), round(class_f1[2], 3), round(class_f1[1], 3)])

        table_2 = PrettyTable()
        table_2.field_names = ['Metrics\\Classes', ' ']
        table_2.add_row(['Weighted_Precision', round(precision_score(target_.cpu().numpy(), pred_.cpu().numpy(), average="weighted"), 3)])
        table_2.add_row(['Weighted_Recall', round(recall_score(target_.cpu().numpy(), pred_.cpu().numpy(), average="weighted"), 3)])
        table_2.add_row(['Weighted_F1_score', round(f1_score(target_.cpu().numpy(), pred_.cpu().numpy(), average="weighted"), 3)])

        print(table_1)
        print(table_2)

        self.target_prog_stack = []
        self.pred_prog_stack = []
    def _clear_cache(self):
        self.running_metric.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        self.G_pred = self.net_G(img_in1, img_in2)

    def eval_models(self,checkpoint_name='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()
        # print(self.net_G)
        num_params = sum(p.numel() for p in self.net_G.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_params}")

        total = len(self.dataloader)
        # Iterate over data.
        for self.batch_id, batch in tqdm(enumerate(self.dataloader, 0), total=total):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states()
        self._collect_epoch_states_prog()
    
