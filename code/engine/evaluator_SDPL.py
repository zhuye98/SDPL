import os
import numpy as np
import matplotlib.pyplot as plt

from models.networks import *
from misc.metric_tool import ConfuseMatrixMeter
from misc.logger_tool import Logger
from utils import de_norm
import utils
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import f1_score, precision_score, recall_score
import pdb
from tqdm import tqdm
from prettytable import PrettyTable



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
        self.running_metric_sympLevel_prog = ConfuseMatrixMeter(n_class=3)
        self.running_metric_prog = ConfuseMatrixMeter(n_class=3)

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

        self.pred_stack = []
        self.target_stack = []
        self.pred_sigmoid_stack = []
        self.target_prog_stack = []
        self.pred_prog_stack = []

    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):
        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)

            self.net_G.load_state_dict(checkpoint['model_G_state_dict'], strict=False)
            self.net_G.to(self.device)

        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)

    def _update_metric_symp_prog(self):
        """
        update metric
        """
        name = self.batch['name']
        target = torch.stack(self.batch['L_prog'], dim=1).to(self.device).detach()

        self.pred_prog_stack.append(torch.stack(self.prog_pred).detach().cpu().numpy())
        self.target_prog_stack.extend(target.cpu().numpy())

        return ''

    def _calculate_auroc_auprc(self, pred_sigmoid_stack, target_stack):
        # Calculate AUPRC for each class
        auprc_per_class = []
        auroc_per_class = []
        
        for idx in range(pred_sigmoid_stack.shape[1]):
            mask = ~torch.isnan(target_stack)[:, idx]
            predictions = pred_sigmoid_stack[:, idx][mask]
            targets = target_stack[:, idx][mask]

            # auroc
            fpr, tpr, _ = roc_curve(targets, predictions)
            auroc = auc(fpr, tpr)
            auroc_per_class.append(round(auroc, 3))
            
            # auprc
            precision, recall, _ = precision_recall_curve(targets, predictions)
            auprc = auc(recall, precision)
            auprc_per_class.append(round(auprc, 3))
        return auroc_per_class, auprc_per_class

    def _calculate_multilabel_confusion_matrix(self, ground_truth, predictions, num_classes):
        confusion_matrix = torch.zeros(num_classes, 2, 2)
        # Accuracy = torch.zeros(num_classes)
        Precision = torch.zeros(num_classes)
        Recall = torch.zeros(num_classes)
        # FPRate = torch.zeros(num_classes)
        F1_score = torch.zeros(num_classes)
        Specificity = torch.zeros(num_classes)
        
        for i in range(num_classes):
            mask = ~torch.isnan(ground_truth)[:, i]
            tp = torch.logical_and(predictions[:, i][mask] == 1, ground_truth[:, i][mask] == 1)
            tn = torch.logical_and(predictions[:, i][mask] == 0, ground_truth[:, i][mask] == 0)
            fp = torch.logical_and(predictions[:, i][mask] == 1, ground_truth[:, i][mask] == 0)
            fn = torch.logical_and(predictions[:, i][mask] == 0, ground_truth[:, i][mask] == 1)
            
            tn = confusion_matrix[i, 0, 0] = torch.sum(tn)
            fp = confusion_matrix[i, 0, 1] = torch.sum(fp)
            fn = confusion_matrix[i, 1, 0] = torch.sum(fn)
            tp = confusion_matrix[i, 1, 1] = torch.sum(tp)

            # Accuracy[i] = (tp+tn)/(tp+tn+fp+fn)
            Precision[i] = tp/(tp+fp+1e-6)
            Recall[i] = tp/(tp+fn+1e-6)
            # FPRate[i] = fp/(tn+fp)
            F1_score[i] = (2*Precision[i]*Recall[i])/(Precision[i]+Recall[i]+1e-6)
            Specificity[i] = tn/(tn+fp+1e-6)
        
        # Accuracy = Accuracy.numpy().round(3)
        Precision = Precision.numpy().round(3)
        Recall = Recall.numpy().round(3)
        # FPRate = FPRate.numpy().round(3)
        F1_score = F1_score.numpy().round(3)
        Specificity = Specificity.numpy().round(3)

        # get the AUPRC
        pred_sigmoid_stack_tensor = torch.tensor(np.array(self.pred_sigmoid_stack))
        auroc_per_class, auprc_per_class = self._calculate_auroc_auprc(pred_sigmoid_stack_tensor, ground_truth)
        return {'F1_score': F1_score, 'Precision': Precision, 'Sensitivity': Recall, 'Specificity': Specificity, "AUPRC": auprc_per_class, "AUROC": auroc_per_class}


    def _collect_running_batch_states_prog(self):
        self._update_metric_symp_prog()


    def _collect_epoch_states_prog(self):

        pred_prog_stack = torch.tensor(np.concatenate(self.pred_prog_stack, axis=1))
        target_stack_tensor = torch.tensor(np.array(self.target_prog_stack))
        all_tar = []
        all_pred = []

        for cls_ in range(target_stack_tensor.shape[-1]):
            mask_C = ~torch.isnan(target_stack_tensor[:, cls_])
            # calculate the metrics in symptom level progression
            if mask_C.any():
                target_ = target_stack_tensor[mask_C][:, cls_].long()
                prog_pred = pred_prog_stack[cls_].detach()
                prog_pred = prog_pred[mask_C]
                pred_ = torch.argmax(prog_pred, dim=1)
                # current_score = self.running_metric_sympLevel_prog.update_cm(pr=pred_.cpu().numpy(), gt=target_.cpu().numpy())
                
                all_tar.append(target_)
                all_pred.append(pred_)  
        # calculate the metrics overall progression classes
        target_ = torch.cat(all_tar, dim=0)
        pred_ = torch.cat(all_pred, dim=0)

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
        self.running_metric_prog.clear()
        self.running_metric_sympLevel_prog.clear()

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        output = self.net_G(img_in1, img_in2)
        # self.symp_pred_A = output['symp_pred_A']
        # self.symp_pred_B = output['symp_pred_B']
        self.prog_pred = output['prog_pred']
    
    def eval_models(self,checkpoint_name='best_ckpt.pt'):

        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.net_G.eval()
        # print(self.net_G)sum(p.numel() 
        num_params = sum(p.numel() for p in self.net_G.parameters() if p.requires_grad)/(1024*1024)
        print(f"Number of parameters: {round(num_params, 3)}")

        total = len(self.dataloader)
        # Iterate over data.
        for self.batch_id, batch in tqdm(enumerate(self.dataloader, 0), total=total):
            with torch.no_grad():
                self._forward_pass(batch)
            self._collect_running_batch_states_prog()
        self._collect_epoch_states_prog()
