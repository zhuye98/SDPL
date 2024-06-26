import numpy as np
import matplotlib.pyplot as plt
import os
import utils
from models.networks import *
import torch
import torch.optim as optim
import numpy as np
from misc.metric_tool import ConfuseMatrixMeter, ConfuseMatrixMeter_prog
from misc.logger_tool import Logger, Timer
from utils import de_norm
from tqdm import tqdm
import csv
import wandb
import pdb
from sklearn.metrics import multilabel_confusion_matrix
from prettytable import PrettyTable
from sklearn.metrics import precision_recall_curve, roc_curve, auc

class Trainer():

    def __init__(self, args, dataloaders):
        self.args = args
        self.dataloaders = dataloaders

        self.n_class = args.n_class
        # define G
        self.net_G = define_G(args=args, gpu_ids=args.gpu_ids)

        # self.device = torch.device("cuda:%s" % args.gpu_ids[0] if torch.cuda.is_available() and len(args.gpu_ids)>0
        #                            else "cpu")
        self.device = torch.device("cuda")
        print(self.device)

        # Learning rate and Beta1 for Adam optimizers
        self.lr = args.lr

        # define optimizers
        if args.optimizer == "sgd":
            self.optimizer_G = optim.SGD(self.net_G.parameters(), lr=self.lr,
                                     momentum=0.9,
                                     weight_decay=5e-4)
        elif args.optimizer == "adam":
            self.optimizer_G = optim.Adam(self.net_G.parameters(), lr=self.lr,
                                     weight_decay=0)
        elif args.optimizer == "adamw":
            self.optimizer_G = optim.AdamW(self.net_G.parameters(), lr=self.lr,
                                    betas=(0.9, 0.999), weight_decay=0.01)

        # define lr schedulers
        self.exp_lr_scheduler_G = get_scheduler(self.optimizer_G, args)

        self.running_metric_sympLevel_prog = ConfuseMatrixMeter(n_class=3)
        self.running_metric_prog = ConfuseMatrixMeter_prog(n_class=3)

        # define logger file
        logger_path = os.path.join(args.checkpoint_dir, 'log.txt')
        self.logger = Logger(logger_path)
        self.logger.write_dict_str(args.__dict__)
        # define timer
        self.timer = Timer()
        self.batch_size = args.batch_size

        #  training log
        self.best_val_auroc_symp = 0.0
        self.best_val_auprc_symp = 0.0
        self.best_val_mF1_prog = 0.0
        self.best_epoch_id = 0
        self.epoch_to_start = 0
        self.max_num_epochs = args.max_epochs

        self.global_step = 0
        self.steps_per_epoch = len(dataloaders['train'])
        self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch

        self.G_pred = None
        self.batch = None
        self.G_loss = None
        self.is_training = False
        self.batch_id = 0
        self.epoch_id = 0
        self.checkpoint_dir = args.checkpoint_dir

        # define the loss functions
        self.criterion_symp_cls = torch.nn.BCEWithLogitsLoss().to(self.device)
        # Binary Cross Entropy Loss for mortality prediction
        if args.loss == 'ce':
            self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        else:
            raise NotImplemented(args.loss)

        self.VAL_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'val_acc.txt')):
            self.VAL_ACC = np.load(os.path.join(self.checkpoint_dir, 'val_acc.txt'))
        self.TRAIN_ACC = np.array([], np.float32)
        if os.path.exists(os.path.join(self.checkpoint_dir, 'train_acc.txt')):
            self.TRAIN_ACC = np.load(os.path.join(self.checkpoint_dir, 'train_acc.txt'))

        # check and create model dir
        if os.path.exists(self.checkpoint_dir) is False:
            os.mkdir(self.checkpoint_dir)

        self.pred_stack = []
        self.target_stack = []
        self.pred_sigmoid_stack = [] 
        self.target_prog_stack = []
        self.pred_prog_stack = []


    def _load_checkpoint(self, ckpt_name='last_ckpt.pt'):
        print("\n")
        if os.path.exists(os.path.join(self.checkpoint_dir, ckpt_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, ckpt_name),
                                    map_location=self.device)
            # update net_G states
            self.net_G.load_state_dict(checkpoint['model_G_state_dict'])

            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.exp_lr_scheduler_G.load_state_dict(checkpoint['exp_lr_scheduler_G_state_dict'])

            self.net_G.to(self.device)

            # Set the ramdom state
            np.random.set_state(checkpoint['numpy_random_state'])
            torch.random.set_rng_state(checkpoint['torch_random_state'].to('cpu'))

            # update some other states
            self.epoch_to_start = checkpoint['epoch_id'] + 1
            self.best_val_mF1_prog = checkpoint['best_val_mF1_prog']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.total_steps = (self.max_num_epochs - self.epoch_to_start) * self.steps_per_epoch

            self.logger.write('Epoch_to_start = %d, Historical_best_mF1_prog = %.4f (at epoch %d)\n' %
                  (self.epoch_to_start, self.best_val_mF1_prog, self.best_epoch_id))
            self.logger.write('\n')

        else:
            print('training from scratch...')
        print("\n")

    def _timer_update(self):
        self.global_step = (self.epoch_id-self.epoch_to_start) * self.steps_per_epoch + self.batch_id

        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.batch_size / self.timer.get_stage_elapsed()
        return imps, est

  

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch_id,
            'best_val_auroc_symp': self.best_val_auroc_symp,
            'best_val_auprc_symp': self.best_val_auprc_symp,
            'best_val_mF1_prog': self.best_val_mF1_prog,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.net_G.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'exp_lr_scheduler_G_state_dict': self.exp_lr_scheduler_G.state_dict(),
            'torch_random_state' : torch.random.get_rng_state(),
            'numpy_random_state' : np.random.get_state()
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_lr_schedulers(self):
        self.exp_lr_scheduler_G.step()
    
    def _update_metric_symp(self):
        """
        update metric
        """
        name = self.batch['name']
        target_A = torch.stack(self.batch['L_symp_A'], dim=1).to(self.device).detach()
        target_B = torch.stack(self.batch['L_symp_B'], dim=1).to(self.device).detach()
        target = torch.concat((target_A, target_B), dim=0)

        # scale 3
        pred_A = self.symp_pred_A.detach()
        pred_B = self.symp_pred_B.detach()
        pred_ = torch.concat((pred_A, pred_B), dim=0)

        pred_sigmoid = torch.sigmoid(pred_)
        pred_ = torch.where(pred_sigmoid>0.5, torch.ones_like(pred_sigmoid), torch.zeros_like(pred_sigmoid))

        # collect the prediction and target for one epoch
        self.pred_stack.extend(pred_.cpu().numpy())
        self.target_stack.extend(target.cpu().numpy())
        self.pred_sigmoid_stack.extend(pred_sigmoid.cpu().numpy())
        return ''

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

    def _collect_running_batch_states_symp(self):
        self._update_metric_symp()

        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])

        imps, est = self._timer_update()

        if np.mod(self.batch_id, 50) == 0:
            message = '\nIs_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, loss_1_A: %.5f, loss_1_B: %.5f\n' %\
                        (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m, imps*self.batch_size, est,
                        self.loss_A.item(), self.loss_B.item())
            self.logger.write(message)
            # if self.is_training:
            #     wandb.log({'loss_1': round(self.loss_A.item()+self.loss_B.item(), 3)})

                
    def _collect_running_batch_states_prog(self):
        self._update_metric_symp_prog()
        m = len(self.dataloaders['train'])
        if self.is_training is False:
            m = len(self.dataloaders['val'])

        imps, est = self._timer_update()
        if np.mod(self.batch_id, 50) == 0:
            message = '\nIs_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, loss_2: %.5f\n' %\
                        (self.is_training, self.epoch_id, self.max_num_epochs-1, self.batch_id, m, imps*self.batch_size, est,
                        self.loss_2.item())
            self.logger.write(message)
            # if self.is_training:
            #     wandb.log({'loss_2': round(self.loss_2.item(), 3)})

    def _collect_epoch_states_symp(self):
        pred_stack_tensor = torch.tensor(np.array(self.pred_stack))
        target_stack_tensor = torch.tensor(np.array(self.target_stack))
        metrics = self._calculate_multilabel_confusion_matrix(target_stack_tensor, pred_stack_tensor, num_classes=8)
        
        F1_score = metrics['F1_score']
        Precision = metrics['Precision']
        Sensitivity = metrics['Sensitivity']
        Specificity = metrics['Specificity']
        AUROC = metrics['AUROC']
        AUPRC = metrics['AUPRC']
        
        if self.is_training:
            class_names = ['train_LO', 'train_PE', 'train_At', 'train_Ca', 'train_Ed', 'train_PnTh', 'train_Co', 'train_Pn']
            overall_name = 'train_Overall'
        else:
            class_names = ['val_LO', 'val_PE', 'val_At', 'val_Ca', 'val_Ed', 'val_PnTh', 'val_Co', 'val_Pn']
            overall_name = 'val_Overall'
        # for idx in range(len(class_names)):
        #     wandb.log({'{}_F1-score'.format(class_names[idx]): F1_score[idx]})
        #     wandb.log({'{}_precision'.format(class_names[idx]): Precision[idx]})
        #     wandb.log({'{}_sensitivity'.format(class_names[idx]): Sensitivity[idx]})
        #     wandb.log({'{}_specificity'.format(class_names[idx]): Specificity[idx]})
        #     wandb.log({'{}_AUPRC'.format(class_names[idx]): AUPRC[idx]})
        #     wandb.log({'{}_AUROC'.format(class_names[idx]): AUROC[idx]})

        self.mF1_score = np.mean(F1_score)
        self.mPresicion = np.mean(Precision)
        self.mSensitivity = np.mean(Sensitivity)
        self.mSpecificity = np.mean(Specificity)
        self.mAUROC = np.mean(AUROC)
        self.mAUPRC = np.mean(AUPRC)

        # wandb.log({'{}_symp_F1-score'.format(overall_name): self.mF1_score})
        # wandb.log({'{}_symp_precision'.format(overall_name): self.mPresicion})
        # wandb.log({'{}_symp_sensitivity'.format(overall_name): self.mSensitivity})
        # wandb.log({'{}_symp_specificity'.format(overall_name): self.mSpecificity})
        # wandb.log({'{}_symp_AUPRC'.format(overall_name): self.mAUPRC})
        # wandb.log({'{}_symp_AUROC'.format(overall_name): self.mAUROC})

        self.logger.write('Is_training: %s. Epoch %d / %d\n' %
              (self.is_training, self.epoch_id, self.max_num_epochs-1))

        self.pred_stack = []
        self.target_stack = []
        self.pred_sigmoid_stack = []

    def _collect_epoch_states_prog(self):

        pred_prog_stack = torch.tensor(np.concatenate(self.pred_prog_stack, axis=1))
        target_stack_tensor = torch.tensor(np.array(self.target_prog_stack))
        
        all_tar = []
        all_pred = []
        if self.is_training:
            class_names = ['train_LO', 'train_PE', 'train_At', 'train_Ca', 'train_Ed', 'train_PnTh', 'train_Co', 'train_Pn']
            overall_name = 'train_Overall'
        else:
            class_names = ['val_LO', 'val_PE', 'val_At', 'val_Ca', 'val_Ed', 'val_PnTh', 'val_Co', 'val_Pn']
            overall_name = 'val_Overall'
        for cls_ in range(target_stack_tensor.shape[-1]):
            mask_C = ~torch.isnan(target_stack_tensor[:, cls_])
            # calculate the metrics in symptom level progression
            if mask_C.any():
                target_ = target_stack_tensor[mask_C][:, cls_].long()
                prog_pred = pred_prog_stack[cls_].detach()
                prog_pred = prog_pred[mask_C]
                pred_ = torch.argmax(prog_pred, dim=1)
                current_score = self.running_metric_sympLevel_prog.update_cm(pr=pred_.cpu().numpy(), gt=target_.cpu().numpy())
                # wandb log
                # wandb.log({'{}_prog_accuracy'.format(class_names[cls_]): current_score['mAccuracy']})
                # wandb.log({'{}_prog_F1-score'.format(class_names[cls_]): current_score['mF1_score']})
                # wandb.log({'{}_prog_precision'.format(class_names[cls_]): current_score['mPrecision']})
                # wandb.log({'{}_prog_sensitivity'.format(class_names[cls_]): current_score['mSensitivity']})
                # wandb.log({'{}_prog_specificity'.format(class_names[cls_]): current_score['mSpecificity']})
                all_tar.append(target_)
                all_pred.append(pred_)  
        # calculate the metrics overall progression classes
        target_ = torch.cat(all_tar, dim=0)
        pred_ = torch.cat(all_pred, dim=0)
        current_score_prog = self.running_metric_prog.update_cm(pr=pred_.cpu().numpy(), gt=target_.cpu().numpy())
        # wandb.log({'{}_prog_accuracy'.format(overall_name): current_score_prog['mAccuracy']})
        # wandb.log({'{}_prog_F1-score'.format(overall_name): current_score_prog['mF1_score']})
        # wandb.log({'{}_prog_precision'.format(overall_name): current_score_prog['mPrecision']})
        # wandb.log({'{}_prog_sensitivity'.format(overall_name): current_score_prog['mSensitivity']})
        # wandb.log({'{}_prog_specificity'.format(overall_name): current_score_prog['mSpecificity']})

        self.mF1_score_prog = np.mean(current_score_prog['mF1_score'])
        self.mPresicion = np.mean(current_score_prog['mPrecision'])
        self.mSensitivity = np.mean(current_score_prog['mSensitivity'])
        self.mSpecificity = np.mean(current_score_prog['mSpecificity'])
        self.mAccuracy_prog = np.mean(current_score_prog['mAccuracy'])


        self.logger.write('Is_training: %s. Epoch %d / %d\n' %
              (self.is_training, self.epoch_id, self.max_num_epochs-1))

        self.target_prog_stack = []
        self.pred_prog_stack = []

    def _update_checkpoints(self):
        # save current model

        self.logger.write('Lastest model updated. Epoch_auroc_symp=%.3f, Epoch_auprc_symp=%.3f, Epoch_mF1_prog=%.3f \
                          Historical_best_auroc_symp=%.3f, Historical_best_auprc_symp=%.3f, Historical_best_mF1_prog = %.3f (at epoch %d)\n'
                            % (self.mAUROC, self.mAUPRC, self.mF1_score_prog, self.best_val_auroc_symp, self.best_val_auprc_symp, self.best_val_mF1_prog, self.best_epoch_id))
        self.logger.write('\n')
        
        # update the best symp performance
        if self.mAUPRC > self.best_val_auprc_symp and self.mAUROC > self.best_val_auroc_symp:
            self.best_val_auprc_symp = self.mAUPRC
            self.best_val_auroc_symp = self.mAUROC

        # update the best prog performance
        if self.mF1_score_prog > self.best_val_mF1_prog:
            self.best_val_mF1_prog = self.mF1_score_prog
            self.best_epoch_id = self.epoch_id
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')
            

    def _clear_cache(self):
        self.running_metric_prog.clear()
        self.running_metric_sympLevel_prog.clear()
      

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        output = self.net_G(img_in1, img_in2)
        self.symp_pred_A = output['symp_pred_A']
        self.symp_pred_B = output['symp_pred_B']
        self.prog_pred = output['prog_pred']
                
    def _backward_G_symp(self):
        # ground truth for symptom classification task
        gt_symp_A = torch.stack(self.batch['L_symp_A'], dim=1).to(self.device)
        gt_symp_B = torch.stack(self.batch['L_symp_B'], dim=1).to(self.device)
        
        # Compute the loss
        mask_A = ~torch.isnan(gt_symp_A)
        mask_B = ~torch.isnan(gt_symp_B)
        
        self.symp_pred_A_masked = self.symp_pred_A[mask_A]
        gt_symp_A = gt_symp_A[mask_A]
        
        self.loss_A = self.criterion_symp_cls(self.symp_pred_A_masked, gt_symp_A.to(self.device))

        self.symp_pred_B_masked = self.symp_pred_B[mask_B]
        gt_symp_B = gt_symp_B[mask_B]
        self.loss_B = self.criterion_symp_cls(self.symp_pred_B_masked, gt_symp_B.to(self.device))

        self.loss_1 = self.loss_A + self.loss_B

        #TODO Compute the loss of symptom progression
        gt_prog = torch.stack(self.batch['L_prog'], dim=1).to(self.device)
        self.loss_2 = 0.0
        self.count_symp = 0.0
        self.loss_2_list = []

        for cls_ in range(gt_prog.shape[-1]):
            mask_C = ~torch.isnan(gt_prog[:, cls_])
            if mask_C.any():
                self.count_symp += 1
                gt_prog_ = gt_prog[mask_C][:, cls_]
                self.prog_pred_ = self.prog_pred[cls_][mask_C]
                
                loss = self.criterion(self.prog_pred_, gt_prog_.long())
                self.loss_2_list.append(round(loss.item(), 3))
                self.loss_2 += loss
            else:
                self.loss_2_list.append(None)

        self.loss_2 = self.loss_2/self.count_symp
    
        self.loss_total = self.loss_1 + self.loss_2
        self.loss_total.backward()

    def train_models(self):
            self._load_checkpoint()
            # loop over the dataset multiple times
            for self.epoch_id in range(self.epoch_to_start, self.max_num_epochs):
                ################## train #################
                self._clear_cache()
                self.is_training = True
                self.net_G.train()  # Set model to training mode
                # Iterate over data.
                total = len(self.dataloaders['train'])
                self.logger.write('lr: %0.7f\n \n' % self.optimizer_G.param_groups[0]['lr'])
                for self.batch_id, batch in tqdm(enumerate(self.dataloaders['train'], 0), total=total):
                    self._forward_pass(batch)
                    self.optimizer_G.zero_grad()
                    # update G
                    self._backward_G_symp()
                    self.optimizer_G.step()
                    self._collect_running_batch_states_symp()
                    self._collect_running_batch_states_prog()
                    self._timer_update()

                self._collect_epoch_states_symp()
                self._collect_epoch_states_prog()
                self._update_lr_schedulers()

                ################## Eval ##################
                self.logger.write('Begin evaluation...\n')
                self._clear_cache()
                self.is_training = False
                self.net_G.eval()

                # Iterate over data.
                for self.batch_id, batch in enumerate(self.dataloaders['val'], 0):
                    with torch.no_grad():
                        self._forward_pass(batch)
                    self._collect_running_batch_states_symp()
                    self._collect_running_batch_states_prog()
                self._collect_epoch_states_symp()
                self._collect_epoch_states_prog()


                ########### Update_Checkpoints ###########
                self._update_checkpoints()
                self._clear_cache()