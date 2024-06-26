from argparse import ArgumentParser
import torch
from engine.trainer import *
import wandb

print(torch.cuda.is_available())
#torch.cuda.memory_summary(device=None, abbreviated=False)

"""
For training the networks
"""


def train(args):
    dataloaders = utils.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()


def test(args):
    from engine.evaluator import CDEvaluator
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test')
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models()


if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='CheXRelFormer', type=str)
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)
   

    # data
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='CXRData', type=str)

    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--shuffle_AB', default=False, type=str)

    # model
    parser.add_argument('--n_class', default=3, type=int)
    parser.add_argument('--n_symp_class', default=8, type=int)
    parser.add_argument('--embed_dim', default=64, type=int)
    parser.add_argument('--pretrain', default=None, type=str)
    parser.add_argument('--multi_scale_train', default=False, type=str)
    parser.add_argument('--multi_scale_infer', default=False, type=str)
    parser.add_argument('--multi_pred_weights', nargs = '+', type = float, default = [0.5, 0.5, 0.5, 0.8, 1.0])

    parser.add_argument('--net_G', default='base_transformer_pos_s4_dd8', type=str,
                        help='base_resnet18 | base_transformer_pos_s4 | '
                             'base_transformer_pos_s4_dd8 | '
                             'base_transformer_pos_s4_dd8_dedim8|CheXRelFormer')
    parser.add_argument('--loss', default='ce', type=str)

    # optimizer
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--max_epochs', default=100, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str,
                        help='linear | step')
    parser.add_argument('--lr_decay_iters', default=10, type=int)

    args = parser.parse_args()
    utils.get_device(args)
    # print(args.gpu_ids)
    
    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
 
    
    train(args)
    test(args)

