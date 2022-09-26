from train import Trainer
import argparse

parser = argparse.ArgumentParser(description="enhanced_gan")

parser.add_argument('--gpu_ids', default='0', dest='gpu_ids')

parser.add_argument('--mode', default='train', choices=['train', 'test'], dest='mode')
parser.add_argument('--running_mode', default='production', choices=['production', 'development'], dest='running_mode')

parser.add_argument('--disp_batch', default=1, type=int, dest='disp_batch')
parser.add_argument('--save_ckpt_freq', default=20, type=int, dest='save_ckpt_freq')
parser.add_argument('--epoch_num', default=500, type=int, dest='epoch_num')
parser.add_argument('--num_workers', default=4, type=int, dest='num_workers')


parser.add_argument('--data_dir', default='', dest='data_dir')
parser.add_argument('--b', type=int, default=1000, dest='b')
parser.add_argument('--q_mask_path', default='', dest='q_mask_path')
parser.add_argument('--label_select_index_path', default='', dest='label_select_index_path')
parser.add_argument('--batch_size', default=1, type=int, dest='batch_size')


parser.add_argument('--model_save_path', default='./result/model/', dest='model_save_path')
parser.add_argument('--predict_save_dir', default='./result/predict/', dest='predict_save_dir')
parser.add_argument('--train_log_dir', default='./result/log/train/', dest='train_log_dir')
parser.add_argument('--val_log_dir', default='./result/log/val/', dest='val_log_dir')
parser.add_argument('--log_port', default=28097)

parser.add_argument("--start_epoch", default=365)


parser.add_argument('--lr_G', type=float, default=2e-4, dest="lr_G")
parser.add_argument('--lr_D', type=float, default=2e-4, dest="lr_D")
parser.add_argument('--beta1', default=0.5, dest='beta1')

parser.add_argument('--wgt_l1', type=float, default=1e0, dest="wgt_l1")
parser.add_argument('--wgt_adv', type=float, default=1e0, dest="wgt_adv")
parser.add_argument('--wgt_tv', type=float, default=1e-4, dest="wgt_tv")

parser.add_argument('--ch_in_D', type=int, default=64, dest="ch_in_D")
parser.add_argument('--ch_inter_D', type=int, default=64, dest="ch_inter_D")

args = parser.parse_args()


def main():
    # args = parser.parse_args()

    trainer = Trainer(args)

    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'test':
        trainer.predict()
        # pass


if __name__ == '__main__':
    main()
