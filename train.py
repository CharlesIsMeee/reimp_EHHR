from asyncore import write
from dis import dis
import os
from statistics import mean
from models.nn1 import Generator, Discriminator, init_net
import torch.nn as nn
import torch
from alive_progress import alive_bar
from data import HCPDataset
from torch.utils.data import DataLoader
from models.loss import TVLoss
from einops import rearrange
from tqdm import tqdm
import numpy as np

from tensorboardX import SummaryWriter


class Trainer():
    def __init__(self, args) -> None:

        self.mode = args.mode
        self.running_mode = args.running_mode

        self.args = args
        self.epoch_num = args.epoch_num

        self.disp_batch = args.disp_batch
        self.save_ckpt_freq = args.save_ckpt_freq
        self.model_save_path = args.model_save_path

        self.lr_G = args.lr_G
        self.lr_D = args.lr_D

        self.beta1 = args.beta1

        self.wgt_l1 = args.wgt_l1
        self.wgt_adv = args.wgt_adv
        self.wgt_tv = args.wgt_tv

        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

        self.train_log_dir = args.train_log_dir
        self.val_log_dir = args.val_log_dir
        self.log_port = args.log_port

        # self.gpu_ids = args.gpu_ids

        str_ids = args.gpu_ids.split(',')
        self.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.gpu_ids.append(id)

        if self.gpu_ids and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.gpu_ids[0]}")
            torch.cuda.set_device(self.gpu_ids[0])
        else:
            self.device = torch.device("cpu")

    def save(self, netG, netD, optimG, optimD, epoch):
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)
        
        torch.save({'netG': netG.state_dict(),
                    'netD': netD.state_dict(),
                    'optimG': optimG.state_dict(),
                    'optimD': optimD.state_dict()},
                    f'{self.model_save_path}/model_epoch{epoch}.pth'
                    )

    def load(self, model_save_path, netG, netD=[], optimG=[], optimD=[], epoch=[], mode='train'):
        if not epoch:
            ckpt = os.listdir(model_save_path)
            ckpt.sort()
            epoch = int(ckpt[-1].split('epoch')[1].split('.pth')[0])

        
        ckpt_path = f"{model_save_path}/model_epoch{epoch}.pth"
        print(ckpt_path)
        ckpt_dict = torch.load(ckpt_path)

        if mode == "train":
            netG.load_state_dict(ckpt_dict['netG'])
            netD.load_state_dict(ckpt_dict['netD'])
            optimG.load_state_dict(ckpt_dict['optimG'])
            optimD.load_state_dict(ckpt_dict['optimD'])

            return netG, netD, optimG, optimD, epoch

        elif mode == "test":
            netG.load_state_dict(ckpt_dict['netG'])

            return netG, epoch

    def train(self):

        running_mode = self.running_mode
        epoch_num = self.epoch_num

        disp_batch = self.disp_batch
        save_ckpt_freq = self.save_ckpt_freq

        lr_G = self.lr_G
        lr_D = self.lr_D

        wgt_l1 = self.wgt_l1
        wgt_adv = self.wgt_adv
        wgt_tv =self.wgt_tv

        batch_size = self.batch_size
        num_workers = self.num_workers

        gpu_ids = self.gpu_ids

        device = self.device

        train_log_dir = self.train_log_dir
        val_log_dir = self.val_log_dir
        log_port = self.log_port

        netG = Generator()
        netD = Discriminator(nch_in=64, nch_ker=64)

        init_net(netG, gpu_ids=gpu_ids)
        init_net(netD, gpu_ids=gpu_ids)

        loss_l1 = nn.L1Loss().to(device)
        loss_adv = nn.BCEWithLogitsLoss().to(device)
        loss_tv = TVLoss().to(device)

        paramsG = netG.parameters()
        paramsD = netD.parameters()

        optimG = torch.optim.Adam(paramsG, lr=lr_G, betas=(self.beta1, 0.999))
        optimD = torch.optim.Adam(paramsD, lr=lr_D, betas=(self.beta1, 0.999))

        train_dataset = HCPDataset(self.args, "train")
        train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)

        val_dataset = HCPDataset(self.args, "val")
        val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers)
        
        start_epoch = 0

        write_train = SummaryWriter(log_dir=train_log_dir)
        write_val = SummaryWriter(log_dir=val_log_dir)

        for epoch in range(start_epoch+1, epoch_num):

            # train
            netG.train()
            netD.train()

            loss_G_l1_train = []
            loss_G_adv_train = []
            loss_G_tv_train = []
            loss_D_real_train = []
            loss_D_fake_train = []
            loss_D_train = []
            loss_G_train = []

            # with alive_bar(len(train_dataloader)) as bar:
            for i, data_batch in enumerate(tqdm(train_dataloader)):

                if self.running_mode == 'development' and i > 2:
                    print("train dataloader is fine!")
                    break

                input_data = data_batch[0].to(device)
                label = data_batch[1].to(device)

                # forward netG
                output_g = netG(input_data)

                # backward netD
                fake = torch.cat([input_data, output_g], dim=1)
                real = torch.cat([input_data, label], dim=1)

                set_requires_grad(netD, True)
                optimD.zero_grad()

                pred_real = netD(real)
                pred_fake = netD(fake.detach())

                loss_D_real = loss_adv(pred_real, torch.ones_like(pred_real))
                loss_D_fake = loss_adv(pred_fake, torch.zeros_like(pred_fake))

                loss_D = 0.5 * (loss_D_real + loss_D_fake)

                loss_D.backward()
                optimD.step()

                # backward netG
                fake = torch.cat([input_data, output_g], dim=1)

                set_requires_grad(netD, False)
                optimG.zero_grad()

                pred_fake = netD(fake)

                loss_G_adv = loss_adv(pred_fake, torch.ones_like(pred_fake))
                loss_G_l1 = loss_l1(output_g, label)
                loss_G_tv = loss_tv(output_g)

                loss_G = wgt_l1 * loss_G_l1 + wgt_tv * loss_G_tv + wgt_adv * loss_G_adv

                loss_G.backward()
                optimG.step()

                loss_G_l1_train += [loss_G_l1.item()]
                loss_G_adv_train += [loss_G_adv.item()]
                loss_G_tv_train += [loss_G_tv.item()]
                loss_D_fake_train += [loss_D_fake.item()]
                loss_D_real_train += [loss_D_real.item()]
                loss_D_train += [loss_D.item()]
                loss_G_train += [loss_G.item()]

                    # print(f"TRAIN: EPOCH {epoch}: BATCH {i}: \n GEN: L1 {loss_G_l1} ADV {loss_G_adv} TV {loss_G_tv} DISC: fake {loss_D_fake} real {loss_D_real}")

                    # bar()

            write_train.add_scalar('loss_G_l1', mean(loss_G_l1_train), epoch)
            write_train.add_scalar('loss_G_adv', mean(loss_G_adv_train), epoch)
            write_train.add_scalar('loss_G_tv', mean(loss_G_tv_train), epoch)
            write_train.add_scalar('loss_G', mean(loss_G_train), epoch)
            write_train.add_scalar('loss_D_fake', mean(loss_D_fake_train), epoch)
            write_train.add_scalar('loss_D_real', mean(loss_D_real_train), epoch)
            write_train.add_scalar('loss_D', mean(loss_D_train), epoch)

            print(f"TRAIN: EPOCH {epoch}: \n GEN: L1 {mean(loss_G_l1_train)} ADV {mean(loss_G_adv_train)} TV {mean(loss_G_tv_train)} DISC: fake {mean(loss_D_fake_train)} real {mean(loss_D_real_train)}")

            with torch.no_grad():
                netG.eval()
                netD.eval()

                loss_G_l1_val = []
                loss_G_adv_val = []
                loss_G_tv_val = []
                loss_D_real_val = []
                loss_D_fake_val = []
                loss_G_val = []
                loss_D_val = []

                # with alive_bar(len(val_dataloader)) as bar:
                for i, data_batch in enumerate(tqdm(val_dataloader)):

                    if self.running_mode == 'development' and i > 2:
                        print("val dataloader is fine!")
                        break

                    input_data = data_batch[0].to(device)
                    label = data_batch[1].to(device)

                    # forward netG
                    output_g = netG(input_data)

                    # forward netD
                    fake = torch.cat([input_data, output_g], dim=1)
                    real = torch.cat([input_data, label], dim=1)

                    pred_real = netD(real)
                    pred_fake = netD(fake.detach())

                    loss_D_real = loss_adv(pred_real, torch.ones_like(pred_real))
                    loss_D_fake = loss_adv(pred_fake, torch.zeros_like(pred_fake))

                    loss_D = 0.5 * (loss_D_real + loss_D_fake)

                    loss_G_adv = loss_adv(pred_fake, torch.ones_like(pred_fake))
                    loss_G_l1 = loss_l1(output_g, label)
                    loss_G_tv = loss_tv(output_g)

                    loss_G = wgt_l1 * loss_G_l1 + wgt_tv * loss_G_tv + wgt_adv * loss_G_adv

                    loss_G_l1_val += [loss_G_l1.item()]
                    loss_G_adv_val += [loss_G_adv.item()]
                    loss_G_tv_val += [loss_G_tv.item()]
                    loss_D_real_val += [loss_D_fake.item()]
                    loss_D_fake_val += [loss_D_real.item()]
                    loss_D_val += [loss_D.item()]
                    loss_G_val += [loss_G.item()]

                    if i == disp_batch:
                        write_val.add_images('input images', rearrange(input_data, "b (c t) h w -> (b c) t h w", t=1))
                        write_val.add_images('label images', rearrange(label, "b (c t) h w -> (b c) t h w", t=1))
                        # bar()
            
            write_val.add_scalar('loss_G_l1', mean(loss_G_l1_val), epoch)
            write_val.add_scalar('loss_G_adv', mean(loss_G_adv_val), epoch)
            write_val.add_scalar('loss_G_tv', mean(loss_G_tv_val), epoch)
            write_val.add_scalar('loss_G', mean(loss_G_val), epoch)
            write_val.add_scalar('loss_D_fake', mean(loss_D_fake_val), epoch)
            write_val.add_scalar('loss_D_real', mean(loss_D_real_val), epoch)
            write_val.add_scalar('loss_D', mean(loss_D_val), epoch)

            print(f"VAL: EPOCH {epoch}: \n GEN: L1 {mean(loss_G_l1_val)} ADV {mean(loss_G_adv_val)} TV {mean(loss_G_tv_val)} DISC: fake {mean(loss_D_fake_val)} real {mean(loss_D_real_val)}")

            if (epoch % save_ckpt_freq) == 0 or running_mode == "development":
                self.save(netG, netD, optimG, optimD, epoch)

        write_train.close()
        write_val.close()
    
    def predict(self):

        running_mode = self.running_mode

        wgt_l1 = self.wgt_l1
        wgt_adv = self.wgt_adv
        wgt_tv =self.wgt_tv

        batch_size = self.batch_size

        gpu_ids = self.gpu_ids

        device = self.device

        q_mask_path = self.args.q_mask_path
        q_mask = np.load(q_mask_path)
        predict_save_dir = self.args.predict_save_dir
        label_save_path = "./result/label"

        if not os.path.exists(predict_save_dir):
            os.makedirs(predict_save_dir)


        netG = Generator()
        netD = Discriminator(nch_in=64, nch_ker=64)

        netG, start_epoch = self.load(self.model_save_path, netG, [netD], epoch=365, mode="test")

        dataset = HCPDataset(self.args, mode="test");
        dataloader = DataLoader(dataset, batch_size)

        predict_result = np.empty_like(dataset.data)
        if not os.path.exists(label_save_path):
            h, w, s, d = predict_result.shape
            label_data = np.empty((h, w, s, int(d/2)))

        with torch.no_grad():
            for i, data_batch in enumerate(tqdm(dataloader)):

                if self.running_mode == 'development' and i > 2:
                        print("test dataloader is fine!")
                        break

                input_data = data_batch[0]

                if not os.path.exists(label_save_path):
                    label = data_batch[1]
                    label = rearrange(label, "b c h w -> h w (b c)")
                    label_data[:, :, i, :] = label.numpy()

                output_g = netG(input_data)

                output_g = rearrange(output_g, "b c h w -> h w (b c)")
                input_data = rearrange(input_data, "b c h w -> h w (b c)")

                h, w, d = output_g.shape
                d = d * 2

                cur_slice = np.empty((h, w, d))
                cur_slice[:, :, q_mask] = input_data.numpy()
                cur_slice[:, :, q_mask+1] = output_g.numpy()
                predict_result[:, :, i, :] = cur_slice

        np.save(f"{predict_save_dir}/predict_{365}.npy", predict_result)
        if not os.path.exists(label_save_path):
            np.save(f"{label_save_path}", label_data)
        

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

            
