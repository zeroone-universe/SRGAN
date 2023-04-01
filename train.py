import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio.transforms as T
import random

import torchaudio as ta

import os

from model.discriminator import Discriminator
from model.generator import Generator
from model.contentloss import ContentLoss

class SRGANTrain(pl.LightningModule):
    def __init__(self, config):
        super(SRGANTrain, self).__init__()
        self.automatic_optimization = False
        
        self.lr1 = config['optim']['learning_rate1']
        self.lr2 = config['optim']['learning_rate2']
        self.B1 = config['optim']['B1']
        self.B2 = config['optim']['B2']
        
        self.generator = Generator()
        self.discriminator = Discriminator()
        self.contentloss = ContentLoss()
        
        self.val_step = config['train']['val_step']
        
        self.lr_scheduler = lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda epoch: 0.0001 if self.trainer.global_step < 100000 else 0.00001
        )

        
    def forward(self,x):
        

        return output

    def configure_optimizers(self):

        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas = (self.B1, self.B2)
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas = (self.B1, self.B2))

        return optimizer_d, optimizer_g #, [lrscheduler_d, lr_scheduler_g])


    def training_step(self, batch, batch_idx):
        
        optimizer_d, optimizer_g = self.optimizers()
        
        wav_nb, wav_wb, _ = batch
        
        if self.normalize:
            wav_std = wav_nb.std(dim=-1, keepdim=True) + 1e-3
            wav_nb = wav_nb/wav_std        
        else:
            wav_std = 1
            
        wav_bwe = self.forward(wav_nb)
        wav_bwe = wav_bwe * wav_std

        #optimize discriminator
        
        loss_d =self.discriminator.loss_D(wav_bwe, wav_wb)
        
        optimizer_d.zero_grad()
        self.manual_backward(loss_d)
        optimizer_d.step()

        
        #optimize generator

        loss_g = self.discriminator.loss_G(wav_bwe, wav_wb)
        
        optimizer_g.zero_grad()
        self.manual_backward(loss_g)
        optimizer_g.step()
        
        
        self.log("train_loss_d", loss_d, prog_bar = True, batch_size = self.config['dataset']['batch_size'])
        self.log("train_loss_g", loss_g, prog_bar = True, batch_size = self.config['dataset']['batch_size'])

            

    def validation_step(self, batch, batch_idx):
        
        wav_nb, wav_wb, filename = batch

        if self.normalize:
            wav_std = wav_nb.std(dim=-1, keepdim=True) + 1e-3
            wav_nb = wav_nb/wav_std        
        else:
            wav_std = 1
    
        wav_bwe = self.forward(wav_nb)
        
        wav_bwe = wav_bwe * wav_std
        
        loss_d = self.discriminator.loss_D(wav_bwe, wav_wb)
        loss_g = self.discriminator.loss_G(wav_bwe, wav_wb)
       
        if self.current_epoch%self.save_epoch == self.save_epoch-1:
            wav_bwe_cpu = wav_bwe.squeeze(0).cpu()
            val_dir_path = f"{self.output_dir_path}/epoch_{self.current_epoch}"
            check_dir_exist(val_dir_path)
            ta.save(os.path.join(val_dir_path, f"{filename[0]}.wav"), wav_bwe_cpu, 16000)
        else:
            wav_bwe_cpu = wav_bwe.squeeze(0).cpu()
            val_dir_path = f"{self.output_dir_path}/epoch_current"
            check_dir_exist(val_dir_path)
            ta.save(os.path.join(val_dir_path, f"{filename[0]}.wav"), wav_bwe_cpu, 16000)

        wav_wb = wav_wb.squeeze().cpu().numpy()
        wav_bwe = wav_bwe.squeeze().cpu().numpy()

        val_pesq_wb = pesq(fs = 16000, ref = wav_wb, deg = wav_bwe, mode = "wb")
        val_pesq_nb = pesq(fs = 16000, ref = wav_wb, deg = wav_bwe, mode = "nb")
        
        self.log_dict({"val_loss/val_loss_d": loss_d, "val_loss/val_loss_g": loss_g}, batch_size = self.config['dataset']['batch_size'])
        self.log('val_pesq_wb', val_pesq_wb)
        self.log('val_pesq_nb', val_pesq_nb)




    def test_step(self, batch, batch_idx):
        
        wav_nb, wav_wb, filename = batch

        if self.normalize:
            wav_std = wav_nb.std(dim=-1, keepdim=True) + 1e-3
            wav_nb = wav_nb/wav_std        
        else:
            wav_std = 1

        wav_bwe = self.forward(wav_nb)

        wav_bwe = wav_bwe * wav_std
        
        wav_bwe_cpu = wav_bwe.squeeze(0).cpu()
        test_dir_path = f"{self.output_dir_path}/epoch_test"
        check_dir_exist(test_dir_path)
        ta.save(os.path.join(test_dir_path, f"{filename[0]}.wav"), wav_bwe_cpu, 16000)

        wav_wb = wav_wb.squeeze().cpu().numpy()
        wav_bwe = wav_bwe.squeeze().cpu().numpy()

        test_pesq_wb = pesq(fs = 16000, ref = wav_wb, deg = wav_bwe, mode = "wb")
        test_pesq_nb = pesq(fs = 8000, ref = wav_wb, deg = wav_bwe, mode = "nb")

        self.log('test_pesq_wb', test_pesq_wb)
        self.log('test_pesq_nb', test_pesq_nb)

    


    def predict_step(self, batch, batch_idx):
        pass