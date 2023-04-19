import pytorch_lightning as pl
import torch

from torchvision import transforms


from model.discriminator import Discriminator
from model.generator import Generator
from model.contentloss import ContentLoss
from model.tvloss import TVLoss

from math import sqrt, ceil
from torchvision.utils import make_grid

from utils import *


class SRGANTrain(pl.LightningModule): 
    def __init__(self, config):
        super(SRGANTrain, self).__init__()
        self.automatic_optimization = False
        
        self.lr = config['optim']['learning_rate']
        self.B1 = config['optim']['B1']
        self.B2 = config['optim']['B2']
        
        self.generator = Generator()
        self.discriminator = Discriminator()
        
        self.mse_loss = torch.nn.MSELoss()
        self.contentloss = ContentLoss(config)
        self.tv_loss = TVLoss()
        
        self.lambda_image = config['loss']['lambda_image']
        self.lambda_adv = config['loss']['lambda_adv']
        self.lambda_content = config['loss']['lambda_content']
        self.lambda_tv = config['loss']['lambda_tv']
              
        self.output_dir_path = config['train']['output_dir_path']
        
        self.config = config

        self.to_pil = transforms.Compose([transforms.ToPILImage()])
    

    def forward(self,x):
        output = self.generator(x)
        return output

    def configure_optimizers(self):

        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas = (self.B1, self.B2))
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas = (self.B1, self.B2))
        
        # return optimizer_d, optimizer_g

        lr_scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size = 1e+5, gamma= 0.1)
        lr_scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size = 1e+5, gamma= 0.1)

        return ({'optimizer': optimizer_d, 'lr_scheduler': lr_scheduler_d}, {'optimizer': optimizer_g, 'lr_scheduler': lr_scheduler_g})

    def training_step(self, batch, batch_idx):
        
        optimizer_d, optimizer_g= self.optimizers()
        lr_scheduler_d, lr_scheduler_g = self.lr_schedulers()
        
        image_lr, image_hr, filename = batch
        
        image_sr = self.forward(image_lr)
               
        #optimizer G
        #followed loss combination of https://github.com/leftthomas/SRGAN/tree/bed685fd4f48cfcecfc5b3a728aa0d220b1a11c0
        
        self.toggle_optimizer(optimizer_g)
        
        image_loss = self.mse_loss(image_sr * 2 - 1, image_hr * 2 - 1) #range [-1. 1]
        advloss_g = self.discriminator.loss_g(image_sr, image_hr)
        contentloss = self.contentloss(image_sr, image_hr)
        tv_loss = self.tv_loss(image_sr)
        
        
        loss_g = self.lambda_image * image_loss + self.lambda_adv * advloss_g + self.lambda_content * contentloss + self.lambda_tv * tv_loss
        
        optimizer_g.zero_grad()
        self.manual_backward(loss_g)
        optimizer_g.step()
        
        self.untoggle_optimizer(optimizer_g)
        
        #optimizer D
        
        self.toggle_optimizer(optimizer_d)
        
        advloss_d = self.discriminator.loss_d(image_sr, image_hr)
        loss_d = advloss_d
        
        optimizer_d.zero_grad()
        self.manual_backward(loss_d)
        optimizer_d.step()
        
        self.untoggle_optimizer(optimizer_d)

        
        self.log("train/advloss_d", advloss_d,  prog_bar = True,  batch_size = self.config['dataset']['batch_size'])
        
        self.log("train/image_loss", image_loss, prog_bar = True,  batch_size = self.config['dataset']['batch_size'])
        self.log("train/advloss_g", advloss_g, prog_bar = True,  batch_size = self.config['dataset']['batch_size'])
        self.log("train/contentloss", contentloss,  prog_bar = True,  batch_size = self.config['dataset']['batch_size'])        
        self.log("train/tv_loss", tv_loss,  prog_bar = True, batch_size = self.config['dataset']['batch_size'])
        
        lr_scheduler_d.step()
        lr_scheduler_g.step()
        

    
    
    def validation_step(self, batch, batch_idx):
        
        image_lr, image_hr, filename = batch

        image_sr = self.forward(image_lr)
        
        advloss_d = self.discriminator.loss_d(image_sr, image_hr)
        
        image_loss = self.mse_loss(image_sr, image_hr)
        advloss_g = self.discriminator.loss_g(image_sr, image_hr)
        contentloss = self.contentloss(image_sr, image_hr)
        tv_loss = self.tv_loss(image_sr)        
        
        
        self.log("val/advloss_d", advloss_d, batch_size = self.config['dataset']['batch_size'])
        
        self.log("val/image_loss", image_loss, batch_size = self.config['dataset']['batch_size'])
        self.log("val/advloss_g", advloss_g,batch_size = self.config['dataset']['batch_size'])
        self.log("val/contentloss", contentloss, batch_size = self.config['dataset']['batch_size'])        
        self.log("val/tv_loss", tv_loss,  batch_size = self.config['dataset']['batch_size'])



        nrow = ceil(sqrt(self.config['dataset']['batch_size']))
        self.logger.experiment.add_image(
            tag='train/lr_img',
            img_tensor=make_grid(image_lr, nrow=nrow, padding=0),
            global_step=self.global_step
        )
        self.logger.experiment.add_image(
            tag='train/hr_img',
            img_tensor=make_grid(image_hr, nrow=nrow, padding=0),
            global_step=self.global_step
        )
        self.logger.experiment.add_image(
            tag='train/sr_img',
            img_tensor=make_grid(image_sr, nrow=nrow, padding=0),
            global_step=self.global_step
        )

    def test_step(self, batch, batch_idx):
        image_lr, image_hr, filename = batch

        image_sr = self.forward(image_lr)
        image_sr = image_sr.squeeze(0)
        
        image_sr = self.to_pil(image_sr)
        image_sr.save(f"{self.output_dir_path}/{filename[0]}")
        

    def predict_step(self, batch, batch_idx):
        pass