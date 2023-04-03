import torch 
import torch.nn as nn

class DiscBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(DiscBlock, self).__init__()
        self.net =nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self,x):
        x = self.net(x)
        
        return x
        
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.adv_loss = BCEWithLogitsLoss()
        
        #Feature Extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.disc_blocks = nn.ModuleList([
            DiscBlock(64,64,2),
            DiscBlock(64,128,1),
            DiscBlock(128,128,2),
            DiscBlock(128,256,1),
            DiscBlock(256,256,2),
            DiscBlock(256,512,1),
            DiscBlock(512,512,2),
        ])
        
        #Classification
        self.classification = nn.Sequential(
            nn.Linear(512*24*24,1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )


    def forward(self,x):
        x = self.conv1(x)
        for disc_block in self.disc_blocks:
            x = disc_block(x)
        print(x.shape)
        x = x.view(x.shape[0],-1)
        output = self.classification(x)
        
        return output
    
    def loss_d(self, x_fake, x_real):
        x_fake = x_fake.detach()
        y_fake = self.forward(x_fake)
        y_real = self.forward(x_real)
        real_loss = adv_loss(y_real, torch.ones_like(y_real))
        fake_loss = adv_loss(y_fake, torch.zeros_like(y_fake))
        loss_disc = real_loss + fake_loss
        
        return loss_disc
        
        
    def loss_g(self, x_fake, x_real):
        y_fake = self.forward(x_fake)
        loss_gen = adv_loss(y_fake, torch.ones_like(y_fake))
        
        return loss_gen