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
        
        self.adv_loss = nn.BCELoss()
        
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
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
            nn.Sigmoid()
        )

        
        
    def forward(self,x):
        x = self.conv1(x)
        for disc_block in self.disc_blocks:
            x = disc_block(x)
        
        output = self.classification(x)
        
        return output
    
    def loss_d(self, x_fake, x_real):
        y_fake = self.forward(x_fake.detach())
        y_real = self.forward(x_real)
        d_loss = 1 - y_real.mean() + y_fake.mean()
        return d_loss
        
        
    def loss_g(self, x_fake, x_real):
        y_fake = self.forward(x_fake)
        g_loss = 1 - y_fake.mean()
        return g_loss