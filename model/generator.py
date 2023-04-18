import torch 
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        
        #feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 9, padding = 4, stride = 1),
            nn.PReLU()
        )
        
        #residual blocks
        self.residual_blocks =  nn.ModuleList([ResidualBlock() for _ in range(4)])
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64)
        )
        
        #Upscale block
        
        self.upscale_blocks = nn.Sequential(
            UpscaleBlock(64, upscale_factor=2),
            UpscaleBlock(64, upscale_factor=2)
        )
        
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, padding=4, stride=1)
        
    def forward(self,x):
        x = self.conv1(x)
        residual = x
        
        for residual_block in self.residual_blocks:
            x = residual_block(x)
        
        x = self.conv2(x)
        x += residual
        
        for upscale_block in self.upscale_blocks:
            x = upscale_block(x)
            
        x = self.conv3(x)
          
        return x
    
class UpscaleBlock(nn.Module):
    def __init__(self, in_channels, upscale_factor):
        super(UpscaleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * upscale_factor ** 2, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(upscale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        x = self.prelu(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock,self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64)
        )
        
    def forward(self,x):
        output = self.net(x)
        
        #elementwise sum
        output = output + x
        return output
    
if __name__ == "__main__":
    generator = Generator