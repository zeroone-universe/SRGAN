import torch
import torch.nn as nn
import torchvision.models as models

#from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution/blob/8a2ecba423760bfed791a92748bdc58de7fed918/models.py#L291

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        vgg19 = models.vgg19(pretrained=True)

        loss_network = nn.Sequential(*list(vgg19.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
            
        self.loss_network = loss_network

    
    def forward(self, input):
        """
        Forward propagation
        :param input: high-resolution or super-resolution images, a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        :return: the specified VGG19 feature map, a tensor of size (N, feature_map_channels, feature_map_w, feature_map_h)
        """
        output = self.loss_network(input)  # (N, feature_map_channels, feature_map_w, feature_map_h)

        return output
    
class ContentLoss(nn.Module):
    def __init__(self, config):
        super(ContentLoss, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.mse_loss = nn.MSELoss()

    def forward(self, x_fake, x_real):
        # 생성된 이미지의 feature map 추출
        fake_feature_map = self.feature_extractor(x_fake)
        # 원본 고해상도 이미지의 feature map 추출
        real_feature_map = self.feature_extractor(x_real)
        # content loss 계산
        loss_content = self.mse_loss(fake_feature_map, real_feature_map.detach())

        return loss_content
    
if __name__ == "__main__":
    fe = FeatureExtractor(5,4)
    inp = torch.rand((16,3,96,96))
    print(inp.shape)
    output = fe.forward(inp)
    print(output.shape)