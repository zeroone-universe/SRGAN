import torch.nn as nn
import torchvision.models as models

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:35])

    def forward(self, img):
        # feature map 추출
        feature_map = self.feature_extractor(img)
        return feature_map
    
class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.feature_extractor = FeatureExtractor()
        self.mse_loss = nn.MSELoss()

    def forward(self, x_fake, x_real):
        # 생성된 이미지의 feature map 추출
        fake_feature_map = self.feature_extractor(x_fake)

        # 원본 고해상도 이미지의 feature map 추출
        real_feature_map = self.feature_extractor(x_real)

        # content loss 계산
        loss_content = self.mse_loss(fake_feature_map, real_feature_map)

        return loss_content