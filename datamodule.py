import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision import transforms
from PIL import Image
import yaml
import os
from torchvision.transforms import functional
from torchvision.transforms import functional as TF
import functools

from utils import *

class SRDataset(Dataset): 
  #데이터셋의 전처리를 해주는 부분
  def __init__(self, dataset_path, image_size, mode):
        super().__init__()
        self.dataset_path = dataset_path
        self.image_size = image_size
        
        self.image_filenames = os.listdir(self.dataset_path)
        self.image_filenames = [file for file in self.image_filenames if file.endswith('.png')]
        if mode == "train":
            self.transforms = transforms.Compose([
                    transforms.RandomCrop(self.image_size, pad_if_needed=True, padding_mode='reflect'),
                    transforms.RandomApply([
                        functools.partial(TF.rotate, angle=0),
                        functools.partial(TF.rotate, angle=90),
                        functools.partial(TF.rotate, angle=180),
                        functools.partial(TF.rotate, angle=270),
                    ]),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                ])
                
        elif mode == "eval":
            self.transforms = transforms.Compose([
                    transforms.CenterCrop(self.image_size)
                ])

        
  # 총 데이터의 개수를 리턴
  def __len__(self):
        return len(self.image_filenames)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx):
      
        image_path = os.path.join(self.dataset_path, self.image_filenames[idx])
        image = Image.open(image_path).convert('RGB')
        
        image_hr = self.transforms(image)
        image_lr = functional.resize(image_hr, (self.image_size//4, self.image_size//4), interpolation=Image.BICUBIC)
        
    
        return functional.to_tensor(image_lr), functional.to_tensor(image_hr), self.image_filenames[idx]


class SRDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.train_dataset_path = config["dataset"]["train_dataset_path"]
        self.val_dataset_path = config["dataset"]["val_dataset_path"]
        self.test_dataset_path = config["dataset"]["test_dataset_path"]
        self.batch_size = config["dataset"]["batch_size"]
        self.image_size = config["dataset"]["image_size"]
        
        self.num_workers = config['dataset']['num_workers']
        
    def setup(self, stage=None):
        self.train_dataset = SRDataset(self.train_dataset_path, self.image_size, mode = 'train')
        self.val_dataset = SRDataset(self.val_dataset_path, self.image_size, mode = 'eval')    
        self.test_dataset = SRDataset(self.test_dataset_path, self.image_size, mode = 'eval')  
       
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers = self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False)

    def predict_dataloader(self):
        pass
 
if __name__ == "__main__":
    config = yaml.load(open("/media/youngwon/Neo/NeoChoi/Projects/SRGAN/config.yaml", "r"), Loader=yaml.FullLoader)
    sr_datamodule = SRDataModule(config = config)
    sr_datamodule.setup()
    train_dataloader = sr_datamodule.train_dataloader()
    print(next(iter(train_dataloader))[0].shape)
 