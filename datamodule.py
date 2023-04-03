import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class SRDataset(Dataset): 
  #데이터셋의 전처리를 해주는 부분
  def __init__(self, dataset_path, image_size)):
        super().__init__()
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.image_filenames = os.listdir(self.dataset_path)
        
  # 총 데이터의 개수를 리턴
  def __len__(self):
        return len(self.image_filenames)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
  def __getitem__(self, idx):
      
        image_path = os.path.join(self.dataset_path, self.image_filenames[idx])
        image = Image.open(image_path).convert('RGB')
        image_lr = image.resize((self.image_size // 4, self.image_size // 4), Image.BICUBIC)
        image_hr = image.resize((self.image_size, self.image_size), Image.BICUBIC)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        image_lr = transform(image_lr)
        image_hr = transform(image_hr)

        return image_lr, image_hr


class SRDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.batch_size = batch_size
        self.image_size = image_size
        
    def setup(self, stage=None):
        self.train_dataset = SRDataset(self.train_dataset_path, self.image_size)
        self.val_dataset = SRDataset(self.val_dataset_path, self.image_size)    
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        return 

    def predict_dataloader(self):
        return
 