import argparse
from torchvision import transforms

from PIL import Image

from train import SRGANTrain
from datamodule import *
from utils import *
import yaml

def inference(config,args):
    srgan_train = SRGANTrain.load_from_checkpoint(args.path_ckpt, config = config)
    
    image = Image.open(args.path_png).convert("RGB")
    image_hr = transforms.functional.to_tensor(image)
    
    image_lr = transforms.functional.resize(image, size = (image_hr.shape[1]//4, image_hr.shape[2]//4), interpolation=Image.BICUBIC)
    image_lr.save(f"{os.path.dirname(args.path_png)}/{get_filename(args.path_png)[0]}_lr.png")
    image_lr = transforms.functional.to_tensor(image_lr)
   
    image_sr = srgan_train.forward(image_lr.unsqueeze(0))
    image_sr = transforms.functional.to_pil_image(image_sr.squeeze(0))
    
    image_sr.save(f"{os.path.dirname(args.path_png)}/{get_filename(args.path_png)[0]}_sr.png")
    


if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_ckpt", type = str, default = "./output/final_model.ckpt" )
    parser.add_argument("--path_png", type = str)
    parser.add_argument("--path_config", type = str, default = "./config.yaml")
    args = parser.parse_args()
    
    config = yaml.load(open(args.path_config, 'r'), Loader=yaml.FullLoader)
    
    inference(config, args)

