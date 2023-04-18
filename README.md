# SRGAN Implementation

This is an unofficial pytorch lightning implementation of SRGAN (Super-Resolution Generative Adversarial Network), a deep learning model for image super-resolution. The model is trained on the DIV2K dataset and can upscale images by a factor of 4x.

This code was developed for the final project of the Computer Vision course (EC4216) at GIST in 2021, and has recently been modified to comply with the pytorch lightning 2.0.0 framework.

## Requirements
 
To run this code, you will need:

- torch==2.0.0
- torchvision==0.15.0
- Pillow==9.5.0
- pytorch_lightning==2.0.0
- PyYAML==6.0

To automatically install these libraries, run the following command:

```pip install -r requirements.txt```

## Usage

To run the code on your own machine, follow these steps:

1. Open the 'config.yaml' file and modify the file paths (and hyperparameters as needed).
2. Run the 'main.py' file to start training the model.

The trained model will be saved as ckpt file in 'logger' directory. You can then use the trained model to perform the image super-resolution on your own png wav file by running the 'inference.py' file as

```python inference.py --path_png <path of png file> --path_ckpt <path of checkpoint file> --path_config <path of config.yaml>```


## Note
- Due to the limited dataset and low computer system during the practice environment, the performance of the code has not been thoroughly validated.

## Citation

```bibtex
@article{ledig2017photo,
  title={Photo-realistic single image super-resolution using a generative adversarial network},
  author={Ledig, Christian and Theis, Lucas and Husz{\'a}r, Ferenc and Caballero, Jose and Cunningham, Andrew and Acosta, Alejandro and Aitken, Andrew and Tejani, Alykhan and Totz, Johannes and Wang, Zehan and others},
  journal={arXiv preprint arXiv:1609.04802},
  year={2017}
}
```

