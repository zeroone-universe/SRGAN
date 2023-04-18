from datamodule import SRDataModule

from train import SRGANTrain

from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl

from utils import *
import yaml


def main(config):
    
    pl.seed_everything(config['random_seed'], workers=True)
    sr_datamodule = SRDataModule(config)
    srgan_train = SRGANTrain(config)
    
    check_dir_exist(config['train']['output_dir_path'])
    check_dir_exist(config['train']['logger_path'])
    
    tb_logger = pl_loggers.TensorBoardLogger(config['train']['logger_path'], name=f"SRGAN_logs")
    
    #-----textlogger-----
    # textlogger = logging.getLogger("TEXT")
    # textlogger.setLevel(logging.INFO)

    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # file_handler = logging.FileHandler(f"{LOGGER_PATH}/modelconfig.log")
    # file_handler.setFormatter(formatter)
    # textlogger.addHandler(file_handler)

    # textlogger.info(train_codecpp.get_config())

    # progress_bar = ProgressBar()
    # progress_bar.log_to_file = True

    tb_logger.log_hyperparams(config)

    trainer=pl.Trainer(devices=1, accelerator="gpu",
    max_steps=config['train']['total_step'],
    logger=tb_logger,
    check_val_every_n_epoch=config['train']['val_epoch'],
    profiler = "simple"
    )
    
    trainer.fit(srgan_train, sr_datamodule)
    trainer.save_checkpoint(os.path.join(config['train']['output_dir_path'],'final_model.ckpt'))
    
    trainer.test(srgan_train, sr_datamodule)
    
if __name__ == "__main__":
    config = yaml.load(open("/media/youngwon/Neo/NeoChoi/Projects/SRGAN/config.yaml", 'r'), Loader=yaml.FullLoader)
    
    main(config)
    
    