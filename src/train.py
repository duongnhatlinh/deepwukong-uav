from os.path import basename, join
import torch
from omegaconf import DictConfig
import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

def train(model: L.LightningModule, data_module: L.LightningDataModule,
          config: DictConfig):
    # Define logger
    model_name = model.__class__.__name__
    dataset_name = basename(config.dataset.name)
    
    # tensorboard logger
    tensorlogger = TensorBoardLogger(
        save_dir=join("ts_logger", model_name),
        name=dataset_name
    )
    
    # define model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=join(tensorlogger.log_dir, "checkpoints"),
        monitor="val_loss",
        filename="{epoch:02d}-{step:02d}-{val_loss:.4f}",
        save_top_k=5,
        mode="min"
    )
    
    early_stopping_callback = EarlyStopping(
        patience=config.hyper_parameters.patience,
        monitor="val_loss",
        verbose=True,
        mode="min"
    )
    
    lr_logger = LearningRateMonitor("step")
    
    # Setup trainer arguments
    trainer_args = {
        "max_epochs": config.hyper_parameters.n_epochs,
        "gradient_clip_val": config.hyper_parameters.clip_norm,
        "deterministic": True,
        "val_check_interval": config.hyper_parameters.val_every_step,
        "log_every_n_steps": config.hyper_parameters.log_every_n_steps,
        "logger": [tensorlogger],
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "callbacks": [
            lr_logger,
            early_stopping_callback,
            checkpoint_callback
        ]
    }
    
    # Handle resume from checkpoint if specified
    if hasattr(config.hyper_parameters, 'resume_from_checkpoint') and \
       config.hyper_parameters.resume_from_checkpoint:
        trainer = L.Trainer(**trainer_args)
        trainer.fit(
            model=model, 
            datamodule=data_module,
            ckpt_path=config.hyper_parameters.resume_from_checkpoint
        )
    else:
        trainer = L.Trainer(**trainer_args)
        trainer.fit(model=model, datamodule=data_module)
    
    trainer.test(model=model, datamodule=data_module)

