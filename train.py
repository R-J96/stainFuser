from omegaconf import DictConfig
import os

import pytorch_lightning as pl
import torch
import hydra
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins.environments import ClusterEnvironment

from src.dataset import TrainDataset
from src.models.arch import OveriddenUnet2D, OveriddenControlNet, StainFuserArchitecture
from src.recipe import StainFuserEngine, OptimizerConfig

from src.logger import ImageLogger

from src.misc.utils import get_type_from_config, get_kwargs_from_config, recur_find_ext


@hydra.main(version_base="1.3", config_path="conf/", config_name="train")
def main(cfg: DictConfig) -> None:
    """
    Main function for training the model.

    Args:
        cfg (DictConfig): Configuration settings.

    Returns:
        None
    """
    assert not os.path.exists(os.path.join(cfg.paths.output_dir, "weights"))

    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    # Load noise scheduler configuration
    scheduler_config = torch.load(cfg.model.net.noise_scheduler_path)
    scheduler_config["config"] = dict(scheduler_config["config"])
    scheduler_config["config"].pop("clip_sample")
    noise_scheduler = hydra.utils.instantiate(
        {**cfg.model.noise_scheduler, **scheduler_config["config"]}
    )

    NOISE_PREDICTOR_DEC_UNLOCK = cfg.model.net.unlock_decoder
    NOISE_PREDICTOR_ENC_UNLOCK = cfg.model.net.unlock_encoder

    # Load pretrained VAE and UNet
    vae_pretrained = torch.load(cfg.model.net.vae_path)
    vae = AutoencoderKL(**vae_pretrained["config"])
    vae.load_state_dict(vae_pretrained["weights"])
    vae.requires_grad_(False)

    unet_pretrained = torch.load(cfg.model.net.unet_path)
    unet = UNet2DConditionModel(**unet_pretrained["config"])
    unet.load_state_dict(unet_pretrained["weights"])
    noise_predictor = OveriddenUnet2D(unet, encoder_hid_dim=4)
    controlnet = OveriddenControlNet(
        unet,
        cross_attention_dim=4,
        conditioning_embedding_in_channels=4,
    )

    # Initialise the architecture
    arch = StainFuserArchitecture(
        encoder=vae,
        conditioner=controlnet,
        noise_predictor=noise_predictor,
        noise_scheduler=noise_scheduler,
    )

    optimizer_kwargs = get_kwargs_from_config(cfg.model.optimizer)
    param_dict = dict()
    param_dict["conditioner"] = optimizer_kwargs
    param_dict["noise_predictor_encoder_projector"] = optimizer_kwargs
    if NOISE_PREDICTOR_ENC_UNLOCK:
        param_dict["noise_predictor_unlock_encoder"] = optimizer_kwargs
    if NOISE_PREDICTOR_DEC_UNLOCK:
        param_dict["noise_predictor_unlock_decoder"] = optimizer_kwargs

    # set up optimizer configurations
    optim_config = OptimizerConfig(
        optimizer=get_type_from_config(cfg.model.optimizer._target_),
        optimizer_kwargs=optimizer_kwargs,
        scheduler=get_type_from_config(cfg.model.scheduler._target_),
        scheduler_kwargs=get_kwargs_from_config(cfg.model.scheduler),
        param_dict=param_dict,
    )

    # initialise the training engine
    engine = StainFuserEngine(arch=arch, optim_configs=optim_config)

    ds: TrainDataset = hydra.utils.instantiate(cfg.data.dataset)

    # create dataloader
    loader = torch.utils.data.DataLoader(
        ds,
        num_workers=cfg.data.loader.num_workers,
        batch_size=cfg.data.loader.batch_size,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=cfg.data.loader.prefetch_factor,
    )

    # Save training state every epoch for resuming if needed
    state_callback: ModelCheckpoint = hydra.utils.instantiate(
        cfg.logger.state_checkpoint
    )

    callbacks = [
        state_callback,
    ]

    if cfg.get("image_log"):
        # save images during training
        image_callback: ImageLogger = hydra.utils.instantiate(cfg.logger.image_logger)
        callbacks.append(image_callback)

    if cfg.get("weights_steps_log"):
        # save weights every n steps
        weights_step_callback: ModelCheckpoint = hydra.utils.instantiate(
            cfg.logger.weights_checkpoint_step
        )
        callbacks.append(weights_step_callback)

    if cfg.get("weights_epoch_log"):
        # Save weights every n epoch
        weights_epoch_callback: ModelCheckpoint = hydra.utils.instantiate(
            cfg.logger.weights_checkpoint_epoch
        )
        callbacks.append(weights_epoch_callback)

    if cfg.get("slurm"):
        plugins: ClusterEnvironment = hydra.utils.instantiate(cfg.plugins)

        trainer: pl.Trainer = hydra.utils.instantiate(
            cfg.trainer,
            strategy=DDPStrategy(find_unused_parameters=False),
            callbacks=callbacks,
            logger=False,
            plugins=plugins,
        )
    else:
        trainer: pl.Trainer = hydra.utils.instantiate(
            cfg.trainer, callbacks=callbacks, logger=False
        )

    if cfg.get("compile"):
        engine = torch.compile(engine)

    if cfg.get("resume"):
        checkpoint_path = recur_find_ext(
            cfg.logger.state_checkpoint.dirpath, [".ckpt"]
        )[0]
        print("Resuming from checkpoint")
        trainer.fit(model=engine, train_dataloaders=loader, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model=engine, train_dataloaders=loader)

    return


if __name__ == "__main__":
    main()
    # ! for debugging
    # from hydra import compose, initialize
    # from omegaconf import OmegaConf
    # # from hydra import HydraConfig
    # with initialize(version_base=None, config_path="conf/"):
    #     cfg = compose(config_name="debug", return_hydra_config=True)
    #     # OmegaConf.resolve(cfg)
    # print(cfg.hydra.run.dir)
    # pass
    # # main(cfg)
