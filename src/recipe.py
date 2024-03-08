from typing import Dict, List, cast, Tuple, Any
from dataclasses import dataclass

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from PIL import Image
from pathlib import Path
from pytorch_lightning.core.optimizer import LightningOptimizer

from src.models.arch import StainFuserArchitecture


@dataclass
class OptimizerConfig:
    # target_model: str
    optimizer: torch.optim.Optimizer
    optimizer_kwargs: Dict[str, object]
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scheduler_kwargs: Dict[str, object]
    param_dict: Dict[str, object]


class StainFuserEngine(pl.LightningModule):
    """
    Note:
        Do not mix definition of `models` with their training!

    Args:
        arch (stainFuserArchitecture): A dictionary of `nn.Module` that this
            engine defines the training information for.
        optim_configs (List[OptimizerConfig]): List of optimizer configurations.
    """

    def __init__(
        self,
        arch: StainFuserArchitecture,
        optim_configs: List[OptimizerConfig],
    ) -> None:
        super().__init__()
        self.arch = arch
        self.optim_configs = optim_configs

        # !!! Important:
        # Set this property to `False` to set PytorchLightning
        # to use manual loss definition in training step. This
        # is important when using many optimizers and procedure
        # like GAN.
        self.automatic_optimization = False

        self.optimizer_mapping = {
            "noise_predictor_unlock_decoder": {
                "target_model": "noise_predictor",
                "module_list": ["up_blocks", "conv_norm_out", "conv_act", "conv_out"],
            },
            "noise_predictor_unlock_encoder": {
                "target_model": "noise_predictor",
                "module_list": [
                    "conv_in",
                    "time_proj",
                    "time_embedding",
                    "down_blocks",
                    "mid_block",
                ],
            },
            "noise_predictor_encoder_projector": {
                "target_model": "noise_predictor",
                "module_list": ["encoder_projector"],
            },
        }

        # Define training loop parameters

    # def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
    #     print("batch_idx", batch_idx)
    #     if (batch_idx + 1) % 10 == 0:
    #         print('training stopped')
    #         return -1

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[dict]]:
        """
        Pytorch Lightning will call this function to retrieve the
        optimizers and their respective schedulers.
        """
        # Optimizer creation

        param_list = []
        for idx, param_name in enumerate(self.optim_configs.param_dict):
            # print(param_name, self.optim_configs.param_dict[param_name])
            if param_name not in self.optimizer_mapping:
                submodel = getattr(self.arch, param_name)
                if not isinstance(submodel, nn.Module):
                    raise ValueError(
                        f"Value for `target_model` for the {idx}-th optimizer must"
                        " be an attribute defined as `nn.Module` within"
                        " `self.arch`."
                    )
                params_to_optimize = submodel.parameters()
                param_list.append(
                    {
                        "params": params_to_optimize,
                        **self.optim_configs.param_dict[param_name],
                    }
                )
                print(param_name, "Added to optimiser")
            else:
                target_model_name = self.optimizer_mapping[param_name]["target_model"]
                module_list = self.optimizer_mapping[param_name]["module_list"]

                submodel = getattr(self.arch, target_model_name)
                if not isinstance(submodel, nn.Module):
                    raise ValueError(
                        f"Value for `target_model` for the {idx}-th optimizer must"
                        " be an attribute defined as `nn.Module` within"
                        " `self.arch`."
                    )
                params_to_optimize = []
                for name, parameters in self.named_parameters():
                    for module_name in module_list:
                        if f"{target_model_name}.{module_name}" in name:
                            params_to_optimize.append(parameters)
                            break
                param_list.append(
                    {
                        "params": params_to_optimize,
                        **self.optim_configs.param_dict[param_name],
                    }
                )
                print(module_list, "Added to optimiser")

        optimizer = self.optim_configs.optimizer(
            param_list, **self.optim_configs.optimizer_kwargs
        )
        scheduler = self.optim_configs.scheduler(
            optimizer, **self.optim_configs.scheduler_kwargs
        )
        # refert to Pytorch Lightning `configure_optimizers` for additional
        # customized keywords
        # https://lightning.ai/docs/pytorch/latest/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.configure_optimizers
        # https://lightning.ai/docs/pytorch/latest/model/build_model_advanced.html#learning-rate-scheduling
        scheduler_config = {
            "scheduler": scheduler,
        }
        return [optimizer], [scheduler_config]

    def retrieve_loss_target(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the target for loss depending on the prediction type
        defined within the noise scheduler.
        """
        scheduler = self.arch.noise_scheduler
        # Get the target for loss depending on the prediction type
        if scheduler.config.prediction_type == "epsilon":
            target = noise
        elif scheduler.config.prediction_type == "v_prediction":
            target = scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                "Unknown prediction type"
                f" `{scheduler.config.prediction_type}` in "
                "`self.arch.noise_scheduler`"
            )
        return target

    def prepare_latent(self, batch_input):
        """
        Prepare latent representations.
        """
        # in NCHW
        # batch_images = batch_input.type(self.arch.weight_dtype)
        batch_images = batch_input

        # * ===== Prepare Representation
        # Convert images to latent space
        with torch.no_grad():
            latents = self.arch.encoder.encode(batch_images)
        latents = latents.latent_dist.sample()

        latents = latents * self.arch.encoder.config.scaling_factor
        # latents  # shape: (N, 4, H/8, W/8)
        return latents

    def get_input(self, batch):
        """
        Get input for training.
        """
        tgt_latents = self.prepare_latent(batch["Target Images"])

        tsf_latents = self.prepare_latent(batch["Transformed Images"])

        src_latents = batch["Source Images"]
        return src_latents, tgt_latents, tsf_latents

    def get_input_for_inference(self, batch):
        """
        Get input for inference.
        """
        tgt_latents = self.prepare_latent(batch["Target Images"])

        src_latents = batch["Source Images"]
        return src_latents, tgt_latents

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """

        Note:
            To debug this function, must set the `Trainer` to `accelerator="cpu"`
            or manually putting `breakpoint()` at the location.

        Args:
            batch (Dict[str, torch.Tensor]): _description_
            batch_idx (int): _description_

        """
        optimizer = cast(LightningOptimizer, self.optimizers())
        optimizer.zero_grad()

        batch_size = int(batch["Source Images"].shape[0])

        src_latents, tgt_latents, tsf_latents = self.get_input(batch=batch)

        # * ===== Prepare Diffusion Process
        # Sample noise that we will add to the latents
        noise = torch.randn_like(tsf_latents)
        # Sample a random timestep for each image
        min_timesteps: int = 0
        max_timesteps: int = self.arch.noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(
            min_timesteps,
            max_timesteps,
            (batch_size,),
            device=self.device,
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at
        # each timestep (this is the forward diffusion process)
        # this latents should be the expected output given the source
        # and conditioning information
        noisy_tsf_latents = self.arch.noise_scheduler.add_noise(
            tsf_latents, noise, timesteps
        )

        # * ===== Start Diffusion Process
        # breakpoint()

        _, latent_channels, _, _ = tgt_latents.shape
        # print(tgt_latents.shape)
        # N x C x H*W
        tgt_latents = tgt_latents.reshape(batch_size, latent_channels, -1)
        # N x H*W x C i.e. (batch_size, sequence_length, hidden_size)
        tgt_latents = tgt_latents.permute(0, 2, 1)

        (down_block_res_samples, mid_block_res_sample) = self.arch.conditioner(
            noisy_tsf_latents,
            timesteps,
            encoder_hidden_states=tgt_latents,
            controlnet_cond=src_latents,
            return_dict=False,
        )

        # Predict the noise residual
        model_pred = self.arch.noise_predictor(
            noisy_tsf_latents,
            timesteps,
            encoder_hidden_states=tgt_latents,
            down_block_additional_residuals=down_block_res_samples,
            mid_block_additional_residual=mid_block_res_sample,
        )

        loss_target = self.retrieve_loss_target(
            latents=tsf_latents, noise=noise, timesteps=timesteps
        )
        loss = F.mse_loss(
            # model_pred.type(self.arch.weight_dtype),
            # loss_target.type(self.arch.weight_dtype),
            model_pred,
            loss_target,
            reduction="mean",
        )

        # * ===== Start Optimization (backward)
        # `self.manual_backward(loss)` is compatible with normal pytorch
        # loop `loop.backward()`. However, the added benefits are automatic
        # rescaling wrt precision etc.
        self.manual_backward(loss)

        # Gradient accumulation or Gradient Clipping during mixed training
        # should be defined or called here before `step`
        optimizer.step()

        # retrieve scheduler
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()

        # print("here")

    def denoising(
        self, batch: Dict[str, Any], diffusion_steps: int
    ) -> Tuple[Any, torch.Tensor]:
        # 1. Prepare noise scheduler
        noise_scheduler = self.arch.noise_scheduler

        # 2. Prepare timesteps
        noise_scheduler.set_timesteps(diffusion_steps, device=self.device)
        timesteps = noise_scheduler.timesteps

        # 3. Prepare inputs. (source images, target latents and transformed latents)
        src_latents, tgt_latents, tsf_latents = self.get_input(batch=batch)
        batch_size = src_latents.shape[0]

        # 4. Flatten target latents to word vec
        _, latent_channels, _, _ = tgt_latents.shape
        # N x C x H*W
        tgt_latents = tgt_latents.reshape(batch_size, latent_channels, -1)
        # N x H*W x C i.e. (batch_size, sequence_length, hidden_size)
        tgt_latents = tgt_latents.permute(0, 2, 1)

        # 5. Prepare noise latents
        noise_latents = torch.randn_like(tsf_latents)
        noise_latents = noise_latents * noise_scheduler.init_noise_sigma

        # 6. Denoising loop
        for i, t in enumerate(timesteps):
            latent_model_input = noise_scheduler.scale_model_input(noise_latents, t)
            control_model_input = latent_model_input

            down_block_res_samples, mid_block_res_sample = self.arch.conditioner(
                control_model_input,
                t,
                encoder_hidden_states=tgt_latents,
                controlnet_cond=src_latents,
                return_dict=False,
            )

            # predict the noise residual
            noise_pred = self.arch.noise_predictor(
                latent_model_input,
                t,
                encoder_hidden_states=tgt_latents,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

            # compute the previous noisy sample x_t -> x_t-1
            noise_latents = noise_scheduler.step(
                noise_pred, t, noise_latents, return_dict=False
            )[0]

        # 7. Decode the prediction.
        output = self.decode_first_stage(noise_latents)
        return output, noise_latents

    def denoising_inference(
        self, batch: Dict[str, Any], diffusion_steps: int
    ) -> Tuple[Any, torch.Tensor]:
        # 1. Prepare noise scheduler
        noise_scheduler = self.arch.noise_scheduler

        # 2. Prepare timesteps
        noise_scheduler.set_timesteps(diffusion_steps, device=self.device)
        timesteps = noise_scheduler.timesteps

        # 3. Prepare inputs. (source images, target latents and transformed latents)
        src_latents, tgt_latents = self.get_input_for_inference(batch=batch)
        batch_size = src_latents.shape[0]

        noise_latents = torch.randn_like(tgt_latents)

        # 4. Flatten target latents to word vec
        _, latent_channels, _, _ = tgt_latents.shape
        # N x C x H*W
        tgt_latents = tgt_latents.reshape(batch_size, latent_channels, -1)
        # N x H*W x C i.e. (batch_size, sequence_length, hidden_size)
        tgt_latents = tgt_latents.permute(0, 2, 1)

        # 5. Prepare noise latents
        noise_latents = noise_latents * noise_scheduler.init_noise_sigma

        # 6. Denoising loop
        for i, t in enumerate(timesteps):
            latent_model_input = noise_scheduler.scale_model_input(noise_latents, t)
            control_model_input = latent_model_input

            down_block_res_samples, mid_block_res_sample = self.arch.conditioner(
                control_model_input,
                t,
                encoder_hidden_states=tgt_latents,
                controlnet_cond=src_latents,
                return_dict=False,
            )

            # predict the noise residual
            noise_pred = self.arch.noise_predictor(
                latent_model_input,
                t,
                encoder_hidden_states=tgt_latents,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

            # compute the previous noisy sample x_t -> x_t-1
            noise_latents = noise_scheduler.step(
                noise_pred, t, noise_latents, return_dict=False
            )[0]

        # 7. Decode the prediction.
        output = self.decode_first_stage(noise_latents)
        return output, noise_latents

    @torch.no_grad()
    def decode_first_stage(self, z: torch.Tensor) -> torch.Tensor:
        z = (1.0 / self.arch.encoder.config.scaling_factor) * z
        return self.arch.encoder.decode(z).sample


class InferenceRecipe(pl.LightningModule):
    """
    Note:
        Do not mix definition of `models` with their training!

    Args:
        arch (dict): A dictionary of `nn.Module` that this
            engine defines the training information for.

    """

    def __init__(
        self,
        arch: StainFuserArchitecture,
        optim_configs: List[OptimizerConfig],
        output_dir: str,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.optim_configs = optim_configs
        self.output_dir = output_dir

        # !!! Important:
        # Set this property to `False` to set PytorchLightning
        # to use manual loss definition in training step. This
        # is important when using many optimizers and procedure
        # like GAN.
        self.automatic_optimization = False

        self.optimizer_mapping = {
            "noise_predictor_unlock_decoder": {
                "target_model": "noise_predictor",
                "module_list": ["up_blocks", "conv_norm_out", "conv_act", "conv_out"],
            },
            "noise_predictor_unlock_encoder": {
                "target_model": "noise_predictor",
                "module_list": [
                    "conv_in",
                    "time_proj",
                    "time_embedding",
                    "down_blocks",
                    "mid_block",
                ],
            },
            "noise_predictor_encoder_projector": {
                "target_model": "noise_predictor",
                "module_list": ["encoder_projector"],
            },
        }

        # Define training loop parameters

    # def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
    #     print("batch_idx", batch_idx)
    #     if (batch_idx + 1) % 10 == 0:
    #         print('training stopped')
    #         return -1

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[dict]]:
        """Pytorch Lightning will call this function to retrieve the
        optimizers and their respective schedulers."""
        # Optimizer creation

        param_list = []
        # for idx, param_name in enumerate(self.optim_configs.param_dict):
        #     # print(param_name, self.optim_configs.param_dict[param_name])
        #     if param_name not in self.optimizer_mapping:
        #         submodel = getattr(self.arch, param_name)
        #         if not isinstance(submodel, nn.Module):
        #             raise ValueError(
        #                 f"Value for `target_model` for the {idx}-th optimizer must"
        #                 " be an attribute defined as `nn.Module` within"
        #                 " `self.arch`."
        #             )
        #         params_to_optimize = submodel.parameters()
        #         param_list.append(
        #             {
        #                 "params": params_to_optimize,
        #                 **self.optim_configs.param_dict[param_name],
        #             }
        #         )
        #         print(param_name, 'Added to optimiser')
        #     else:
        #         target_model_name = self.optimizer_mapping[param_name]["target_model"]
        #         module_list = self.optimizer_mapping[param_name]["module_list"]

        #         submodel = getattr(self.arch, target_model_name)
        #         if not isinstance(submodel, nn.Module):
        #             raise ValueError(
        #                 f"Value for `target_model` for the {idx}-th optimizer must"
        #                 " be an attribute defined as `nn.Module` within"
        #                 " `self.arch`."
        #             )
        #         params_to_optimize = []
        #         for name, parameters in self.named_parameters():
        #             for module_name in module_list:
        #                 if f"{target_model_name}.{module_name}" in name:
        #                     params_to_optimize.append(parameters)
        #                     break
        #         param_list.append(
        #             {
        #                 "params": params_to_optimize,
        #                 **self.optim_configs.param_dict[param_name],
        #             }
        #         )
        #         print(module_list, 'Added to optimiser')

        # optimizer = self.optim_configs.optimizer(
        #     param_list, **self.optim_configs.optimizer_kwargs
        # )
        # scheduler = self.optim_configs.scheduler(
        #     optimizer, **self.optim_configs.scheduler_kwargs
        # )
        # # refert to Pytorch Lightning `configure_optimizers` for additional
        # # customized keywords
        # # https://lightning.ai/docs/pytorch/latest/api/pytorch_lightning.core.LightningModule.html#pytorch_lightning.core.LightningModule.configure_optimizers
        # # https://lightning.ai/docs/pytorch/latest/model/build_model_advanced.html#learning-rate-scheduling
        # scheduler_config = {
        #     "scheduler": scheduler,
        # }
        # return [optimizer], [scheduler_config]
        return [], []

    def prepare_latent(self, batch_input):
        # in NCHW
        # batch_images = batch_input.type(self.arch.weight_dtype)
        batch_images = batch_input

        # * ===== Prepare Representation
        # Convert images to latent space
        with torch.no_grad():
            latents = self.arch.encoder.encode(batch_images)
        latents = latents.latent_dist.sample()

        latents = latents * self.arch.encoder.config.scaling_factor
        # latents  # shape: (N, 4, H/8, W/8)
        return latents

    def get_input(self, batch):
        tgt_latents = self.prepare_latent(batch["Target Images"])

        tsf_latents = self.prepare_latent(batch["Transformed Images"])

        # in NCHW
        # src_latents = batch["Source Images"].type(self.arch.weight_dtype)
        src_latents = batch["Source Images"]
        return src_latents, tgt_latents, tsf_latents

    def training_step(self, batch: Dict[str, object], batch_idx: int):
        """

        Note:
            To debug this function, must set the `Trainer` to `accelerator="cpu"`
            or manually putting `breakpoint()` at the location.

        Args:
            batch (Dict[str, torch.Tensor]): _description_
            batch_idx (int): _description_

        """

        batch_item = {
            "Source Images": batch["Source Images"],
            "Target Images": batch["Target Images"],
            "Transformed Images": batch["Target Images"],  # just a placeholder
        }

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        with torch.no_grad():
            start_event.record()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output, noise_latents = self.denoising(
                    batch=batch_item, diffusion_steps=10
                )
            end_event.record()
            torch.cuda.synchronize()
        inference_time = (
            start_event.elapsed_time(end_event) / 1000.0
        )  # Convert to seconds

        file = open(f"{self.output_dir}/time_log.csv", "+a")
        file.write(f"{batch_idx},{inference_time}\n")
        file.close()

        output = output.transpose(1, 2).transpose(2, 3)
        output = (output + 1.0) / 2.0
        output = output.detach().cpu().numpy()
        output = (output * 255).astype(np.uint8)

        # with Pool(processes=32) as pool:
        #     params = [(item, self.output_dir, batch['Source path'][idx], batch['Target path'][idx]) for idx, item in enumerate(output)]
        #     pool.starmap(self.save_image, params)
        for idx, item in enumerate(output):
            self.save_image(
                item,
                self.output_dir,
                batch["Source path"][idx],
                batch["Target path"][idx],
            )

    def save_image(self, item, output_dir, source_path, target_path):
        image = Image.fromarray(item)
        source_name = Path(source_path).stem
        target_name = Path(target_path).stem
        save_folder = os.path.join(output_dir, target_name)
        os.makedirs(save_folder, exist_ok=True)
        image.save(os.path.join(save_folder, f"{source_name}.png"))

    def denoising(self, batch, diffusion_steps):
        # 1. Prepare noise scheduler
        noise_scheduler = self.arch.noise_scheduler

        # 2. Prepare timesteps
        noise_scheduler.set_timesteps(diffusion_steps, device=self.device)
        timesteps = noise_scheduler.timesteps

        # 3. Prepare inputs. (source images, target latents and transformed latents)
        src_latents, tgt_latents, tsf_latents = self.get_input(batch=batch)
        batch_size = src_latents.shape[0]

        # 4. Flatten target latents to word vec
        _, latent_channels, _, _ = tgt_latents.shape
        # N x C x H*W
        tgt_latents = tgt_latents.reshape(batch_size, latent_channels, -1)
        # N x H*W x C i.e. (batch_size, sequence_length, hidden_size)
        tgt_latents = tgt_latents.permute(0, 2, 1)

        # 5. Prepare noise latents
        noise_latents = torch.randn_like(tsf_latents)
        noise_latents = noise_latents * noise_scheduler.init_noise_sigma

        # 6. Denoising loop
        for i, t in enumerate(timesteps):
            latent_model_input = noise_scheduler.scale_model_input(noise_latents, t)
            control_model_input = latent_model_input

            down_block_res_samples, mid_block_res_sample = self.arch.conditioner(
                control_model_input,
                t,
                encoder_hidden_states=tgt_latents,
                controlnet_cond=src_latents,
                return_dict=False,
            )

            # predict the noise residual
            noise_pred = self.arch.noise_predictor(
                latent_model_input,
                t,
                encoder_hidden_states=tgt_latents,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

            # compute the previous noisy sample x_t -> x_t-1
            noise_latents = noise_scheduler.step(
                noise_pred, t, noise_latents, return_dict=False
            )[0]

        # 7. Decode the prediction.
        output = self.decode_first_stage(noise_latents)
        return output, noise_latents

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = (1.0 / self.arch.encoder.config.scaling_factor) * z
        return self.arch.encoder.decode(z).sample
