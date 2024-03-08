from typing import Tuple
import torch
import numpy as np

from tiatoolbox.models.models_abc import ModelABC

from src.misc.utils import inverse_transform
from src.models.arch import StainFuserArchitecture


class WSIStainFuser(ModelABC):
    """
    Wrapper inheriting from toolbox for WSI stuff
    """

    def __init__(
        self,
        stainFuser: StainFuserArchitecture,
        target: np.ndarray,
        diffusion_steps: int = 20,
        fp16: bool = False,
    ) -> None:
        """
        Initialize the WSIStainFuser model.

        Args:
            stainFuser (StainFuserArchitecture): The stainFuser instance.
            target (np.ndarray): The target image.
            diffusion_steps (int): The number of diffusion steps.
            fp16 (bool, optional): Whether to use mixed precision (fp16) or not. Defaults to False.
        """
        super().__init__()
        self.stainFuser = stainFuser
        self.target = target
        self.diffusion_steps = diffusion_steps
        self.fp16 = fp16

    @staticmethod
    def preproc(image: np.ndarray) -> np.ndarray:
        """Pre-processing function."""
        return image

    @staticmethod
    def postproc(image: np.ndarray) -> np.ndarray:
        """Post-processing function."""
        return image

    def _prepare_latent(self, batch_input):
        """
        Prepare latent representations.
        """
        # in NCHW
        # batch_images = batch_input.type(self.arch.weight_dtype)
        batch_images = batch_input

        # * ===== Prepare Representation
        # Convert images to latent space
        with torch.no_grad():
            latents = self.stainFuser.encoder.encode(batch_images)
        latents = latents.latent_dist.sample()

        latents = latents * self.stainFuser.encoder.config.scaling_factor
        # latents  # shape: (N, 4, H/8, W/8)
        return latents

    def _get_input_for_inference(self, batch):
        """
        Get input for inference.
        """
        tgt_latents = self._prepare_latent(batch["Target Images"])

        src_latents = batch["Source Images"]
        return src_latents, tgt_latents

    def forward(self, x, diffusion_steps, target_image, device):
        """Model forward function."""
        # 1. Prepare noise scheduler
        noise_scheduler = self.stainFuser.noise_scheduler

        # 2. Prepare timesteps
        noise_scheduler.set_timesteps(diffusion_steps, device=device)
        timesteps = noise_scheduler.timesteps

        # 2a. Prepare batch
        batch = {
            "Target Images": target_image.unsqueeze(0)
            .repeat(x.shape[0], 1, 1, 1)
            .to(device),
            "Source Images": x.to(device),
        }

        # 3. Prepare inputs. (source images, target latents and transformed latents)
        src_latents, tgt_latents = self._get_input_for_inference(batch=batch)
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

            down_block_res_samples, mid_block_res_sample = self.stainFuser.conditioner(
                control_model_input,
                t,
                encoder_hidden_states=tgt_latents,
                controlnet_cond=src_latents,
                return_dict=False,
            )

            # predict the noise residual
            noise_pred = self.stainFuser.noise_predictor(
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
        output = self._decode_first_stage(noise_latents)
        return output

    @torch.no_grad()
    def _decode_first_stage(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent representation to the output image.

        Args:
            z (torch.Tensor): The latent representation.

        Returns:
            torch.Tensor: The decoded output image.
        """
        z = (1.0 / self.stainFuser.encoder.config.scaling_factor) * z
        return self.stainFuser.encoder.decode(z).sample

    def infer_batch(
        self,
        model: torch.nn.Module,
        batch_data: np.ndarray,
        on_gpu: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform inference on a batch of data.

        Args:
            model (torch.nn.Module): The model to use for inference.
            batch_data (np.ndarray): The input batch data.
            on_gpu (bool): Whether to run inference on GPU or not.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The predicted R, G, and B channels.
        """
        patch_imgs = batch_data

        device = "cuda" if on_gpu else "cpu"
        patch_imgs_ = patch_imgs.to(device).type(torch.float32)
        patch_imgs_ = patch_imgs_.permute(0, 3, 1, 2).contiguous()

        model.eval()

        with torch.no_grad():
            patch_imgs_ = patch_imgs_ - patch_imgs_.min()
            patch_imgs_ = patch_imgs_ / patch_imgs_.max()
            if self.fp16:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    patch_predictions = model.forward(
                        patch_imgs_, self.diffusion_steps, self.target, device
                    )
            else:
                patch_predictions = model.forward(
                    patch_imgs_, self.diffusion_steps, self.target, device
                )

        patch_predictions = inverse_transform(
            patch_predictions, return_numpy=True, is_tsf=True
        )
        return (
            patch_predictions[:, :, :, 0],
            patch_predictions[:, :, :, 1],
            patch_predictions[:, :, :, 2],
        )
