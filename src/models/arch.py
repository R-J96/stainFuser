from typing import Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn

from diffusers.models.controlnet import ControlNetModel, ControlNetOutput
from diffusers.models.unets.unet_2d_condition import (
    UNet2DConditionModel,
    UNet2DConditionOutput,
)
from diffusers import (
    AutoencoderKL,
    PNDMScheduler,
)


class OveriddenControlNet(ControlNetModel):
    """
    Overwriting certain operations of the original `ControlNetModel` so that
    pretrained `UNet2DConditionModel` can be loaded.

    Args:
        pretrained_unet (UNet2DConditionModel): Pretrained UNet2DConditionModel instance.
        controlnet_conditioning_channel_order (str, optional): Order of conditioning channels for ControlNetModel. Defaults to "rgb".
        conditioning_embedding_out_channels (Tuple[int, ...], optional): Output channels for conditioning embeddings. Defaults to (16, 32, 96, 256).
        conditioning_embedding_in_channels (int, optional): Number of input channels for conditioning embeddings. Defaults to 3.
        cross_attention_dim (int, optional): Dimensionality of cross attention. Defaults to 1024.
    """

    def __init__(
        self,
        pretrained_unet: UNet2DConditionModel,
        controlnet_conditioning_channel_order: str = "rgb",
        conditioning_embedding_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
        conditioning_embedding_in_channels: int = 3,
        cross_attention_dim: int = 1024,
    ) -> None:
        cfg = pretrained_unet.config
        super().__init__(
            in_channels=cfg.in_channels,
            flip_sin_to_cos=cfg.flip_sin_to_cos,
            freq_shift=cfg.freq_shift,
            down_block_types=cfg.down_block_types,
            only_cross_attention=cfg.only_cross_attention,
            block_out_channels=cfg.block_out_channels,
            layers_per_block=cfg.layers_per_block,
            downsample_padding=cfg.downsample_padding,
            mid_block_scale_factor=cfg.mid_block_scale_factor,
            act_fn=cfg.act_fn,
            norm_num_groups=cfg.norm_num_groups,
            norm_eps=cfg.norm_eps,
            cross_attention_dim=cfg.cross_attention_dim,
            attention_head_dim=cfg.attention_head_dim,
            use_linear_projection=cfg.use_linear_projection,
            class_embed_type=cfg.class_embed_type,
            num_class_embeds=cfg.num_class_embeds,
            upcast_attention=cfg.upcast_attention,
            resnet_time_scale_shift=cfg.resnet_time_scale_shift,
            projection_class_embeddings_input_dim=cfg.projection_class_embeddings_input_dim,
            controlnet_conditioning_channel_order=controlnet_conditioning_channel_order,
            conditioning_embedding_out_channels=conditioning_embedding_out_channels,
        )

        self.conv_in.load_state_dict(pretrained_unet.conv_in.state_dict())
        self.time_proj.load_state_dict(pretrained_unet.time_proj.state_dict())
        self.time_embedding.load_state_dict(pretrained_unet.time_embedding.state_dict())

        if self.class_embedding:
            self.class_embedding.load_state_dict(
                pretrained_unet.class_embedding.state_dict()
            )

        self.down_blocks.load_state_dict(pretrained_unet.down_blocks.state_dict())
        self.mid_block.load_state_dict(pretrained_unet.mid_block.state_dict())

        self.encoder_projector = None
        if cross_attention_dim != cfg.cross_attention_dim:
            self.encoder_projector = nn.Linear(
                cross_attention_dim, cfg.cross_attention_dim
            )

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.FloatTensor,
        conditioning_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guess_mode: bool = False,
        return_dict: bool = True,
    ) -> Union[ControlNetOutput, Tuple]:
        """
        Forward pass of the model.

        Args:
            sample (torch.FloatTensor): Input tensor.
            timestep (Union[torch.Tensor, float, int]): Time step.
            encoder_hidden_states (torch.Tensor): Encoder hidden states.
            controlnet_cond (torch.FloatTensor): Controlnet conditioning tensor.
            conditioning_scale (float, optional): Scale factor for conditioning. Defaults to 1.0.
            class_labels (Optional[torch.Tensor], optional): Class labels. Defaults to None.
            timestep_cond (Optional[torch.Tensor], optional): Time step conditioning tensor. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Attention mask tensor. Defaults to None.
            cross_attention_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for cross attention. Defaults to None.
            guess_mode (bool, optional): Whether guess mode is enabled. Defaults to False.
            return_dict (bool, optional): Whether to return a dictionary. Defaults to True.

        Returns:
            Union[ControlNetOutput, Tuple]: ControlNetOutput instance or a tuple.
        """
        if self.encoder_projector:
            encoder_hidden_states = self.encoder_projector(encoder_hidden_states)
        output = super().forward(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            conditioning_scale=conditioning_scale,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            guess_mode=guess_mode,
            return_dict=return_dict,
        )

        return output


class OveriddenUnet2D(UNet2DConditionModel):
    """
    Overwriting certain operations of the original `UNet2DConditionModel` so that
    pretrained `UNet2DConditionModel` can be loaded.

    Args:
        pretrained_unet (UNet2DConditionModel): Pretrained UNet2DConditionModel instance.
        encoder_hid_dim (int, optional): Dimensionality of encoder hidden states. Defaults to None.
    """

    def __init__(
        self,
        pretrained_unet: UNet2DConditionModel,
        encoder_hid_dim: int,
    ):
        super().__init__(**pretrained_unet.config)
        self.load_state_dict(pretrained_unet.state_dict())
        self.encoder_projector = None
        if encoder_hid_dim and encoder_hid_dim != self.config.cross_attention_dim:
            self.encoder_projector = nn.Linear(
                encoder_hid_dim, self.config.cross_attention_dim
            )

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        timestep_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        """
        Forward pass of the model.

        Args:
            sample (torch.FloatTensor): Input tensor.
            timestep (Union[torch.Tensor, float, int]): Time step.
            encoder_hidden_states (torch.Tensor): Encoder hidden states.
            class_labels (Optional[torch.Tensor], optional): Class labels. Defaults to None.
            timestep_cond (Optional[torch.Tensor], optional): Time step conditioning tensor. Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Attention mask tensor. Defaults to None.
            cross_attention_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments for cross attention. Defaults to None.
            down_block_additional_residuals (Optional[Tuple[torch.Tensor]], optional): Additional residuals for down blocks. Defaults to None.
            mid_block_additional_residual (Optional[torch.Tensor], optional): Additional residual for the mid block. Defaults to None.
            return_dict (bool, optional): Whether to return a dictionary. Defaults to True.

        Returns:
            Union[UNet2DConditionOutput, Tuple]: UNet2DConditionOutput instance or a tuple.
        """
        if self.encoder_projector:
            encoder_hidden_states = self.encoder_projector(encoder_hidden_states)
        output = super().forward(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            timestep_cond=timestep_cond,
            attention_mask=attention_mask,
            cross_attention_kwargs=cross_attention_kwargs,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            return_dict=return_dict,
        )
        return output.sample


class StainFuserArchitecture(nn.Module):
    """Wrapper for all architecture definitions."""

    def __init__(
        self,
        conditioner: ControlNetModel,
        encoder: AutoencoderKL,
        noise_scheduler: PNDMScheduler,
        noise_predictor: UNet2DConditionModel,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.conditioner = conditioner
        self.noise_scheduler = noise_scheduler
        self.noise_predictor = noise_predictor
