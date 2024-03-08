from typing import Optional
import os
import torch
import colored

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    PNDMScheduler,
)

from models.arch import OveriddenUnet2D, OveriddenControlNet, StainFuserArchitecture
from src.misc.utils import (
    load_json,
)


def load_stainFuser(
    pretrained: Optional[str] = None, config_dir: str = "path/to/config/dir"
) -> StainFuserArchitecture:
    """
    Load the StainFuser model with the given pretrained weights and configurations.

    Args:
        pretrained (Optional[str]): Path to the pretrained model weights. If None, the model is initialized with random weights.
        config_dir (str): Path to the directory containing the configuration files for the model's components.

    Returns:
        StainFuserArchitecture: The loaded StainFuser model.
    """
    vae_config = load_json(f"{config_dir}/sd21b_vae_config.json")
    unet_config = load_json(f"{config_dir}/sd21b_unet_config.json")
    schedule_config = load_json(f"{config_dir}/sd21b_scheduler_config.json")

    unet = UNet2DConditionModel(**unet_config)

    noise_predictor = OveriddenUnet2D(unet, encoder_hid_dim=4)
    controlnet = OveriddenControlNet(
        unet,
        cross_attention_dim=4,
        conditioning_embedding_in_channels=4,
    )
    vae = AutoencoderKL(**vae_config)
    noise_scheduler = PNDMScheduler(**schedule_config)

    model = StainFuserArchitecture(
        encoder=vae,
        conditioner=controlnet,
        noise_predictor=noise_predictor,
        noise_scheduler=noise_scheduler,
    )

    if pretrained is None:
        return model

    if os.path.exists(pretrained):
        print(f"Loading: {pretrained}")
        pretrained = torch.load(pretrained, map_location=torch.device("cpu"))
        pretrained = convert_pytorch_checkpoint(pretrained)
        (missing_keys, unexpected_keys) = model.load_state_dict(
            pretrained, strict=False
        )
        print("missing keys: ", missing_keys)
        print("unexpected keys: ", unexpected_keys)
    else:
        assert os.path.exists(pretrained), f"Pretrained model not found: {pretrained}"
    return model


def convert_pytorch_checkpoint(net_state_dict):
    variable_name_list = list(net_state_dict.keys())
    is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
    if is_in_parallel_mode:
        colored_word = colored("WARNING", color="red", attrs=["bold"])
        print(
            (
                f"{colored_word}: Detect checkpoint saved in data-parallel mode."
                " Converting saved model to single GPU mode."
            ).rjust(80)
        )
        net_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
        }
    return net_state_dict
